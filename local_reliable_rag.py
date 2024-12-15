import logging
from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass
import google.generativeai as genai
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from fastembed import SparseTextEmbedding
from fastembed import LateInteractionTextEmbedding
from qdrant_client.http.models import SparseVector, PointStruct, Distance, VectorParams, SparseVectorParams, SparseIndexParams
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration parameters for document processing."""
    collection_name: str = 'llama-index-test'
    vector_size: int = 768
    # Vector configurations
    dense_model: str = 'models/text-embedding-004'
    sparse_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    # late_model: str = "colbert-ir/colbertv2.0"
    late_model: str = 'jinaai/jina-colbert-v2'
    late_vector_size: int = 128
    # Vector names in collection
    dense_vector_name: str = "gemini-embed"
    sparse_vector_name: str = "text-sparse"
    late_vector_name: str = "colbert"
    #hybrid search params
    fusion_limit: int = 10
    prefetch_limit: int = 20

@dataclass
class ChatConfig:
    """Configuration parameters for Gemini chat."""
    chat_model: str = 'gemini-1.5-flash-002'
    temperature: float = 0.0
    system_prompt: Optional[str] = None

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass

class ChatProcessingError(Exception):
    """Custom exception for chat processing errors."""
    pass

class DocumentProcessor:
    """Handles document processing, embedding, and vector storage."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.client = QdrantClient(path = "qdrant")
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        self.splitter = DoclingNodeParser()
        self.sparse_model = SparseTextEmbedding(
            model_name=self.config.sparse_model
        )
        # Initialize late interaction model once
        self.late_model = LateInteractionTextEmbedding(
            model_name=self.config.late_model
        )
        self.setup_collection()

    def setup_collection(self) -> bool:
        """Initialize Qdrant collection with multiple vector types."""
        try:
            
            self.client.recreate_collection(
                collection_name=self.config.collection_name,
                vectors_config={
                    self.config.dense_vector_name: VectorParams(
                        size=self.config.vector_size,
                        distance=Distance.COSINE,
                    ),
                    self.config.late_vector_name: VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM #similarity metric between multivectors (matrices) used by colbert
                        )
                    )
                },
                sparse_vectors_config={
                    self.config.sparse_vector_name: SparseVectorParams(
                        index=SparseIndexParams()
                    )
                },
            )
            
            logger.info(f"Created collection with vector types {self.config.dense_vector_name} and {self.config.sparse_vector_name}")
            return True

        except UnexpectedResponse as e:
            raise DocumentProcessingError(f"Failed to setup collection: {e}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for document content."""
        try:
            result = genai.embed_content(
                model=self.config.dense_model,
                content=text,
                task_type='RETRIEVAL_DOCUMENT'
            )
            if not result['embedding']:
                raise DocumentProcessingError("Empty dense embedding returned")
            return result['embedding']
        except Exception as e:
            raise DocumentProcessingError(f"Dense embedding generation failed: {e}")

    def get_sparse_vectors(self, text: str) -> tuple[list, list]:
        """Generate sparse vectors for text using fastembed."""
        embeddings = self.sparse_model.embed([text])
        indices, values = zip(*[(
            embedding.indices.tolist(), 
            embedding.values.tolist()
        ) for embedding in embeddings])
        return indices[0], values[0]

    def get_late_interaction_vectors(self, text, type: str):
        """Generate late interaction vectors for text using ColBERT."""

        if type == 'document':
            try:
                embeddings = self.late_model.embed(text)
            except Exception as e:
                logger.error(f"Late interaction document embedding error: {str(e)}")
                raise DocumentProcessingError(f"Late interaction document embedding failed: {e}")
        elif type == 'query':
            try:
                embeddings = self.late_model.query_embed(text)
            except Exception as e:
                logger.error(f"Late interaction query embedding error: {str(e)}")
                raise DocumentProcessingError(f"Late interaction query embedding failed: {e}")
        else:
            logger.error(f"Invalid text type for late interaction query embedding")
            raise DocumentProcessingError(f"Invalid text type for late interaction query embedding")

        embeddings = next(embeddings)
        
        return embeddings

    def process_document(self, file_path: str) -> None:
        """Process document and store vectors with hybrid search support."""
        try:
            if not os.path.exists(file_path):
                raise DocumentProcessingError(f"File not found: {file_path}")

            # Load and parse document in docling json format
            logger.info(f"Processing document: {file_path}")
            docs = DoclingReader(export_type=DoclingReader.ExportType.JSON).load_data(file_path)
            if not docs:
                raise DocumentProcessingError("No content extracted from document")
                
            nodes = self.splitter.get_nodes_from_documents(docs, show_progress=True)
            
            if not nodes:
                raise DocumentProcessingError("No nodes generated from document")
            
            logger.info(f"Generated {len(nodes)} nodes from document")
            logger.info(f"sample node extracted - {nodes[2].get_content()[:50]}")
            
            # Process nodes
            points = []
            for i, node in enumerate(tqdm(nodes)):
                headings = node.metadata.get('headings', [])
                _text = node.text
                if headings and headings[0]:
                    _text = headings[0] + _text
                    
                dense_vector = self.embed_text(_text)
                sparse_indices, sparse_values = self.get_sparse_vectors(_text)
                late_vector = self.get_late_interaction_vectors(_text,'document')

                
                
                point = PointStruct(
                    id= node.id_,
                   vector= {
                        self.config.dense_vector_name: dense_vector,
                        self.config.sparse_vector_name: SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        ),
                        self.config.late_vector_name: late_vector
                    },
                    payload= {
                        'content': _text,
                        'metadata': {
                            'file_name': node.metadata['origin']['filename'],
                        }
                    }
                )

                points.append(point)

            # Store vectors
            if points:
                self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=points
                )
                logger.info(f"Successfully processed and stored {len(points)} vectors")
            else:
                raise DocumentProcessingError("No vectors generated from document")

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise DocumentProcessingError(f"Document processing failed: {str(e)}")

    def query(self, query_text: str) -> List[float]:
        """Generate query vector for similarity search using query-specific embedding."""
        return self.embed_text(query_text)

    def hybrid_search(self, query_text: str, limit: int = 5) -> list:
        """Perform hybrid search using RRF fusion of multiple vector types."""
        try:
            # Generate query vectors
            dense_vector = self.query(query_text)
            sparse_indices, sparse_values = self.get_sparse_vectors(query_text)
            late_vector = self.get_late_interaction_vectors(query_text,'query')
            
            # Prepare prefetch queries
            prefetch = [
                models.Prefetch(
                    query=dense_vector,
                    using=self.config.dense_vector_name,
                    limit=self.config.prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    ),
                    using=self.config.sparse_vector_name,
                    limit=self.config.prefetch_limit,
                )
            ]
            
            # Perform fusion search
            results = self.client.query_points(
                collection_name=self.config.collection_name,
                prefetch=prefetch,
                query=late_vector,
                using=self.config.late_vector_name,
                with_payload=True,
                limit=limit or self.config.fusion_limit,
            )
            
            return results.points

        except Exception as e:
            logger.error(f"Search parameters: {prefetch}")
            raise DocumentProcessingError(f"Hybrid search failed: {e}")

class GeminiChat:
    """Handles chat interactions with Gemini model."""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.model = None
        self.chat = None
        self.generation_config = genai.GenerationConfig(temperature = self.config.temperature)
        
        # Ensure API key is configured
        if not os.getenv('GEMINI_API_KEY'):
            raise ChatProcessingError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

    def start_chat(self, history: List = []) -> None:
        """Initialize a new chat session with optional history."""
        try:
            if self.config.system_prompt:
                self.model = genai.GenerativeModel(
                    model_name=self.config.chat_model,
                    system_instruction=self.config.system_prompt,
                    generation_config=self.generation_config
                )
            else:
                self.model = genai.GenerativeModel(
                    model_name=self.config.chat_model,
                    generation_config=self.generation_config
                )

            self.chat = self.model.start_chat(history=history or [])
            logger.info(f"Chat session initialized successfully with system prompt \n {self.config.system_prompt}")
        
        except Exception as e:
            raise ChatProcessingError(f"Failed to start chat: {e}")

    def send_message(self, user_input: str, context: str = "") -> str:
        """Send message to chat with context from retrieved documents."""
        try:
            if self.chat is None:
                self.start_chat()
            
            prompt = f"Context:\n{context}\n\nUser Question: {user_input}" if context else user_input
            response = self.chat.send_message(prompt)
            return response.text
        
        except Exception as e:
            raise ChatProcessingError(f"Failed to send message: {e}")

    def get_chat_history(self) -> List:
        """Retrieve current chat history."""
        if not self.chat:
            logger.warning("No active chat session")
            return []
        return self.chat.history

    def clear_chat(self) -> None:
        """Reset the chat session."""
        self.chat = self.model = None
        logger.info("Chat session cleared")

# Usage example
if __name__ == "__main__":
    config = ProcessingConfig()
    processor = DocumentProcessor(config)
    processor.process_document('path to file')
    
    # Perform hybrid search
    query = 'your sample question'
    search_results = processor.hybrid_search(query)
    
    # Format context from search results
    context = "\n\n".join([
        f"Content: {result.payload['content']}\n" 
        f"Source: {result.payload['metadata']['file_name']}"
        for result in search_results
    ])
    
    # Initialize chat and get response
    chat_config = ChatConfig(
        system_prompt="""You are a helpful RAG assistant. 
        When user asks you a question, answer only based on given context. 
        If you cant find an answer based on context, reply "I Dont Know".
        """
    )
    chat = GeminiChat(chat_config)
    
    # Get response from Gemini
    response = chat.send_message(query, context=context)
    print(f"\nQuery: {query}")
    print(f"\nResponse: {response}")

# Make classes and configs available for import
__all__ = ['DocumentProcessor', 'GeminiChat', 'ProcessingConfig', 'ChatConfig']
