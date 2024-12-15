import logging
from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass
import google.generativeai as genai
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig
)
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from fastembed import SparseTextEmbedding
from qdrant_client.http.models import SparseVector, PointStruct, Distance, VectorParams, SparseVectorParams, SparseIndexParams
from tqdm import tqdm
import requests

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
    
    # Vector names in collection
    dense_vector_name: str = "gemini-embed"
    sparse_vector_name: str = "text-sparse"
    
    min_relevancy_score: float = 0.3
    fusion_limit: int = 10
    prefetch_limit: int = 20
    
    # Jina.ai Configuration
    jina_model: str = "jina-colbert-v2"
    jina_top_k: int = 5

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
        self.client = QdrantClient(
                            url="your qdrant cloud url", 
                            api_key=os.getenv('QDRANT_API')
                        )
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

        self.sparse_model = SparseTextEmbedding(
            model_name=self.config.sparse_model
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

    def rerank_with_jina(self, query: str, documents: list) -> list:
        """Rerank documents using Jina.ai ColBERT API."""
        try:
            url = 'https://api.jina.ai/v1/rerank'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {os.getenv("JINA_API")}'
            }
            data = {
                "model": self.config.jina_model,
                "query": query,
                "top_n": self.config.jina_top_k,
                "documents": [doc.payload['content'] for doc in documents]
            }
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code != 200:
                raise DocumentProcessingError(f"Jina reranking failed: {response.text}")
                
            reranked = response.json()
            scored_docs = []
            results = reranked["results"]

            for result in results:
                text = result["document"]["text"]
                relevance_score = result["relevance_score"]
                scored_docs.append((text, relevance_score))
            
            # Sort by score and return documents
            results = [doc for doc, _ in sorted(scored_docs, key=lambda x: x[1], reverse=True)]
            return results[:self.config.jina_top_k]
        except Exception as e:
            logger.error(f"Jina reranking failed: {str(e)}")
            return documents  # fall back to original order if reranking fails

    def process_document(self, file_path: str) -> None:
        """Process document and store vectors with hybrid search support."""
        try:
            if not os.path.exists(file_path):
                raise DocumentProcessingError(f"File not found: {file_path}")

            # Load and parse document
            logger.info(f"Processing document: {file_path}")
            reader = DoclingReader()
            docs = reader.load_data(file_path)
            
            if not docs:
                raise DocumentProcessingError("No content extracted from document")
            
            config = LanguageConfig(language="english", spacy_model="en_core_web_lg")
            splitter = SemanticDoubleMergingSplitterNodeParser(
                language_config=config,
                initial_threshold=0.6,
                appending_threshold=0.8,
                merging_range = 1,
                merging_threshold=0.8,
                max_chunk_size=1000,
            )

            nodes = splitter.get_nodes_from_documents(docs, show_progress=True)
            
            if not nodes:
                raise DocumentProcessingError("No nodes generated from document")
            
            logger.info(f"Generated {len(nodes)} nodes from document")
            logger.info(f"sample node extracted - {nodes[2].get_content()[:50]}")
            
            # Process points
            points = []
            for node in tqdm(nodes):
                
                _text = node.text
                dense_vector = self.embed_text(_text)
                sparse_indices, sparse_values = self.get_sparse_vectors(_text)
                
                point = PointStruct(
                    id=node.id_,
                    vector={
                        self.config.dense_vector_name: dense_vector,
                        self.config.sparse_vector_name: SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        )
                    },
                    payload={
                        'content': _text
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
        """Perform hybrid search using vector similarity and reranking."""
        try:
            # Generate query vectors
            dense_vector = self.query(query_text)
            sparse_indices, sparse_values = self.get_sparse_vectors(query_text)
            
            # First stage retrieval using vector similarity and BM42
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
            
            results = self.client.query_points(
                collection_name=self.config.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF,
                ),
                with_payload=True,
                limit=self.config.prefetch_limit,
            )
            
            # Second stage reranking with Jina ColBERT
            reranked_results = self.rerank_with_jina(query_text, results.points)
            return reranked_results[:limit]

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
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
        """Send message to chat with optional context from retrieved documents."""
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
    
    # Initialize chat and get response
    chat_config = ChatConfig(
        system_prompt="""You are a helpful RAG assistant. 
        When user asks you a question, answer only based on given context. 
        If you cant find an answer based on context, reply "I Dont Know".
        """
    )
    chat = GeminiChat(chat_config)
    
    # Get response from Gemini
    response = chat.send_message(query, context=search_results)
    print(f"\nQuery: {query}")
    print(f"\nResponse: {response}")

# Make classes and configs available for import
__all__ = ['DocumentProcessor', 'GeminiChat', 'ProcessingConfig', 'ChatConfig']
