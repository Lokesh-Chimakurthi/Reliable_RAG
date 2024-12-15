# Hybrid RAG with Gemini and Qdrant

[![Python](https://img.shields.io/badge/python-blue.svg)](https://www.python.org/downloads/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini-orange.svg)](https://ai.google.dev/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-green.svg)](https://qdrant.tech/)

A production-ready Retrieval-Augmented Generation (RAG) system combining:
- Google's Gemini for embeddings and chat responses
- Qdrant's hybrid search capabilities
- ColBERT reranking via Jina.ai or fastembed
- Semantic double-pass merging for optimal chunking

## Key Features

- **Hybrid Search Architecture**
  - Dense embeddings via Gemini
  - Sparse embeddings (BM42)
  - Two-stage retrieval with ColBERT reranking
  
- **Document Processing**
  - Multiple format support (PDF, DOCX, TXT, MD)
  - Semantic double-pass merging chunking
  - Optimized chunk sizes for LLM context

- **Production Ready**
  - Interactive Gradio UI
  - Robust error handling
  - Configurable processing parameters

## Quick Start

### Prerequisites

- Python
- Qdrant Cloud account / Qdrant local setup
- Google AI (Gemini) API key
- Jina.ai API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Lokesh-Chimakurthi/Reliable_RAG.git
cd Reliable_RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

3. Set up environment variables:
```bash
export QDRANT_API="your-qdrant-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export JINA_API="your-jina-api-key"
```

### Configuration

#### Required Code Changes

In `reliable_rag.py`, update the following:

1. Qdrant connection (around line 90):
```python
self.client = QdrantClient(
    url="your-qdrant-cloud-url",  # Replace with your Qdrant Cloud URL
    api_key=os.getenv('QDRANT_API')
)
```

2. Optional: Adjust model configurations in `ProcessingConfig`:
- `dense_model`: Gemini embedding model
- `sparse_model`: Sparse embedding model
- `collection_name`: Qdrant collection name
- `vector_size`: Embedding dimension (default: 768)

3. Optional: Modify chat settings in `ChatConfig`:
- `chat_model`: Gemini model version
- `temperature`: Response temperature
- `system_prompt`: Custom system instructions

### Usage

1. Start the Gradio interface:
```bash
python app.py
```

2. Upload documents (max 5 files at once) through the web interface

3. Ask questions about your documents

## Technical Details

### Document Processing Pipeline

1. Document Loading: Uses DoclingReader for multiple formats
2. Chunking: Semantic double merging with configurable thresholds
3. Embedding Generation:
   - Dense vectors via Gemini
   - Sparse vectors via BM42
4. Vector Storage: Hybrid vectors in Qdrant

### Search Pipeline

1. First-stage Retrieval:
   - Dense similarity search
   - Sparse BM42 search
   - RRF fusion
2. Second-stage Reranking:
   - ColBERT reranking via Jina.ai

## Environment Variables

```bash
QDRANT_API=your-qdrant-api-key
GEMINI_API_KEY=your-gemini-api-key
JINA_API=your-jina-api-key
```

For better understanding of this project, please refer to the [guide](guide.md).