# Building Reliable RAG System with Semantic Double-Pass Merging and Hybrid Search

## Hybrid Search with BM42

I used Qdrant's BM42 sparse embeddings alongside dense embeddings since they are lightweight compared to SPLADE. When building this RAG project, I want to build a robust search system that could handle both semantic meaning and exact matches. Traditional dense embeddings are great for understanding context but miss exact matches, while classic BM25 keyword search can't grasp semantic relationships. Thats where Qdrant's BM42 comes in place.

My three-way hybrid implementation combines:
1. Dense embeddings (Google)
2. Sparse vectors (Qdrant BM42)
3. Late interaction embeddings (ColBERT)

The combination provides:
- Better accuracy through multi-vector representation
- Enhanced context understanding
- Improved ranking quality through RRF fusion and colBERT

## Semantic Double-Pass Merging: The Heart of RAG

The most crucial component for RAG effectiveness is the chunking strategy. After testing various chunking strategies, I implemented semantic double-pass merging using llamaindex. It works by first using semantic chunking to create initial chunks based on similarity measures. Then, a second pass is performed to merge these initial chunks into larger, more content-rich chunks, even if they are not directly adjacent, as long as they are semantically related.

```python
splitter = SemanticDoubleMergingSplitterNodeParser(
    language_config=config,
    initial_threshold=0.6,
    appending_threshold=0.8,
    merging_range=1,
    merging_threshold=0.8,
    max_chunk_size=1000,
)
```

## Understanding Double-Pass Merging Parameters

The effectiveness of semantic double-pass merging relies heavily on five key parameters:

### 1. initial_threshold (0.6)
* Controls the semantic similarity needed to start a new chunk
* Value range: 0.0 to 1.0
* Lower values (0.6) are ideal for technical content with mixed terminology
* Higher values would create more, smaller chunks
* 0.6 is the sweet spot for balancing chunk size and semantic coherence

### 2. appending_threshold (0.8)
* Determines when to add new sentences to existing chunks
* Value range: 0.0 to 1.0
* Higher value (0.8) ensures strong semantic relationships
* Prevents topic drift within chunks
* Critical for maintaining chunk coherence

### 3. merging_range (1)
* Controls how many chunks ahead to look during second pass
* Value options: 1 or 2
* 1 = look at next chunk only
* 2 = look at next two chunks
* Lower value (1) provides faster processing with good accuracy

### 4. merging_threshold (0.8)
* Sets similarity requirement for combining chunks in second pass
* Value range: 0.0 to 1.0
* High value (0.8) ensures only truly related chunks merge

### 5. max_chunk_size (1000)
* Maximum characters allowed in a single chunk
* Measured in characters
* 1000 optimized for:
  - LLM context window limitations
  - Processing efficiency
  - Memory usage
  - Retrieval accuracy

These parameters work together to create optimal chunks for RAG:
* First pass (initial_threshold + appending_threshold) creates base chunks
* Second pass (merging_range + merging_threshold) intelligently combines them
* max_chunk_size ensures chunks stay manageable

## Why Double-Pass Merging Outperforms Others

1. *Smarter First Pass*:
   - Initial semantic analysis
   - Careful preservation of structure
   - Maintains relationships between segments
   - Identifies natural break points

2. *Intelligent Second Pass*:
   - Merges related chunks
   - Handles special content (formulas, code, quotes)
   - Preserves complex relationships
   - Optimizes chunk size dynamically

## ColBERT Integration for Advanced Reranking

ColBERT's late interaction approach perfectly complements BM42 by enabling:
- Token-level matching between queries and documents
- Support for long sequences (up to 8192 tokens)
- Fine-grained similarity computation

## Supporting Technologies

### FastEmbed
- Unified API for different embedding types
- Built-in optimization and batching
- Production-ready error handling

### Qdrant
- Native multi-vector support
- Efficient hybrid search capabilities
- Production-scale performance
- Qdrant Cloud offers managed service with generous free tier
- Simple API key authentication for accessing qdrant cloud

### Docling
- Comprehensive format support (PDF, DOCX, PPT, Excel, Images)
- Document structure preservation
- Native LlamaIndex integration

### Google Gemini
- Powers the RAG's response generation
- Low latency responses with gemini-1.5-flash-002
- Handles context well with 1M token context window
- Great at following system prompts and instructions consistently

## Results and Impact

This architecture consistently outperforms traditional approaches:
- More precise matching of technical content
- Better handling of complex queries
- Improved context retention
- Efficient resource utilization

The combination of semantic double-pass merging with hybrid BM42 search with ColBERT reranking has been perfect in my experience. Next stop - Vision RAG