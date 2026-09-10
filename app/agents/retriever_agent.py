"""
Retriever Agent - Searches the vector store for relevant context
Uses Hybrid Search (BM25 + FAISS)
"""

from typing import List, Dict, Any

from app.core.config import config
from app.core.embeddings import MedCPTEmbeddings
from app.core.vectorstore import FAISSVectorStore
from app.core.hybrid_search import HybridSearcher, get_hybrid_searcher
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global instances
_embeddings = None
_vectorstore = None
_hybrid_searcher = None


def get_embeddings() -> MedCPTEmbeddings:
    """Singleton pattern for embeddings."""
    global _embeddings
    if _embeddings is None:
        _embeddings = MedCPTEmbeddings()
    return _embeddings


def get_vectorstore() -> FAISSVectorStore:
    """Singleton pattern for vector store."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = FAISSVectorStore(
            embedding_dim=384,
            index_path=config.FAISS_INDEX_PATH,
            metadata_path=config.METADATA_PATH,
        )
    return _vectorstore


def get_hybrid() -> HybridSearcher:
    """Singleton pattern for hybrid searcher."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        vectorstore = get_vectorstore()
        _hybrid_searcher = HybridSearcher(vectorstore)
    return _hybrid_searcher


def retriever_agent_internal(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from the vector store using hybrid search.

    Args:
        query: The user's medical question
        top_k: Number of results to return (uses config if not provided)

    Returns:
        List of dicts with keys: id, text, source, score
    """
    if top_k is None:
        top_k = config.TOP_K

    logger.info(f"Hybrid retriever query: {query[:50]}...")

    # Check if vector store has data
    vectorstore = get_vectorstore()
    stats = vectorstore.get_stats()
    if stats["total_vectors"] == 0:
        logger.warning("Vector store is empty. Please ingest documents first.")
        return []

    # Use hybrid search
    try:
        hybrid = get_hybrid()
        results = hybrid.search(query, top_k=top_k)
        logger.info(f"Hybrid retriever returned {len(results)} chunks")
        return results
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}. Falling back to vector search.")

        # Fallback: vector search only
        embeddings = get_embeddings()
        query_embedding = embeddings.encode_query(query)
        results = vectorstore.search(
            query_embedding,
            top_k=top_k,
            threshold=config.SIMILARITY_THRESHOLD,
        )
        logger.info(f"Fallback vector retriever returned {len(results)} chunks")
        return results


def retriever_agent_web(query: str) -> List[Dict[str, Any]]:
    """
    Web search fallback - placeholder.
    In production, integrate with DuckDuckGo, Tavily, or PubMed API.
    """
    logger.info(f"Web retriever query: {query[:50]}...")
    return [
        {
            "text": f"Web search results for: {query}",
            "source": "web_search_placeholder",
            "id": 999,
            "score": 0.5,
        }
    ]