"""
Hybrid Search Module - Combines BM25 (Keyword) + FAISS (Dense)
Uses Reciprocal Rank Fusion (RRF) to merge results
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

from app.core.vectorstore import FAISSVectorStore
from app.core.embeddings import MedCPTEmbeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HybridSearcher:
    """
    Hybrid search combining BM25 and FAISS.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """

    def __init__(self, vectorstore: FAISSVectorStore):
        self.vectorstore = vectorstore
        self.bm25_index = None
        self.bm25_documents = []
        self.embeddings = None

        # Build BM25 index
        self._build_bm25()

    def _build_bm25(self):
        """Build BM25 index from stored chunks."""
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available. Install rank_bm25: pip install rank-bm25")
            return

        if not self.vectorstore.metadata:
            logger.warning("No metadata available for BM25")
            return

        # Get texts from metadata
        self.bm25_documents = [
            m.get("text", "") for m in self.vectorstore.metadata if m.get("text")
        ]

        if self.bm25_documents:
            tokenized_docs = [doc.lower().split() for doc in self.bm25_documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info(f"BM25 index built with {len(self.bm25_documents)} documents")
        else:
            logger.warning("No texts available for BM25")

    def _get_embeddings(self) -> MedCPTEmbeddings:
        """Lazy load embeddings."""
        if self.embeddings is None:
            self.embeddings = MedCPTEmbeddings()
        return self.embeddings

    def search(
        self,
        query: str,
        top_k: int = 5,
        weights: tuple = (0.6, 0.4),  # (vector_weight, bm25_weight)
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using RRF (Reciprocal Rank Fusion).
        """
        if not query or not query.strip():
            return []

        results = {}

        # ============================================================
        # 1. VECTOR SEARCH (FAISS)
        # ============================================================
        vector_results = []
        try:
            embeddings = self._get_embeddings()
            query_vec = embeddings.encode_query(query)
            vector_results = self.vectorstore.search(query_vec, top_k=top_k)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

        # ============================================================
        # 2. BM25 SEARCH
        # ============================================================
        bm25_results = []
        if self.bm25_index and query.strip():
            try:
                tokenized_query = query.lower().split()
                bm25_scores = self.bm25_index.get_scores(tokenized_query)
                top_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )[:top_k]
                bm25_results = [
                    {
                        "id": i,
                        "text": self.bm25_documents[i],
                        "score": float(bm25_scores[i]),
                    }
                    for i in top_indices
                    if bm25_scores[i] > 0
                ]
            except Exception as e:
                logger.error(f"BM25 search failed: {e}")

        # ============================================================
        # 3. RRF FUSION
        # ============================================================
        if vector_results and bm25_results:
            # Normalize scores
            max_vec_score = max([r.get("score", 0) for r in vector_results]) or 1
            max_bm25_score = max([r.get("score", 0) for r in bm25_results]) or 1

            # Process vector results
            for rank, r in enumerate(vector_results):
                doc_key = self._get_doc_key(r)
                normalized_score = r.get("score", 0) / max_vec_score if max_vec_score > 0 else 0
                results[doc_key] = {
                    "score": weights[0] * (1 / (rank + 1)),
                    "text": r.get("text", ""),
                    "source": r.get("source", "unknown"),
                    "id": r.get("id", 0),
                    "vector_score": normalized_score,
                    "bm25_score": 0,
                    "rrf_score": weights[0] * (1 / (rank + 1)),
                }

            # Process BM25 results
            for rank, r in enumerate(bm25_results):
                doc_key = self._get_doc_key(r)
                normalized_score = r.get("score", 0) / max_bm25_score if max_bm25_score > 0 else 0
                if doc_key in results:
                    results[doc_key]["score"] += weights[1] * (1 / (rank + 1))
                    results[doc_key]["bm25_score"] = normalized_score
                    results[doc_key]["rrf_score"] += weights[1] * (1 / (rank + 1))
                else:
                    results[doc_key] = {
                        "score": weights[1] * (1 / (rank + 1)),
                        "text": r.get("text", ""),
                        "source": "bm25",
                        "id": r.get("id", 0),
                        "vector_score": 0,
                        "bm25_score": normalized_score,
                        "rrf_score": weights[1] * (1 / (rank + 1)),
                    }

        elif vector_results:
            # Only vector results
            max_vec_score = max([r.get("score", 0) for r in vector_results]) or 1
            for rank, r in enumerate(vector_results):
                doc_key = self._get_doc_key(r)
                results[doc_key] = {
                    "score": r.get("score", 0) / max_vec_score if max_vec_score > 0 else 0,
                    "text": r.get("text", ""),
                    "source": r.get("source", "unknown"),
                    "id": r.get("id", 0),
                    "vector_score": r.get("score", 0) / max_vec_score if max_vec_score > 0 else 0,
                    "bm25_score": 0,
                    "rrf_score": r.get("score", 0) / max_vec_score if max_vec_score > 0 else 0,
                }

        elif bm25_results:
            # Only BM25 results
            max_bm25_score = max([r.get("score", 0) for r in bm25_results]) or 1
            for rank, r in enumerate(bm25_results):
                doc_key = self._get_doc_key(r)
                results[doc_key] = {
                    "score": r.get("score", 0) / max_bm25_score if max_bm25_score > 0 else 0,
                    "text": r.get("text", ""),
                    "source": "bm25",
                    "id": r.get("id", 0),
                    "vector_score": 0,
                    "bm25_score": r.get("score", 0) / max_bm25_score if max_bm25_score > 0 else 0,
                    "rrf_score": r.get("score", 0) / max_bm25_score if max_bm25_score > 0 else 0,
                }

        # Sort by RRF score
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        # Attach metadata from vector store
        for r in sorted_results:
            for m in self.vectorstore.metadata:
                if m.get("text") == r.get("text"):
                    r["id"] = m.get("id", r.get("id", 0))
                    if not r.get("source") or r.get("source") == "bm25":
                        r["source"] = m.get("source", "unknown")
                    break

        return sorted_results[:top_k]

    def _get_doc_key(self, result: Dict) -> str:
        """Generate a unique key for a document."""
        return f"doc_{result.get('id', 0)}_{hash(result.get('text', '')) % 10000}"


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_hybrid_searcher_instance = None


def get_hybrid_searcher() -> Optional[HybridSearcher]:
    """Singleton pattern for hybrid searcher."""
    global _hybrid_searcher_instance
    if _hybrid_searcher_instance is None:
        try:
            from app.agents.retriever_agent import get_vectorstore
            vectorstore = get_vectorstore()
            _hybrid_searcher_instance = HybridSearcher(vectorstore)
        except Exception as e:
            logger.error(f"Failed to initialize hybrid searcher: {e}")
            return None
    return _hybrid_searcher_instance