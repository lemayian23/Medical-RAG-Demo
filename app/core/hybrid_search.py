"""
Hybrid Search Module - Combines BM25 (Keyword) + FAISS (Dense)
"""

import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

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
        
        # Build BM25 index if vectorstore has data
        self._build_bm25()

    def _build_bm25(self):
        """Build BM25 index from stored chunks."""
        if not self.vectorstore.metadata:
            logger.warning("No metadata available for BM25")
            return
        
        # Get texts from metadata
        self.bm25_documents = [m.get("text", "") for m in self.vectorstore.metadata if m.get("text")]
        
        if self.bm25_documents:
            tokenized_docs = [doc.split() for doc in self.bm25_documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info(f"BM25 index built with {len(self.bm25_documents)} documents")
        else:
            logger.warning("No texts available for BM25")

    def search(
        self,
        query: str,
        top_k: int = 10,
        weights: tuple = (0.6, 0.4),  # (vector_weight, bm25_weight)
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using RRF (Reciprocal Rank Fusion).
        """
        results = {}
        
        # 1. Vector search (FAISS)
        embeddings = MedCPTEmbeddings()
        query_vec = embeddings.encode_query(query)
        vector_results = self.vectorstore.search(query_vec, top_k=top_k)
        
        # 2. BM25 search (if available)
        bm25_results = []
        if self.bm25_index and query.split():
            tokenized_query = query.split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            # Get top-k BM25 results
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
            bm25_results = [
                {
                    "id": i,
                    "text": self.bm25_documents[i],
                    "score": bm25_scores[i],
                }
                for i in top_indices
            ]
        
        # 3. RRF Fusion
        if vector_results and bm25_results:
            # Normalize scores
            max_vec_score = max([r.get("score", 0) for r in vector_results]) if vector_results else 1
            max_bm25_score = max([r.get("score", 0) for r in bm25_results]) if bm25_results else 1
            
            # Assign ranks
            for rank, r in enumerate(vector_results):
                doc_id = f"vec_{r.get('id')}"
                normalized_score = r.get("score", 0) / max_vec_score if max_vec_score > 0 else 0
                results[doc_id] = {
                    "score": weights[0] * (1 / (rank + 1)),
                    "text": r.get("text", ""),
                    "source": r.get("source", "unknown"),
                    "vector_score": normalized_score,
                    "bm25_score": 0,
                }
            
            for rank, r in enumerate(bm25_results):
                doc_id = f"bm25_{r.get('id')}"
                normalized_score = r.get("score", 0) / max_bm25_score if max_bm25_score > 0 else 0
                if doc_id in results:
                    results[doc_id]["score"] += weights[1] * (1 / (rank + 1))
                    results[doc_id]["bm25_score"] = normalized_score
                else:
                    results[doc_id] = {
                        "score": weights[1] * (1 / (rank + 1)),
                        "text": r.get("text", ""),
                        "source": "bm25",
                        "vector_score": 0,
                        "bm25_score": normalized_score,
                    }
        
        elif vector_results:
            # Only vector results
            for r in vector_results:
                doc_id = f"vec_{r.get('id')}"
                results[doc_id] = {
                    "score": r.get("score", 0),
                    "text": r.get("text", ""),
                    "source": r.get("source", "unknown"),
                    "vector_score": r.get("score", 0),
                    "bm25_score": 0,
                }
        
        # Sort by score
        sorted_results = sorted(results.values(), key=lambda x: x.get("score", 0), reverse=True)
        
        # Add metadata from vector store
        for r in sorted_results:
            # Try to find in vector metadata
            for m in self.vectorstore.metadata:
                if m.get("text") == r.get("text"):
                    r["id"] = m.get("id", 0)
                    if not r.get("source") or r.get("source") == "bm25":
                        r["source"] = m.get("source", "unknown")
                    break
        
        return sorted_results[:top_k]