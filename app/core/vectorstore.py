"""
Vector Store Module - FAISS Index Management
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS Vector Store for efficient similarity search.
    Supports persistent storage to disk.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ):
        """
        Initialize FAISS index.

        Args:
            embedding_dim: Dimension of embeddings (768 for MedCPT)
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Metadata storage: maps index ID -> {text, source, chunk_id}
        self.metadata: List[Dict[str, Any]] = []

        # Initialize index
        self.index = faiss.IndexFlatL2(embedding_dim)
        logger.info(f"FAISS index initialized with dimension {embedding_dim}")

        # Load from disk if exists
        if index_path and os.path.exists(index_path):
            self.load()

    def add_chunks(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        sources: Optional[List[str]] = None,
    ) -> int:
        """
        Add chunks to the vector store.

        Args:
            texts: List of chunk texts
            embeddings: numpy array of shape (n, embedding_dim)
            sources: Optional list of source document names

        Returns:
            int: Number of chunks added
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")

        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)

        # Add to FAISS index
        start_idx = self.index.ntotal
        self.index.add(embeddings)

        # Store metadata
        for i, text in enumerate(texts):
            metadata_entry = {
                "id": start_idx + i,
                "text": text,
                "source": sources[i] if sources else "unknown",
                "chunk_id": i,
            }
            self.metadata.append(metadata_entry)

        logger.info(f"Added {len(texts)} chunks to FAISS index (total: {self.index.ntotal})")

        # Persist to disk if paths are set
        if self.index_path:
            self.save()

        return len(texts)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Embedding of the query (768-dim)
            top_k: Number of results to return
            threshold: Minimum similarity score (distance threshold)

        Returns:
            List of dicts with keys: id, text, source, score
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []

        # Ensure query is float32 and 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue

            # FAISS L2 distance (lower = more similar)
            # Convert to similarity score (0-1)
            similarity = 1.0 / (1.0 + dist)

            if threshold and similarity < threshold:
                continue

            metadata = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append({
                "id": int(idx),
                "text": metadata.get("text", "unknown"),
                "source": metadata.get("source", "unknown"),
                "score": float(similarity),
                "distance": float(dist),
            })

        logger.info(f"Search returned {len(results)} results")
        return results

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        if not self.index_path or not self.metadata_path:
            logger.warning("Index paths not set. Cannot save.")
            return

        # Save FAISS index
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        logger.info(f"FAISS index saved to {self.index_path}")

        # Save metadata
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Metadata saved to {self.metadata_path}")

    def load(self) -> None:
        """Load FAISS index and metadata from disk."""
        if not self.index_path or not self.metadata_path:
            logger.warning("Index paths not set. Cannot load.")
            return

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            logger.info(f"FAISS index loaded from {self.index_path} (total: {self.index.ntotal})")
        else:
            logger.warning(f"FAISS index not found at {self.index_path}")

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Metadata loaded from {self.metadata_path} ({len(self.metadata)} entries)")
        else:
            logger.warning(f"Metadata not found at {self.metadata_path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "metadata_count": len(self.metadata),
        }

    def clear(self) -> None:
        """Clear the index and metadata."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        logger.info("FAISS index cleared")