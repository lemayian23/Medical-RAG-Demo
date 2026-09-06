"""
Embeddings Module - Using all-MiniLM-L6-v2 (Public, No Auth Required)
"""

import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MedCPTEmbeddings:
    """
    Simple embedding wrapper using all-MiniLM-L6-v2.
    This model is public and does not require authentication.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = 384
        logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_queries([text])[0]

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def encode_document(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_text(self, text: str, is_query: bool = True) -> np.ndarray:
        return self.encode_query(text) if is_query else self.encode_document(text)