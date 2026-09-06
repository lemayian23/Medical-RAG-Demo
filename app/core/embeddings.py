"""
Embeddings Module - MedCPT (NIH-trained on 255M PubMed pairs)
"""

import logging
from typing import List, Union

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class MedCPTEmbeddings:
    """
    MedCPT Embedding Wrapper for Biomedical Text
    - Query Encoder: For user questions
    - Article Encoder: For documents/corpus
    Trained by NIH on 255M query-article pairs from PubMed.
    """

    def __init__(
        self,
        query_model_name: str = "MedCPT/Query-Encoder",
        article_model_name: str = "MedCPT/Article-Encoder",
        device: str = "cpu",
    ):
        """
        Initialize both Query and Article encoders.

        Args:
            query_model_name: HuggingFace model for queries
            article_model_name: HuggingFace model for articles
            device: "cpu" or "cuda"
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading Query Encoder: {query_model_name}")
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self.query_model = AutoModel.from_pretrained(query_model_name).to(self.device)
        self.query_model.eval()

        logger.info(f"Loading Article Encoder: {article_model_name}")
        self.article_tokenizer = AutoTokenizer.from_pretrained(article_model_name)
        self.article_model = AutoModel.from_pretrained(article_model_name).to(self.device)
        self.article_model.eval()

        self.embedding_dim = 768  # MedCPT uses 768-dim embeddings

    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a single query (user question) into a vector.

        Args:
            text: User question (e.g., "What are the treatments for breast cancer?")

        Returns:
            np.ndarray: 768-dimensional embedding
        """
        return self.encode_queries([text])[0]

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple queries into vectors.

        Args:
            texts: List of user questions

        Returns:
            np.ndarray: Shape (n, 768)
        """
        with torch.no_grad():
            inputs = self.query_tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            outputs = self.query_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def encode_document(self, text: str) -> np.ndarray:
        """
        Encode a single document (article/chunk) into a vector.

        Args:
            text: Document text

        Returns:
            np.ndarray: 768-dimensional embedding
        """
        return self.encode_documents([text])[0]

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple documents into vectors.

        Args:
            texts: List of document texts

        Returns:
            np.ndarray: Shape (n, 768)
        """
        with torch.no_grad():
            inputs = self.article_tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            outputs = self.article_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def embed_text(self, text: str, is_query: bool = True) -> np.ndarray:
        """
        Convenience method to embed text.

        Args:
            text: Text to embed
            is_query: True if user query, False if document

        Returns:
            np.ndarray: 768-dimensional embedding
        """
        if is_query:
            return self.encode_query(text)
        return self.encode_document(text)