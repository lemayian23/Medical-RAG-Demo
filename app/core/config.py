"""
Configuration Module - Loads environment variables
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """
    Central configuration class for the Medical RAG Demo.
    All settings are loaded from environment variables.
    """

    # Embedding Models (Public - No Auth Required)
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
    )
    ARTICLE_EMBEDDING_MODEL_NAME = os.getenv(
        "ARTICLE_EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
    )

    # FAISS Index Paths
    FAISS_INDEX_PATH = os.getenv(
        "FAISS_INDEX_PATH", "./data/index/faiss_index.bin"
    )
    METADATA_PATH = os.getenv(
        "METADATA_PATH", "./data/index/metadata.pkl"
    )

    # LLM (Ollama)
    OLLAMA_MODEL = os.getenv(
        "OLLAMA_MODEL", "llama3.2:1b"
    )

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Database
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "sqlite:///./data/rag.db"
    )

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

    # Retrieval
    TOP_K = int(os.getenv("TOP_K", 5))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))

    # Critic Retries
    MAX_CRITIC_RETRIES = int(os.getenv("MAX_CRITIC_RETRIES", 2))


# Create a single instance for easy import
config = Config()