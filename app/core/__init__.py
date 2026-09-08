"""
Core Module - Config, Embeddings, Vector Store, OCR
"""

from app.core.config import config, Config
from app.core.embeddings import MedCPTEmbeddings
from app.core.vectorstore import FAISSVectorStore

# OCR is optional - handle import gracefully
try:
    from app.core.ocr import MedicalOCR, get_ocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    MedicalOCR = None
    get_ocr = None

__all__ = [
    "config",
    "Config",
    "MedCPTEmbeddings",
    "FAISSVectorStore",
    "MedicalOCR",
    "get_ocr",
    "OCR_AVAILABLE",
]