"""
Schemas Module - Pydantic Models for API Requests and Responses
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


# ================================================================
# REQUEST MODELS
# ================================================================

class QueryRequest(BaseModel):
    """Request model for the /ask endpoint."""
    query: str = Field(..., description="The medical question to be answered")
    session_id: Optional[str] = Field(None, description="Optional session ID")


class UploadQueryRequest(BaseModel):
    """Request model for document upload with optional question."""
    question: Optional[str] = Field(None, description="Optional question about the document")


# ================================================================
# RESPONSE MODELS
# ================================================================

class Source(BaseModel):
    """Source citation model."""
    id: int = Field(..., description="Unique identifier for the source")
    text: str = Field(..., description="The actual text from the source")
    source: str = Field(..., description="Source document name or identifier")
    score: float = Field(..., description="Similarity score (0-1)")


class QueryResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str = Field(..., description="The generated answer")
    sources: List[Source] = Field(default_factory=list, description="List of sources")
    grounded: bool = Field(..., description="Whether the answer is grounded")
    route_used: str = Field(..., description="Route used: internal_docs, web_search, or both")
    retry_count: int = Field(default=0, description="Number of retries")


class UploadResponse(BaseModel):
    """Response model for document upload."""
    status: str = Field(..., description="Success or failure")
    filename: str = Field(..., description="Name of uploaded file")
    text_preview: str = Field(..., description="First 500 characters of extracted text")
    structured_data: Optional[Dict] = Field(None, description="Structured data extracted")
    ocr_used: bool = Field(False, description="Whether OCR was used")
    message: Optional[str] = Field(None, description="Additional information")


# ================================================================
# HEALTH & HISTORY MODELS
# ================================================================

class HealthResponse(BaseModel):
    status: str
    vector_count: int
    model_loaded: bool
    version: str


class HistoryResponse(BaseModel):
    query_id: int
    query: str
    answer: str
    grounded: bool
    route_used: str
    timestamp: str


# ================================================================
# INGESTION MODELS
# ================================================================

class IngestRequest(BaseModel):
    folder_path: str = Field(default="./data/raw", description="Path to folder containing medical documents")


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    message: Optional[str] = None