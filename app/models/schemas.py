"""
Schemas Module - Pydantic Models for API Requests and Responses
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


# ================================================================
# REQUEST MODELS
# ================================================================

class QueryRequest(BaseModel):
    """
    Request model for the /ask endpoint.
    """

    query: str = Field(
        ...,
        description="The medical question to be answered",
        example="What are the treatments for HER2-positive breast cancer?"
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for conversation tracking"
    )


# ================================================================
# RESPONSE MODELS
# ================================================================

class Source(BaseModel):
    """
    Source citation model.
    """

    id: int = Field(..., description="Unique identifier for the source")
    text: str = Field(..., description="The actual text from the source")
    source: str = Field(..., description="Source document name or identifier")
    score: float = Field(..., description="Similarity score (0-1)")


class QueryResponse(BaseModel):
    """
    Response model for the /ask endpoint.
    """

    answer: str = Field(
        ...,
        description="The generated answer to the user's question"
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer"
    )
    grounded: bool = Field(
        ...,
        description="Whether the answer is grounded in the retrieved context"
    )
    route_used: str = Field(
        ...,
        description="Route used: internal_docs, web_search, or both"
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries performed by the critic agent"
    )


# ================================================================
# HEALTH & STATUS MODELS
# ================================================================

class HealthResponse(BaseModel):
    """
    Response model for the /health endpoint.
    """

    status: str = Field(..., description="Health status", example="ok")
    vector_count: int = Field(..., description="Number of vectors in FAISS index")
    model_loaded: bool = Field(..., description="Whether the LLM is loaded")
    version: str = Field(..., description="API version")


class HistoryResponse(BaseModel):
    """
    Response model for the /history endpoint.
    """

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
    """
    Request model for document ingestion.
    """

    folder_path: str = Field(
        default="./data/raw",
        description="Path to folder containing medical documents"
    )


class IngestResponse(BaseModel):
    """
    Response model for document ingestion.
    """

    status: str = Field(..., description="Success or failure status")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    message: Optional[str] = Field(None, description="Additional information")