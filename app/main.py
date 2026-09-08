"""
Main Application - FastAPI Entrypoint
Medical RAG Demo: Router → Retriever → Synthesizer → Critic
"""

import io
import time
import json
import shutil
from typing import List, Dict, Any, Optional  # ← ADDED Optional here
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import config
from app.core.vectorstore import FAISSVectorStore
from app.core.embeddings import MedCPTEmbeddings
from app.core.ocr import get_ocr, save_uploaded_file

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    Source,
    HealthResponse,
    HistoryResponse,
    UploadResponse,
    UploadQueryRequest,
)

from app.agents import (
    router_agent,
    retriever_agent_internal,
    retriever_agent_web,
    synthesizer_agent,
    critic_agent,
    get_embeddings,
    get_vectorstore,
)
from app.utils.logger import get_logger

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="Medical RAG Demo",
    description="Multi-Agent RAG System for Medical Question Answering",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger(__name__)

# Query history
query_history: List[Dict[str, Any]] = []

# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the system is healthy and ready."""
    try:
        vectorstore = get_vectorstore()
        stats = vectorstore.get_stats()

        return HealthResponse(
            status="ok",
            vector_count=stats["total_vectors"],
            model_loaded=True,
            version="1.0.0",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            vector_count=0,
            model_loaded=False,
            version="1.0.0",
        )


# ============================================================
# QUERY ENDPOINT (The Main Pipeline)
# ============================================================

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Process a medical question through the multi-agent pipeline.
    """
    start_time = time.time()
    query = request.query

    logger.info(f"📥 Received query: {query[:50]}...")

    # STEP 1: ROUTE
    route = router_agent(query)
    logger.info(f"📍 Route: {route}")

    # STEP 2: RETRIEVE
    contexts = []
    if route in ("internal_docs", "both"):
        internal_results = retriever_agent_internal(query)
        contexts.extend(internal_results)
        logger.info(f"📚 Internal retriever: {len(internal_results)} chunks")

    if route in ("web_search", "both"):
        web_results = retriever_agent_web(query)
        contexts.extend(web_results)
        logger.info(f"🌐 Web retriever: {len(web_results)} chunks")

    # STEP 3: SYNTHESIZE (with Critic Retry Loop)
    max_retries = config.MAX_CRITIC_RETRIES
    grounded = True
    retry_count = 0
    answer = ""

    for attempt in range(max_retries):
        answer = synthesizer_agent(query, contexts)
        critique = critic_agent(answer, contexts)
        grounded = critique.get("grounded", True)

        if grounded:
            logger.info(f"✅ Answer grounded")
            break
        else:
            retry_count += 1
            logger.warning(f"⚠️ Answer not grounded, retrying...")

    # STEP 4: BUILD SOURCES
    sources_list = []
    for ctx in contexts:
        if ctx:
            sources_list.append(
                Source(
                    id=ctx.get("id", 0),
                    text=ctx.get("text", ""),
                    source=ctx.get("source", "unknown"),
                    score=ctx.get("score", 0.0),
                )
            )

    # STEP 5: LOG TO HISTORY
    query_history.append({
        "query": query,
        "answer": answer,
        "grounded": grounded,
        "route_used": route,
        "retry_count": retry_count,
        "timestamp": datetime.now().isoformat(),
    })

    if len(query_history) > 100:
        query_history.pop(0)

    elapsed = time.time() - start_time
    logger.info(f"⏱️ Query completed in {elapsed:.2f}s")

    return QueryResponse(
        answer=answer,
        sources=sources_list,
        grounded=grounded,
        route_used=route,
        retry_count=retry_count,
    )


# ============================================================
# DOCUMENT UPLOAD ENDPOINT (WITH OCR)
# ============================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),  # ← Optional is now imported
):
    """
    Upload a medical document (PDF, image, DOCX, TXT).
    - Automatically extracts text using OCR for scanned docs/images.
    - Returns extracted text and structured data.
    """
    logger.info(f"📤 Uploading: {file.filename}")

    # Validate file type
    allowed_extensions = ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp', 'docx', 'txt']
    ext = file.filename.split('.')[-1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    try:
        # Read file
        file_data = await file.read()

        # Save file (optional)
        file_path = save_uploaded_file(file_data, file.filename)
        logger.info(f"📁 File saved: {file_path}")

        # Process with OCR
        ocr = get_ocr()
        result = ocr.process_document(file_data, file.filename)

        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to process document")
            )

        # Log query if question provided
        if question:
            logger.info(f"📝 Question: {question}")

        # Return response
        return UploadResponse(
            status="success",
            filename=file.filename,
            text_preview=result["text"][:500] + ("..." if len(result["text"]) > 500 else ""),
            structured_data=result.get("structured_data"),
            ocr_used=result.get("ocr_used", False),
            message=f"Processed {len(result['text'])} characters",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ============================================================
# UPLOAD AND QUERY (Combine Upload + Query)
# ============================================================

@app.post("/upload-and-query", response_model=QueryResponse)
async def upload_and_query(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """
    Upload a document and immediately ask a question about it.
    Uses OCR if needed.
    """
    logger.info(f"📤 Upload and query: {file.filename}")
    logger.info(f"📝 Question: {question}")

    try:
        # Process file with OCR
        file_data = await file.read()
        ocr = get_ocr()
        result = ocr.process_document(file_data, file.filename)

        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to process document")
            )

        # Get extracted text
        doc_text = result["text"]

        # Create a query that includes the document content
        enhanced_query = f"{question}\n\nContext from document {file.filename}:\n{doc_text[:2000]}"

        # Process through RAG pipeline
        route = router_agent(enhanced_query)
        contexts = []
        if route in ("internal_docs", "both"):
            contexts.extend(retriever_agent_internal(enhanced_query))

        # Generate answer
        max_retries = config.MAX_CRITIC_RETRIES
        grounded = True
        retry_count = 0
        answer = ""

        for attempt in range(max_retries):
            answer = synthesizer_agent(enhanced_query, contexts)
            critique = critic_agent(answer, contexts)
            grounded = critique.get("grounded", True)
            if grounded:
                break
            retry_count += 1

        sources_list = []
        for ctx in contexts:
            if ctx:
                sources_list.append(
                    Source(
                        id=ctx.get("id", 0),
                        text=ctx.get("text", ""),
                        source=ctx.get("source", "unknown"),
                        score=ctx.get("score", 0.0),
                    )
                )

        return QueryResponse(
            answer=answer,
            sources=sources_list,
            grounded=grounded,
            route_used=route,
            retry_count=retry_count,
        )

    except Exception as e:
        logger.error(f"Upload and query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# HISTORY ENDPOINT
# ============================================================

@app.get("/history", response_model=List[HistoryResponse])
async def get_history(limit: int = 10):
    """Get recent query history."""
    recent = query_history[-limit:] if query_history else []
    return [
        HistoryResponse(
            query_id=i,
            query=entry["query"],
            answer=entry["answer"],
            grounded=entry["grounded"],
            route_used=entry["route_used"],
            timestamp=entry["timestamp"],
        )
        for i, entry in enumerate(recent)
    ]


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Medical RAG Demo",
        "version": "1.0.0",
        "description": "Multi-Agent RAG System for Medical Question Answering",
        "endpoints": {
            "/health": "Check system health",
            "/ask": "Ask a medical question (POST)",
            "/upload": "Upload a document (PDF, image, DOCX, TXT) with OCR",
            "/upload-and-query": "Upload a document and ask a question about it",
            "/history": "View query history",
            "/docs": "Swagger API documentation",
        },
    }


# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("🚀 Starting Medical RAG Demo with OCR support...")

    # Check OCR
    try:
        ocr = get_ocr()
        logger.info("📄 OCR initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ OCR not available: {e}")

    # Check vector store
    try:
        vectorstore = get_vectorstore()
        stats = vectorstore.get_stats()
        logger.info(f"📊 Vector store: {stats['total_vectors']} vectors")
        if stats["total_vectors"] == 0:
            logger.warning("⚠️ Vector store is empty! Please run ingest_documents.py first.")
    except Exception as e:
        logger.error(f"❌ Error loading vector store: {e}")

    logger.info("✅ Medical RAG Demo ready!")