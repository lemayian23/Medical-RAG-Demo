"""
Main Application - FastAPI Entrypoint
Medical RAG Demo: Router → Retriever → Synthesizer → Critic
"""

import time
import json
from typing import List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import config
from app.core.vectorstore import FAISSVectorStore
from app.core.embeddings import MedCPTEmbeddings
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    Source,
    HealthResponse,
    HistoryResponse,
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

# ================================================================
# APP SETUP
# ================================================================

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

# ================================================================
# QUERY HISTORY (In-memory for demo)
# ================================================================

query_history: List[Dict[str, Any]] = []


# ================================================================
# HEALTH ENDPOINT
# ================================================================

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


# ================================================================
# QUERY ENDPOINT (The Main Pipeline)
# ================================================================

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Process a medical question through the multi-agent pipeline.
    
    Pipeline: Router → Retriever(s) → Synthesizer → Critic (with retry)
    """
    start_time = time.time()
    query = request.query
    session_id = request.session_id or "default"

    logger.info(f"📥 Received query: {query[:50]}...")

    # ============================================================
    # STEP 1: ROUTE
    # ============================================================
    route = router_agent(query)
    logger.info(f"📍 Route: {route}")

    # ============================================================
    # STEP 2: RETRIEVE
    # ============================================================
    contexts = []
    sources = []

    if route in ("internal_docs", "both"):
        internal_results = retriever_agent_internal(query)
        contexts.extend(internal_results)
        logger.info(f"📚 Internal retriever: {len(internal_results)} chunks")

    if route in ("web_search", "both"):
        web_results = retriever_agent_web(query)
        contexts.extend(web_results)
        logger.info(f"🌐 Web retriever: {len(web_results)} chunks")

    # ============================================================
    # STEP 3: SYNTHESIZE (with Critic Retry Loop)
    # ============================================================
    max_retries = config.MAX_CRITIC_RETRIES
    grounded = True
    retry_count = 0
    answer = ""

    for attempt in range(max_retries):
        # Generate answer
        answer = synthesizer_agent(query, contexts)
        logger.info(f"💬 Synthesizer attempt {attempt + 1}: answer length {len(answer)} chars")

        # Validate with Critic
        critique = critic_agent(answer, contexts)
        grounded = critique.get("grounded", True)
        reason = critique.get("reason", "No reason provided")

        if grounded:
            logger.info(f"✅ Answer grounded: {reason}")
            break
        else:
            retry_count += 1
            logger.warning(f"⚠️ Answer not grounded (attempt {attempt + 1}): {reason}")
            logger.info("🔄 Retrying synthesis...")

    # If still not grounded after all retries, keep the last answer
    if not grounded:
        logger.warning(f"⚠️ Final answer NOT grounded after {max_retries} attempts")

    # ============================================================
    # STEP 4: BUILD SOURCES
    # ============================================================
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

    # ============================================================
    # STEP 5: LOG TO HISTORY
    # ============================================================
    query_history.append({
        "query": query,
        "answer": answer,
        "grounded": grounded,
        "route_used": route,
        "retry_count": retry_count,
        "timestamp": datetime.now().isoformat(),
        "sources": [{"source": s.source, "score": s.score} for s in sources_list],
    })

    # Limit history to 100 entries
    if len(query_history) > 100:
        query_history.pop(0)

    elapsed = time.time() - start_time
    logger.info(f"⏱️ Query completed in {elapsed:.2f}s")

    # ============================================================
    # STEP 6: RETURN RESPONSE
    # ============================================================
    return QueryResponse(
        answer=answer,
        sources=sources_list,
        grounded=grounded,
        route_used=route,
        retry_count=retry_count,
    )


# ================================================================
# HISTORY ENDPOINT
# ================================================================

@app.get("/history", response_model=List[HistoryResponse])
async def get_history(limit: int = 10):
    """
    Get recent query history.
    
    Args:
        limit: Number of recent queries to return (default: 10)
    """
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


# ================================================================
# ROOT ENDPOINT
# ================================================================

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
            "/history": "View query history",
            "/docs": "Swagger API documentation",
        },
    }


# ================================================================
# SHUTDOWN EVENT
# ================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown."""
    logger.info("🛑 Medical RAG Demo shutting down")


# ================================================================
# STARTUP EVENT
# ================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("🚀 Starting Medical RAG Demo...")

    # Check vector store
    try:
        vectorstore = get_vectorstore()
        stats = vectorstore.get_stats()
        logger.info(f"📊 Vector store: {stats['total_vectors']} vectors")
        if stats["total_vectors"] == 0:
            logger.warning("⚠️ Vector store is empty! Please run ingest_documents.py first.")
    except Exception as e:
        logger.error(f"❌ Error loading vector store: {e}")

    # Check Ollama
    try:
        import ollama
        logger.info(f"🦙 Ollama model: {config.OLLAMA_MODEL}")
    except Exception as e:
        logger.error(f"❌ Ollama not available: {e}")

    logger.info("✅ Medical RAG Demo ready!")