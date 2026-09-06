"""
Synthesizer Agent - Generates answer from retrieved context using Ollama
"""

import time
from typing import List, Dict, Any
import ollama

from app.core.config import config
from app.utils.prompts import SYNTHESIZER_PROMPT
from app.utils.logger import get_logger

logger = get_logger(__name__)


def synthesizer_agent(query: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Generate a grounded answer using the retrieved context.

    Args:
        query: The user's medical question
        contexts: List of retrieved chunks from retriever

    Returns:
        str: Grounded answer or "I don't know" message
    """
    if not contexts:
        logger.warning("No context available for synthesis")
        return "I don't have enough information in the available sources to answer that."

    # Build context text
    context_texts = []
    for ctx in contexts[:3]:  # Limit to top 3 contexts
        source = ctx.get("source", "unknown")
        text = ctx.get("text", "")
        if text:
            context_texts.append(f"Source: {source}\n{text}")

    if not context_texts:
        return "I don't have enough information in the available sources to answer that."

    context_text = "\n\n---\n\n".join(context_texts)

    # Build the prompt
    prompt = SYNTHESIZER_PROMPT.format(
        context=context_text,
        query=query,
    )

    logger.info(f"🔄 Synthesizing answer for: {query[:50]}...")
    logger.info(f"📊 Context length: {len(context_text)} chars")

    try:
        # Call Ollama with timeout handling
        start_time = time.time()

        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.2,
                "num_predict": 300,
                "num_ctx": 2048,
            },
        )

        elapsed = time.time() - start_time
        logger.info(f"⏱️ Ollama response time: {elapsed:.2f}s")

        answer = response.get("message", {}).get("content", "").strip()

        if not answer:
            logger.warning("⚠️ Empty response from Ollama")
            return "I couldn't generate an answer. Please try rephrasing your question."

        logger.info(f"✅ Synthesizer produced answer ({len(answer)} chars)")
        return answer

    except Exception as e:
        logger.error(f"❌ Error in synthesizer: {e}")
        return "I encountered an error while generating the answer. Please try again."