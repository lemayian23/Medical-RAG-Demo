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
    """
    if not contexts:
        logger.warning("No context available for synthesis")
        return "I don't have enough information in the available sources to answer that."

    # Build context text from top contexts
    context_texts = []
    for ctx in contexts[:3]:
        source = ctx.get("source", "unknown")
        text = ctx.get("text", "")
        if text:
            context_texts.append(f"[Source: {source}]\n{text}")

    if not context_texts:
        return "I don't have enough information in the available sources to answer that."

    context_text = "\n\n---\n\n".join(context_texts)

    # Build prompt
    prompt = SYNTHESIZER_PROMPT.format(
        context=context_text,
        query=query,
    )

    logger.info(f"🔄 Synthesizing: {query[:40]}...")

    try:
        start = time.time()

        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.2,
                "num_predict": 300,
                "num_ctx": 2048,
            },
        )

        elapsed = time.time() - start
        logger.info(f"⏱️ Ollama: {elapsed:.2f}s")

        answer = response.get("message", {}).get("content", "").strip()

        if not answer:
            return "I couldn't generate an answer. Please try rephrasing."

        return answer

    except Exception as e:
        logger.error(f"❌ Synthesizer error: {e}")
        return "I encountered an error while generating the answer. Please try again."