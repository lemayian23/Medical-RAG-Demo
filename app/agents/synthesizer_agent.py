"""
Synthesizer Agent - Generates answer from retrieved context using Ollama
"""

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
    for ctx in contexts:
        source = ctx.get("source", "unknown")
        text = ctx.get("text", "")
        context_texts.append(f"Source: {source}\n{text}")

    context_text = "\n\n---\n\n".join(context_texts)

    # Build the prompt
    prompt = SYNTHESIZER_PROMPT.format(
        context=context_text,
        query=query,
    )

    logger.info(f"Synthesizing answer for: {query[:50]}...")

    try:
        # Call Ollama
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.2,
                "num_predict": 512,
                "num_ctx": 2048,
            },
        )

        answer = response.get("message", {}).get("content", "").strip()
        if not answer:
            return "I don't have enough information to answer this question."

        logger.info("Synthesizer produced an answer")
        return answer

    except Exception as e:
        logger.error(f"Error in synthesizer: {e}")
        return "I encountered an error while generating the answer. Please try again."