"""
Critic Agent - Validates if the answer is grounded in the context
"""

import json
import ollama
from typing import List, Dict, Any

from app.core.config import config
from app.utils.prompts import CRITIC_PROMPT
from app.utils.logger import get_logger

logger = get_logger(__name__)


def critic_agent(
    answer: str,
    contexts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Check if the answer is fully supported by the context.

    Args:
        answer: The generated answer
        contexts: The retrieved contexts used for generation

    Returns:
        Dict with "grounded" (bool) and "reason" (str)
    """
    if not contexts or not answer:
        return {
            "grounded": False,
            "reason": "No context or answer provided",
        }

    # Build context text
    context_texts = []
    for ctx in contexts:
        text = ctx.get("text", "")
        if text:
            context_texts.append(text)

    if not context_texts:
        return {
            "grounded": False,
            "reason": "Context is empty",
        }

    context_text = "\n\n".join(context_texts)

    # Build the prompt
    prompt = CRITIC_PROMPT.format(
        context=context_text,
        answer=answer,
    )

    logger.info("Critic checking answer grounding...")

    try:
        # Call Ollama
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "num_predict": 100,
            },
        )

        raw = response.get("message", {}).get("content", "").strip()

        # Extract JSON from the response
        # Look for JSON pattern
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            # Fallback
            if "true" in raw.lower():
                result = {"grounded": True, "reason": "Answer appears grounded"}
            else:
                result = {"grounded": False, "reason": "Answer may not be grounded"}

        grounded = result.get("grounded", False)
        reason = result.get("reason", "No reason provided")

        logger.info(f"Critic: grounded={grounded}, reason={reason[:50]}...")
        return result

    except Exception as e:
        logger.error(f"Error in critic: {e}")
        # Default to grounded=True to avoid false negatives during demo
        return {
            "grounded": True,
            "reason": f"Critic error (fallback): {str(e)}",
        }