"""
Router Agent - Decides which source(s) to use
"""

from app.core.config import config
from app.utils.prompts import ROUTER_PROMPT
from app.utils.logger import get_logger

logger = get_logger(__name__)

VALID_ROUTES = {"internal_docs", "web_search", "both"}


def router_agent(query: str) -> str:
    """
    Determine the best route for answering a medical query.

    Args:
        query (str): The user's medical question

    Returns:
        str: "internal_docs", "web_search", or "both"
    """
    logger.info(f"Routing query: {query[:50]}...")

    # Since we don't have an LLM client yet, we'll use a simple rule-based router
    # for now. This will be replaced with LLM-based routing once we integrate it.

    # Keywords that indicate web search needed
    web_keywords = ["news", "latest", "current", "today", "weather", "population", "stock"]

    # Keywords that indicate internal docs
    internal_keywords = [
        "treatment", "cancer", "breast", "tamoxifen", "paclitaxel", "paracetamol",
        "drug", "medicine", "symptom", "diagnosis", "therapy", "oncology", "cardiology",
        "mechanism", "side effect", "guideline", "clinical", "patient", "disease",
        "mutation", "gene", "protein", "inhibitor", "receptor", "her2", "erbb2"
    ]

    query_lower = query.lower()

    # Check for web search keywords
    if any(kw in query_lower for kw in web_keywords):
        # If it has internal keywords too, use both
        if any(kw in query_lower for kw in internal_keywords):
            route = "both"
        else:
            route = "web_search"
    # Check for internal keywords
    elif any(kw in query_lower for kw in internal_keywords):
        route = "internal_docs"
    else:
        # Default to internal_docs for medical queries
        route = "internal_docs"

    logger.info(f"Routed query to: {route}")
    return route