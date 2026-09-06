"""
Test Query Script - Quick CLI Test
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import config
from app.core.embeddings import MedCPTEmbeddings
from app.core.vectorstore import FAISSVectorStore
from app.agents import (
    router_agent,
    retriever_agent_internal,
    synthesizer_agent,
    critic_agent,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def test_query(query: str):
    """Test a single query through the pipeline."""
    logger.info("=" * 60)
    logger.info(f"🔍 Testing Query: {query}")
    logger.info("=" * 60)

    # 1. Route
    route = router_agent(query)
    logger.info(f"📍 Route: {route}")

    # 2. Retrieve
    contexts = retriever_agent_internal(query)
    logger.info(f"📚 Retrieved {len(contexts)} contexts")

    if contexts:
        for i, ctx in enumerate(contexts[:2]):
            logger.info(f"   [{i+1}] Source: {ctx.get('source', 'unknown')}")
            logger.info(f"       Text: {ctx.get('text', '')[:100]}...")
            logger.info(f"       Score: {ctx.get('score', 0):.3f}")

    # 3. Synthesize
    answer = synthesizer_agent(query, contexts)
    logger.info(f"💬 Answer: {answer}")

    # 4. Critic
    critique = critic_agent(answer, contexts)
    logger.info(f"🔬 Critic: grounded={critique.get('grounded')}")
    logger.info(f"   Reason: {critique.get('reason')}")

    logger.info("=" * 60)


if __name__ == "__main__":
    # Test queries
    queries = [
        "What are the treatments for HER2-positive breast cancer?",
        "What is the mechanism of action of Tamoxifen?",
        "What are the side effects of Paclitaxel?",
        "What is the weather in Nairobi?",
    ]

    for q in queries:
        test_query(q)
        print("\n")