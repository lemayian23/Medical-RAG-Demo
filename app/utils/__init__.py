"""
Utils Module - Prompts and Logger
"""

from app.utils.prompts import ROUTER_PROMPT, SYNTHESIZER_PROMPT, CRITIC_PROMPT
from app.utils.logger import get_logger

__all__ = [
    "ROUTER_PROMPT",
    "SYNTHESIZER_PROMPT",
    "CRITIC_PROMPT",
    "get_logger",
]