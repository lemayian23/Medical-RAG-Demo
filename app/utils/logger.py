"""
Logging Module - Shared logging setup
"""

import logging
import sys
from app.core.config import config

# Log level mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with consistent formatting.

    Args:
        name: Module name (typically __name__)

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)

    # Only add handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Set level from config
        level = LOG_LEVELS.get(config.LOG_LEVEL, logging.INFO)
        logger.setLevel(level)

    return logger