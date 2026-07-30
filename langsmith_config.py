"""
langsmith_config.py
───────────────────
LangSmith tracing configuration.

LangSmith provides visual debugging for LangGraph agent flows — you can
see exactly which node ran, what it returned, and how long each step took.

This module is designed to **gracefully degrade**: if the API key is not
set, tracing is silently disabled and the application runs normally.

Setup:
  1. Create a free account at https://smith.langchain.com
  2. Generate an API key
  3. Add to your .env file:
       LANGCHAIN_TRACING_V2=true
       LANGCHAIN_API_KEY=<your-key>
       LANGCHAIN_PROJECT=smart-contract-audit
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def configure_langsmith() -> bool:
    """
    Configures LangSmith tracing via environment variables.

    LangChain/LangGraph automatically detect these env vars and enable
    tracing when they are set. No code changes in the agents are needed.

    Returns:
        True if tracing was enabled, False if skipped.
    """
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    project = os.getenv("LANGCHAIN_PROJECT", "smart-contract-audit")

    if not tracing_enabled:
        logger.info("[LangSmith] Tracing is disabled. Set LANGCHAIN_TRACING_V2=true in .env to enable.")
        return False

    if not api_key or api_key == "your-langsmith-api-key-here":
        logger.warning(
            "[LangSmith] LANGCHAIN_TRACING_V2 is true but LANGCHAIN_API_KEY "
            "is not set. Disabling tracing to avoid errors."
        )
        # Explicitly disable to prevent LangChain from attempting to connect
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    # Ensure env vars are set for LangChain/LangGraph to pick up
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project

    logger.info(
        "[LangSmith] Tracing ENABLED — project: '{}'. View traces at https://smith.langchain.com",
        project,
    )
    return True
