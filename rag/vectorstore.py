"""
rag/vectorstore.py
──────────────────
ChromaDB client and collection management.

Provides:
  - get_chroma_client()       → persistent HttpClient pointing at Docker ChromaDB
  - get_or_create_collection() → returns the "standard_clauses" collection
"""

from __future__ import annotations

import os

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8100"))
COLLECTION_NAME: str = "standard_clauses"


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Returns a ChromaDB HttpClient connected to the Docker container.

    The client uses the persistent storage configured in docker-compose.yml.
    """
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    logger.info("ChromaDB client connected at {}:{}", CHROMA_HOST, CHROMA_PORT)
    return client


def get_or_create_collection(
    client: chromadb.ClientAPI | None = None,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Returns the standard_clauses collection, creating it if necessary.

    Args:
        client: Optional pre-existing ChromaDB client. If None, a new one is created.
        collection_name: Name of the collection to retrieve or create.

    Returns:
        chromadb.Collection: The collection ready for adding or querying documents.
    """
    if client is None:
        client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Standard/clean contract clauses for RAG comparison"},
    )
    logger.info(
        "Collection '{}' ready — {} existing documents",
        collection_name,
        collection.count(),
    )
    return collection
