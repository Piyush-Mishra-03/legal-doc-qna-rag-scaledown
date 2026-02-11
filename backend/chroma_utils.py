import chromadb
from chromadb.config import Settings

_client = None


def get_collection():
    global _client

    if _client is None:
        _client = chromadb.Client(
            Settings(
                persist_directory="chroma_db",
                anonymized_telemetry=False
            )
        )

    collection = _client.get_or_create_collection(
        name="legal_rag_chunks"
    )

    return collection
