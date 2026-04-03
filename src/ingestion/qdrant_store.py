"""
Qdrant vector store — production alternative to FAISS.

Usage:
  1. Run Qdrant locally:
       docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
  2. Set QDRANT_URL in your .env (default: http://localhost:6333)
  3. Replace FAISS calls in loader.py / retriever.py with these functions.

Each user gets their own Qdrant collection: "user_<user_id>".
Deletion is handled natively via payload filter — no sidecar metadata needed.

Dependencies:
    pip install qdrant-client langchain-qdrant
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

_client = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def _collection_name(user_id: str) -> str:
    return f"user_{user_id}"


def create_or_update_qdrant(chunks, embedding_model, user_id: str, filename: str):
    """Upsert chunks into the user's Qdrant collection."""
    from langchain_qdrant import QdrantVectorStore

    # Tag every chunk with source filename for later filtering / deletion
    for chunk in chunks:
        chunk.metadata["source"] = filename

    client = get_qdrant_client()
    col = _collection_name(user_id)

    # Auto-create collection on first write
    existing = [c.name for c in client.get_collections().collections]
    if col not in existing:
        # Infer vector size from the embedding model
        sample_vec = embedding_model.embed_query("test")
        client.create_collection(
            collection_name=col,
            vectors_config=VectorParams(size=len(sample_vec), distance=Distance.COSINE),
        )

    store = QdrantVectorStore(
        client=client,
        collection_name=col,
        embedding=embedding_model,
    )
    store.add_documents(chunks)
    return store


def delete_from_qdrant(filename: str, user_id: str) -> int:
    """Delete all vectors with metadata.source == filename."""
    client = get_qdrant_client()
    col = _collection_name(user_id)

    result = client.delete(
        collection_name=col,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=filename),
                )
            ]
        ),
    )
    return result.status  # "completed" on success


def get_qdrant_retriever(user_id: str, filter_filename: str = None, top_k: int = 5):
    """Return a LangChain retriever backed by Qdrant."""
    from langchain_qdrant import QdrantVectorStore
    from src.ingestion.embedding import get_embedding_model

    embedding_model = get_embedding_model()
    client = get_qdrant_client()
    col = _collection_name(user_id)

    store = QdrantVectorStore(
        client=client,
        collection_name=col,
        embedding=embedding_model,
    )

    search_kwargs: dict = {"k": top_k}
    if filter_filename:
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=filter_filename),
                )
            ]
        )

    return store.as_retriever(search_kwargs=search_kwargs)
