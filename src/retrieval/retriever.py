import os
from langchain_community.vectorstores import FAISS
from src.ingestion.embedding import get_embedding_model

BASE_PATH = "vector_store"


def _load_db(user_id: str) -> FAISS:
    """Load the FAISS vectorstore for a user. Raises FileNotFoundError if missing."""
    user_path = os.path.join(BASE_PATH, user_id)

    if not os.path.exists(os.path.join(user_path, "index.faiss")):
        raise FileNotFoundError(
            f"No vector store found for user '{user_id}'. "
            "Please upload a document first."
        )

    embedding_model = get_embedding_model()
    return FAISS.load_local(
        user_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def _build_filter(user_id: str, filter_filename: str = None) -> dict:
    """Build the FAISS metadata filter dict.

    Always enforces user_id so a user can never retrieve another user's chunks,
    even if index files were somehow mixed.  Optionally narrows further to one file.

    Note: LangChain's FAISS filter performs an exact-match Python-level scan
    of the docstore — all keys in the dict must match the chunk's metadata.
    Old chunks without 'user_id' in metadata will be excluded (correct behaviour:
    they are effectively invisible until re-indexed).
    """
    f: dict = {"user_id": user_id}
    if filter_filename:
        f["filename"] = filter_filename
    return f


def get_retriever(
    user_id: str,
    filter_filename: str = None,
    top_k: int = 7,
    score_threshold: float = 0.0,
):
    """Return a LangChain retriever locked to *user_id*.

    The metadata filter guarantees:
      - Only this user's chunks are returned (user_id match)
      - Optionally: only chunks from one specific file (source match)
      - No cross-user leakage even if the same FAISS binary were shared

    score_threshold=0.0 → plain similarity, always returns top-k results.
    """
    db = _load_db(user_id)

    search_kwargs: dict = {
        "k":      max(top_k, 20),
        "filter": _build_filter(user_id, filter_filename),
    }

    if score_threshold > 0.0:
        search_kwargs["score_threshold"] = score_threshold
        search_type = "similarity_score_threshold"
    else:
        search_type = "similarity"

    print(f"[retriever] filter={search_kwargs['filter']}  k={top_k}  mode={search_type}")
    return db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def similarity_search_with_scores(
    user_id: str,
    query: str,
    top_k: int = 7,
    filter_filename: str = None,
) -> list:
    """Direct FAISS similarity search — fallback when retriever returns [].

    Same user_id filter is applied here too, so the fallback is equally secure.
    Logs similarity scores to help debug retrieval quality.
    """
    db = _load_db(user_id)
    kwargs: dict = {
        "k":      top_k,
        "filter": _build_filter(user_id, filter_filename),
    }

    results = db.similarity_search_with_score(query, **kwargs)

    print(f"[retriever] fallback returned {len(results)} docs:")
    for doc, score in results:
        src = doc.metadata.get("source", "?")
        pg  = doc.metadata.get("page",   "?")
        uid = doc.metadata.get("user_id", "?")
        print(f"  score={score:.4f}  user={uid}  [{src} p.{pg}]  "
              f"{doc.page_content[:80].replace(chr(10), ' ')!r}…")

    return [doc for doc, _ in results]