"""
Cross-encoder reranker.

After FAISS returns a coarse top-k (e.g. 10), the cross-encoder
scores each (query, passage) pair jointly for much higher precision,
then returns only the best top_n passages.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22 MB, runs on CPU, trained on MS MARCO passage ranking
  - Typically cuts irrelevant chunks without losing relevant ones

Dependency:
    pip install sentence-transformers
"""

from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        print("[reranker] Loading cross-encoder (one-time)...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        print("[reranker] Cross-encoder ready.")
    return _reranker


def rerank(query: str, docs: list, top_n: int = 3) -> list:
    """Rerank *docs* against *query* and return the best *top_n*.

    Args:
        query:  The user question.
        docs:   List of LangChain Document objects from the retriever.
        top_n:  How many to keep after reranking.

    Returns:
        Sorted (best-first) subset of docs, length ≤ top_n.
    """
    if not docs:
        return docs

    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    # Sort by score descending, return top_n
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]
