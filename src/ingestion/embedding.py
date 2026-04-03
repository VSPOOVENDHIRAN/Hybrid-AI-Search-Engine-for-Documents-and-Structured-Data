from langchain_huggingface import HuggingFaceEmbeddings

# Singleton — loaded once per process, reused on every request
_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity-ready
        )
    return _model