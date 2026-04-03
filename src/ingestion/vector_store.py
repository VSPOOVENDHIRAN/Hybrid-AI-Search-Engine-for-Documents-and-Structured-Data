import os
import json
import shutil
from langchain_community.vectorstores import FAISS
from src.ingestion.embedding import get_embedding_model

BASE_PATH = "vector_store"
META_FILE = "metadata.json"   # sidecar: { filename: [doc_id, ...] }


# ─── Sidecar helpers ──────────────────────────────────────────────────────────

def _meta_path(user_path: str) -> str:
    return os.path.join(user_path, META_FILE)


def _load_meta(user_path: str) -> dict:
    p = _meta_path(user_path)
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {}


def _save_meta(user_path: str, meta: dict) -> None:
    with open(_meta_path(user_path), "w") as f:
        json.dump(meta, f, indent=2)


def _load_db(user_path: str) -> FAISS:
    embedding_model = get_embedding_model()
    return FAISS.load_local(
        user_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


# ─── Create / update ──────────────────────────────────────────────────────────

def create_or_update_vector_store(chunks, embedding_model, user_id: str, filename: str) -> FAISS:
    """Embed *chunks* and upsert into the user's FAISS index.

    Every chunk must already carry metadata with at least:
        {"filename", "source_path", "user_id", "file type", "chunk_id"}
    """
    for chunk in chunks:
        required_keys = ("user_id", "filename", "source_path", "file type", "chunk_id")
        if not all(k in chunk.metadata for k in required_keys):
            raise ValueError(f"Missing mandatory metadata in chunk. Found keys: {list(chunk.metadata.keys())}")

    user_path = os.path.join(BASE_PATH, user_id)
    index_file = os.path.join(user_path, "index.faiss")
    meta = _load_meta(user_path)

    if os.path.exists(index_file):
        db = _load_db(user_path)
        ids = db.add_documents(chunks)
        print(f"[vector_store] Added {len(ids)} vectors for '{filename}' (existing index)")
    else:
        os.makedirs(user_path, exist_ok=True)
        db = FAISS.from_documents(chunks, embedding_model)
        ids = list(db.docstore._dict.keys())
        print(f"[vector_store] Created new index with {len(ids)} vectors for '{filename}'")

    # Merge IDs (re-upload appends; no duplicates)
    existing = meta.get(filename, [])
    meta[filename] = existing + [i for i in ids if i not in existing]

    _save_meta(user_path, meta)
    db.save_local(user_path)
    return db


# ─── Delete by filename ───────────────────────────────────────────────────────

def delete_from_vector_store(filename: str, user_id: str) -> tuple[int, bool]:
    """Remove all vectors belonging to *filename* from the user's FAISS index.

    Returns:
        (vectors_deleted: int, needs_rebuild: bool)
    """
    user_path = os.path.join(BASE_PATH, user_id)
    index_file = os.path.join(user_path, "index.faiss")
    needs_rebuild = False

    if not os.path.exists(index_file):
        print(f"[vector_store] No index found for user '{user_id}' — nothing to delete.")
        return 0, False

    meta = _load_meta(user_path)
    ids_to_delete: list[str] = meta.pop(filename, [])

    # ── Pass 2: docstore scan fallback ────────────────────────────────────────
    db = _load_db(user_path)

    if not ids_to_delete:
        print(f"[vector_store] '{filename}' not in sidecar — scanning docstore for metadata match…")
        scanned = 0
        for doc_id, doc in db.docstore._dict.items():
            scanned += 1
            doc_meta = getattr(doc, "metadata", {})
            doc_filename = doc_meta.get("filename")
            
            if not doc_meta.get("user_id") or not doc_filename:
                print(f"[vector_store] ⚠ Vector {doc_id} missing mandatory metadata. Flagging for rebuild.")
                needs_rebuild = True
                
            if doc_filename == filename:
                ids_to_delete.append(doc_id)
        print(f"[vector_store] Scanned {scanned} docs, found {len(ids_to_delete)} matching '{filename}'")

    if not ids_to_delete:
        _save_meta(user_path, meta)
        print(f"[vector_store] No vectors found for '{filename}' — index unchanged.")
        return 0, needs_rebuild

    # ── Delete ────────────────────────────────────────────────────────────────
    print(f"[vector_store] Deleting {len(ids_to_delete)} vectors for '{filename}'…")
    
    try:
        db.delete(ids_to_delete)
    except ValueError as e:
        print(f"[vector_store] ⚠ Error during FAISS deletion: {e}. Flagging for rebuild.")
        needs_rebuild = True

    db.save_local(user_path)
    _save_meta(user_path, meta)

    # Verify: confirm IDs are gone from the docstore
    still_present = [i for i in ids_to_delete if i in db.docstore._dict]
    if still_present:
        print(f"[vector_store] ⚠ {len(still_present)} IDs still present after delete — flagging for rebuild.")
        needs_rebuild = True
    else:
        print(f"[vector_store] ✓ All {len(ids_to_delete)} vectors for '{filename}' removed.")

    return len(ids_to_delete), needs_rebuild


# ─── Delete all user data ─────────────────────────────────────────────────────

def delete_user_vector_store(user_id: str) -> int:
    """Wipe the entire FAISS index and sidecar for *user_id*.

    Returns the total number of vectors that were in the index.
    """
    user_path = os.path.join(BASE_PATH, user_id)

    if not os.path.exists(user_path):
        print(f"[vector_store] No vector store directory for '{user_id}' — nothing to wipe.")
        return 0

    # Count vectors before deletion for the log
    try:
        db = _load_db(user_path)
        total = len(db.docstore._dict)
    except Exception:
        total = -1   # index corrupt or unreadable

    shutil.rmtree(user_path)
    print(f"[vector_store] ✓  Wiped vector store for '{user_id}' "
          f"({total if total >= 0 else 'unknown'} vectors removed).")
    return max(total, 0)