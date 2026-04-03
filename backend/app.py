import sys
import os
sys.path.append(os.path.abspath("."))

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import shutil

from src.ingestion.loader import ingest_document
from src.ingestion.vector_store import delete_from_vector_store, delete_user_vector_store

# ✅ Wrap startup logic safely
try:
    from src.generation.rag_chain import get_rag_response
    RAG_INITIALIZED = True
except Exception as e:
    print("ERROR DURING STARTUP:", str(e))
    RAG_INITIALIZED = False

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".csv", ".docx", ".xlsx", ".html")

def rebuild_user_index(user_id: str):
    """Re-ingest all active uploads for a user to repair a corrupted FAISS index."""
    print(f"[api] Rebuilding index for user '{user_id}'...")
    delete_user_vector_store(user_id)
    user_folder = os.path.join(UPLOAD_DIR, user_id)
    if os.path.exists(user_folder):
        for f in os.listdir(user_folder):
            if f.lower().endswith(SUPPORTED_EXTENSIONS):
                file_path = os.path.join(user_folder, f)
                ingest_document(file_path, user_id)
    print(f"[api] Rebuild complete for user '{user_id}'.")

app = FastAPI(title="RAG API", version="2.1")

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"


# ─── ROOT ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "API is running"}


# ─── UPLOAD ───────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Header(..., alias="user_id"),
):
    if not file.filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(status_code=400, detail=f"Only {', '.join(SUPPORTED_EXTENSIONS)} files are accepted.")

    user_folder = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = ingest_document(file_path, user_id)

    return {
        "message": "File processed successfully",
        "filename": file.filename,
        "chunks_created": chunks,
    }


# ─── LIST FILES ───────────────────────────────────────────────────────────────
@app.get("/files")
async def get_files(user_id: str = Header(..., alias="user_id")):
    user_folder = os.path.join(UPLOAD_DIR, user_id)

    if not os.path.exists(user_folder):
        return {"files": []}

    files = [f for f in os.listdir(user_folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    return {"files": files}


# ─── DELETE FILE ──────────────────────────────────────────────────────────────
@app.delete("/files/{filename}")
async def delete_file(
    filename: str,
    user_id: str = Header(..., alias="user_id"),
):
    """Delete one uploaded file and all its embeddings from FAISS.

    Steps:
      1. Verify file exists on disk
      2. Remove the file from disk
      3. Remove its vectors (sidecar lookup → docstore scan fallback)
      4. Return how many vectors were purged
    """
    file_path = os.path.join(UPLOAD_DIR, user_id, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    # Step 1: remove file from disk
    os.remove(file_path)
    print(f"[api] Deleted file from disk: {file_path}")

    # Step 2: remove its vectors from FAISS
    vectors_removed, needs_rebuild = delete_from_vector_store(filename, user_id)
    print(f"[api] {vectors_removed} vectors removed for '{filename}' (user: {user_id})")

    if needs_rebuild:
        print(f"[api] Vector store inconsistency detected. Triggering rebuild for user '{user_id}'.")
        rebuild_user_index(user_id)

    return {
        "message": f"'{filename}' and all its embeddings deleted successfully.",
        "vectors_removed": vectors_removed,
        "rebuild_triggered": needs_rebuild,
    }


# ─── DELETE ALL USER DATA ─────────────────────────────────────────────────────
@app.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    user_id_header: str = Header(..., alias="user_id"),
):
    """Wipe ALL data for a user — uploaded files + entire FAISS index.

    Security: the user_id in the URL must match the user_id header to
    prevent accidental cross-user deletion.
    """
    if user_id != user_id_header:
        raise HTTPException(
            status_code=403,
            detail="user_id in URL does not match the authenticated user_id header.",
        )

    deleted_files: list[str] = []
    user_upload_folder = os.path.join(UPLOAD_DIR, user_id)

    # Remove all uploaded files
    if os.path.exists(user_upload_folder):
        for f in os.listdir(user_upload_folder):
            fpath = os.path.join(user_upload_folder, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
                deleted_files.append(f)
        os.rmdir(user_upload_folder)   # remove empty dir
        print(f"[api] Deleted {len(deleted_files)} files for user '{user_id}'")

    # Wipe the entire FAISS vector store
    vectors_removed = delete_user_vector_store(user_id)

    return {
        "message": f"All data for user '{user_id}' has been deleted.",
        "files_deleted": deleted_files,
        "vectors_removed": vectors_removed,
    }


# ─── RESET SYSTEM ─────────────────────────────────────────────────────────────
@app.delete("/reset")
async def reset_user(user_id: str = Header(..., alias="user_id")):
    """Clean rebuild option: wipe uploads and vector store."""
    deleted_files: list[str] = []
    user_upload_folder = os.path.join(UPLOAD_DIR, user_id)

    if os.path.exists(user_upload_folder):
        for f in os.listdir(user_upload_folder):
            fpath = os.path.join(user_upload_folder, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
                deleted_files.append(f)
        os.rmdir(user_upload_folder)

    vectors_removed = delete_user_vector_store(user_id)

    return {
        "message": f"User '{user_id}' reset successfully.",
        "files_deleted": deleted_files,
        "vectors_removed": vectors_removed,
    }


# ─── QUERY ────────────────────────────────────────────────────────────────────
@app.post("/query")
async def query(
    q: str = Query(..., description="Your question"),
    user_id: str = Header(..., alias="user_id"),
    filter_file: str = Query(None, description="Restrict search to this filename"),
    top_k: int = Query(7, ge=1, le=20, description="Number of chunks to retrieve"),
    use_reranker: bool = Query(True, description="Enable cross-encoder reranking"),
):
    """Query the RAG system with optional file filter, top_k, and reranking."""
    if not RAG_INITIALIZED:
        return {"error": "RAG not initialized"}

    try:
        response_payload = get_rag_response(
            query=q,
            user_id=user_id,
            filter_filename=filter_file,
            top_k=top_k,
            use_reranker=use_reranker,
        )
        # `get_rag_response` now natively returns {"answer": "...", "sources": [{"filename": "...", "page": "..."}]}
        return response_payload
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000)