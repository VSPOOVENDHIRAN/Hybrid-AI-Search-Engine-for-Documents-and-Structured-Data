import os
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedding import get_embedding_model
from src.ingestion.vector_store import create_or_update_vector_store
from langchain_community.document_loaders import TextLoader, CSVLoader, Docx2txtLoader
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import pandas as pd
from pypdf import PdfReader

from src.ingestion.column_store import save_columns

def load_xlsx(file_path: str, filename: str, user_id: str) -> list[Document]:
    try:
        df = pd.read_excel(file_path)
        docs = []

        # Special column index document
        column_data = {}
        for col in df.columns:
            column_data[col] = df[col].astype(str).tolist()

        # Save native schema representation to dedicated store
        save_columns(user_id, filename, column_data)

        # Still append string representation for semantic search context if needed
        docs.append(Document(
            page_content=f"COLUMN_INDEX:\n{column_data}",
            metadata={"source": filename, "page": 0, "is_column_index": True}
        ))

        text = df.to_markdown(index=False)
        docs.append(Document(page_content=text, metadata={"source": filename, "page": 1}))
        return docs
    except Exception as e:
        print(f"[loader] Warning: Error reading XLSX with pandas: {e}")
        return []

def load_html(file_path):
    print(f"[loader] Reading HTML file: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    # ❌ remove useless tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.extract()

    text = soup.get_text(separator="\n")

    # clean text
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join([line for line in lines if line])

    if not text:
        raise ValueError("No meaningful text extracted from HTML")

    print(f"[loader] Extracted text length: {len(text)}")
    print(f"[DEBUG] Raw HTML size: {len(html)}")
    print(f"[DEBUG] Extracted text preview:\n{text[:200]}")

    return [Document(page_content=text)]


def load_pdf(file_path):
    print(f"[loader] Processing PDF: {file_path}")

    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if not text.strip():
        print("[loader] No text found in document (OCR not supported in deployment)")
        return []

    print("[INFO] Normal PDF detected")
    return [Document(page_content=text)]

def ingest_document(file_path: str, user_id: str) -> int:
    """Ingest a document into the user's FAISS vector store.

    Metadata attached to every chunk:
        filename : filename (used by retriever filter + deletion scan)
        file type: file extension / type
        user_id  : owner of the index
        chunk_id : unique identifier for the chunk
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    
    print(f"[loader] File received: '{filename}' (user_id: {user_id})")
    print(f"[loader] File type detected: {ext}")

    try:
        # 1. Load document based on extension
        if ext == ".pdf":
            documents = load_pdf(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        elif ext == ".csv":
            try:
                loader = CSVLoader(file_path, encoding='utf-8')
                documents = loader.load()
            except Exception:
                # Fallback to ansi or other encodings if utf-8 fails
                loader = CSVLoader(file_path)
                documents = loader.load()
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        elif ext == ".html":
            documents = load_html(file_path)
        elif ext == ".xlsx":
            documents = load_xlsx(file_path, filename, user_id)

        else:
            print(f"[loader] Unsupported file type: {ext}")
            return 0
        
        print(f"[loader] Text extraction successful for '{filename}'. Found {len(documents)} document pages/sections.")
    
    except Exception as e:
        print(f"[loader] Text extraction failed for '{filename}': {e}")
        return 0

    # Filter out empty documents before chunking
    valid_documents = []
    for doc in documents:
        if len(doc.page_content.strip()) > 0:
            doc.metadata.update({
                "filename": filename,
                "file type": ext.replace(".", "").upper() if ext else "UNKNOWN",
                "user_id": user_id,
                "source_path": file_path, # keep for vector_store compatibility
            })
            valid_documents.append(doc)
            
    if not valid_documents:
        print(f"[loader] No valid text found in '{filename}'.")
        return 0

    # 3. Chunk (metadata is inherited by every child chunk)
    chunks = chunk_documents(valid_documents)
    
    # Filter empty chunks and assign chunk_id
    valid_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunk.page_content.strip()) > 0:
            chunk.metadata['chunk_id'] = f"{filename}_{i}"
            valid_chunks.append(chunk)

    print(f"[loader] {len(valid_chunks)} chunks created for '{filename}'")

    # 4. Embed + store
    try:
        embedding_model = get_embedding_model()
        create_or_update_vector_store(valid_chunks, embedding_model, user_id, filename)
        print(f"[loader] Embedding and storage success for '{filename}'")
    except Exception as e:
        print(f"[loader] Embedding/Storage failed for '{filename}': {e}")
        return 0

    return len(valid_chunks)