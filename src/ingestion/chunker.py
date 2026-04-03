from langchain_core.documents import Document

def chunk_documents(docs, chunk_size=500, overlap=50):
    chunks = []

    for doc in docs:
        text = doc.page_content

        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i+chunk_size]

            if chunk.strip():
                # Retain metadata for vector store
                chunks.append(Document(page_content=chunk, metadata=doc.metadata.copy() if hasattr(doc, 'metadata') else {}))

    print(f"[loader] Total chunks created: {len(chunks)}")
    return chunks