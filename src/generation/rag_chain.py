import re
from src.retrieval.retriever import get_retriever, similarity_search_with_scores
from src.retrieval.reranker import rerank
from src.llm.llm_model import get_llm_response


# ─── Query normalization ───────────────────────────────────────────────────────
# Vague single-word or colloquial queries often fail semantic search because they
# don't share vocabulary with the document text.  This step expands them so the
# embedding is closer to the actual content.

_QUERY_EXPANSIONS: dict[str, str] = {
    # People
    r"\bwho\b":               "who is mentioned, what are the names of people",
    r"\bintern(s)?\b":        "intern internship trainee placement",
    r"\bauthor\b":            "author prepared by written by submitted by name",
    r"\bsigned\b":            "signed approved by signature",
    # Dates / times
    r"\bwhen\b":              "when date time period",
    r"\bdate\b":              "date created on submitted on prepared on",
    # Roles / designations
    r"\brole\b":              "role designation position title job",
    r"\bdesignation\b":       "designation title position role",
    # Document fields
    r"\bsubject\b":           "subject topic title regarding",
    r"\bproject\b":           "project assignment task",
    r"\bdepartment\b":        "department division unit team",
}


def normalize_query(query: str) -> str:
    """Expand vague/colloquial terms so the query embeds closer to document text.

    The original query is kept as a prefix so the core intent is preserved;
    expansion terms are appended as supplemental signal.

    Example:
        "who got intern" → "who got intern who is mentioned, what are the names
        of people intern internship trainee placement"
    """
    expansions: list[str] = []
    q_lower = query.lower()
    for pattern, expansion in _QUERY_EXPANSIONS.items():
        if re.search(pattern, q_lower):
            expansions.append(expansion)

    if not expansions:
        return query  # nothing to expand

    expanded = query + " " + " ".join(expansions)
    print(f"[rag_chain] Query expanded: {query!r} → {expanded!r}")
    return expanded


# ── ROUTER: DETECT COLUMN QUERIES AND PROCESS NATIVELY ──

def detect_intent(query: str) -> str:
    q = query.lower()
    if any(word in q for word in ["how many", "count", "number of", "total"]):
        return "count"
    if any(word in q for word in ["average", "mean"]):
        return "avg"
    if any(word in q for word in ["max", "highest"]):
        return "max"
    if any(word in q for word in ["min", "lowest"]):
        return "min"
    return "rag"

def normalize(text) -> str:
    return str(text).strip().lower()

def detect_column(query: str, columns: list) -> str:
    q = normalize(query)
    for col in columns:
        if normalize(col) in q:
            return col
    return None

def detect_value(query: str, column_values: list) -> str:
    q = normalize(query)
    for val in set(column_values):
        val_str = normalize(val)
        if not val_str or val_str == 'nan':
            continue
        if len(val_str) > 2 and val_str in q:
            return val
    return None

def handle_column_query(query: str, user_id: str, intent: str, filter_filename: str = None) -> dict:
    from src.ingestion.column_store import load_columns
    store = load_columns(user_id)
    
    if not store:
        return {"answer": "No tabular data available.", "sources": []}

    files_to_check = {}
    if filter_filename and filter_filename in store:
        files_to_check = {filter_filename: store[filter_filename]}
    else:
        files_to_check = store

    print(f"\\n[COLUMN QUERY] Query: {query}")
    print(f"[COLUMN QUERY] Intent: {intent}")

    detected_col_name = None
    detected_val_name = None
    
    file_results = {}
    total_count = 0

    for fname, data in files_to_check.items():
        column = detect_column(query, list(data.keys()))
        if not column:
            continue
            
        detected_col_name = column
        value = detect_value(query, data[column])
        
        if not value:
            continue
            
        detected_val_name = value
        
        # Determine occurrences for count intent
        count = sum(1 for v in data[column] if normalize(value) in normalize(v))
        if count > 0:
            file_results[fname] = count
            total_count += count
        
    print(f"[COLUMN QUERY] Column detected: {detected_col_name}")
    print(f"[COLUMN QUERY] Value detected: {detected_val_name}")
    print(f"[COLUMN QUERY] Result: {total_count}\\n")

    if not detected_col_name:
        return {"answer": f"Column not found in dataset. Ensure the column name matches the query.", "sources": []}
        
    if not detected_val_name:
        return {"answer": f"Could not determine the specific value to filter on.", "sources": []}

    details = []
    for f, c in file_results.items():
        details.append(f"📄 {f}: {c}")

    answer = f"There are **{total_count}** entries where `{detected_col_name}` is '{detected_val_name}'.\\n\\n"
    if len(file_results) > 1:
        answer += "### Breakdown by file:\\n" + "\\n".join(details)
    elif len(file_results) == 1:
        answer += f"(Source: {list(file_results.keys())[0]})"

    return {"answer": answer, "sources": []}


def get_rag_response(
    query: str,
    user_id: str,
    filter_filename: str = None,
    top_k: int = 7,
    score_threshold: float = 0.0,   # disabled: always return top-k results
    use_reranker: bool = True,
    reranker_top_n: int = 3,
) -> str:
    """Retrieve → (rerank) → generate with query normalization and fallback.

    Pipeline:
      1. Normalize query  — expand vague terms
      2. FAISS retriever  — plain similarity, top-k (no threshold cut)
      3. Fallback         — direct similarity_search if retriever returns []
      4. Reranker         — cross-encoder narrows candidates to best N
      5. LLM              — extraction-style prompt
    """
    intent = detect_intent(query)
    if intent != "rag":
        return handle_column_query(query, user_id, intent, filter_filename)

    # Step 1: normalize
    expanded_query = normalize_query(query)

    # Step 2: FAISS retriever (score_threshold=0.0 → always returns results)
    faiss_k = max(top_k, 10) if use_reranker else top_k

    print(f"[rag_chain] Retrieving top-{faiss_k} for: {query!r}")
    try:
        retriever = get_retriever(
            user_id=user_id,
            filter_filename=filter_filename,
            top_k=faiss_k,
            score_threshold=score_threshold,
        )
        docs = retriever.invoke(expanded_query)
    except Exception as e:
        print(f"[rag_chain] Retriever error: {e}")
        docs = []

    # Step 3: fallback — direct similarity_search (ignores any threshold)
    if not docs:
        print(f"[rag_chain] ⚠ Retriever returned 0 docs — running fallback search…")
        try:
            docs = similarity_search_with_scores(
                user_id=user_id,
                query=expanded_query,
                top_k=faiss_k,
                filter_filename=filter_filename,
            )
        except Exception as e:
            print(f"[rag_chain] Fallback also failed: {e}")
            return {"answer": "Not found in the uploaded documents.", "sources": []}

    # Step 3.5: Strong Retrieval Guard
    secured_docs = []
    rejected_count = 0
    for doc in docs:
        m_user = doc.metadata.get("user_id")
        m_file = doc.metadata.get("filename")
        if not m_user or not m_file:
            print(f"[rag_chain] ⚠ Discarding doc: missing mandatory metadata (user={m_user}, file={m_file})")
            rejected_count += 1
            continue
        if m_user != user_id:
            print(f"[rag_chain] 🚨 Discarding doc: INVALID USER_ID! (expected={user_id}, got={m_user})")
            rejected_count += 1
            continue
        secured_docs.append(doc)
    
    docs = secured_docs
    if rejected_count > 0:
        print(f"[rag_chain] Post-retrieval guard discarded {rejected_count} docs.")

    if not docs:
        return {"answer": "Not found in the uploaded documents.", "sources": []}

    # Step 4: rerank
    if use_reranker and len(docs) > 1:
        print(f"[rag_chain] Reranking {len(docs)} candidates → keeping {reranker_top_n}")
        docs = rerank(query, docs, top_n=reranker_top_n)

    # Step 5: Extract metadata and build strict clean context
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  TOP-{len(docs)} RETRIEVED CHUNKS")
    print(f"{'═'*60}")
    
    unique_sources = []
    seen = set()
    context_parts = []
    
    for i, doc in enumerate(docs):
        filename = doc.metadata.get("filename", "unknown")
        # Ensure page is a string
        page = str(doc.metadata.get("page", "?"))
        user_id_m = doc.metadata.get("user_id", "?")
        
        # Logging only - NOT passed to LLM
        print(f"\n[Chunk {i+1}/{len(docs)}] (User: {user_id_m})")
        print(f"  Source: {filename} p.{page} | Chars: {len(doc.page_content)}")
        print(f"  {sep}")
        print(doc.page_content)
        print(f"  {sep}")
        
        # Clean context for LLM (no debug strings!)
        context_parts.append(doc.page_content)
        
        # Build JSON generic metadata
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            unique_sources.append({"filename": filename, "page": page})

    print(f"\n{'═'*60}\n")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant.

- Use retrieved context for general questions
- For structured/tabular questions, rely on computed results provided by the system
- Do not perform your own counting if system data is given

Context:
{context}

Question: {query}

Answer:"""

    llm_answer = get_llm_response(prompt).strip()
    clean_lower = llm_answer.lower()
    
    if "not found" in clean_lower or clean_lower == "not found in documents.":
        return {
            "answer": "Not found in the uploaded documents.",
            "sources": []
        }

    return {
        "answer": llm_answer,
        "sources": unique_sources
    }