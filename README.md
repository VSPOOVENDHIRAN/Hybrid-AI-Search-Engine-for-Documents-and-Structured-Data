# 🚀 Hybrid AI Search Engine for Documents and Structured Data

An advanced **Retrieval-Augmented Generation (RAG)** system that supports **multi-format document ingestion** and enables **intelligent querying over both unstructured text and structured data (CSV/XLSX)**.

---

## 🧠 Overview

This project is a **hybrid AI-powered document assistant** capable of:

* Understanding multiple file formats (PDF, TXT, DOCX, CSV, XLSX, HTML)
* Performing **semantic search using FAISS**
* Handling **column-level queries (like SQL/analytics)**
* Supporting **row-level contextual Q&A**
* Dynamically routing queries for accurate responses

---

## ✨ Key Features

### 📄 Multi-Format Ingestion

* PDF (text + scanned via OCR-ready design)
* TXT, DOCX
* CSV, XLSX (structured data support)
* HTML parsing

---

### 🔍 Intelligent Retrieval (RAG)

* Uses **FAISS vector database**
* Embeddings via **Sentence Transformers**
* Context-aware answer generation using LLM

---

### 📊 Structured Data Intelligence

* Column-wise queries (e.g., *"How many CSE students?"*)
* Aggregations:

  * Count
  * Average
  * Filtering
* JSON-based column store for accurate computation

---

### ⚡ Smart Query Routing

* Automatically detects:

  * Row-level queries
  * Column-level queries
  * Mixed queries
* Uses **deterministic Python logic** for numerical accuracy (no hallucination)

---

### 🛡️ Metadata & Security

* Strict metadata tagging:

  * filename
  * file_type
  * user_id
  * chunk_id
* Prevents cross-user data leakage

---

### 💻 Clean UI (Frontend)

* Upload documents
* Filter by file
* Chat-based interface
* Custom modals (no browser alerts)

---

## 🏗️ Tech Stack

| Layer           | Technology               |
| --------------- | ------------------------ |
| Backend         | FastAPI                  |
| LLM             | Groq (LLaMA 3 / Mixtral) |
| Embeddings      | Sentence Transformers    |
| Vector DB       | FAISS                    |
| Framework       | LangChain                |
| Data Processing | Pandas                   |
| Frontend        | HTML, CSS, JavaScript    |

---

## 📁 Project Structure

```bash
project/
│
├── backend/              # FastAPI app
├── src/
│   ├── ingestion/        # loaders, chunking, embeddings
│   ├── retrieval/        # retriever, reranker
│   ├── generation/       # RAG chain
│   └── llm/              # LLM integration
│
├── frontend/             # UI
├── data/                 # runtime storage (ignored in git)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone Repo

```bash
git clone https://github.com/YOUR_USERNAME/Hybrid-AI-Search-Engine-for-Documents-and-Structured-Data.git
cd Hybrid-AI-Search-Engine-for-Documents-and-Structured-Data
```

---

### 2. Create Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run Backend

```bash
uvicorn backend.app:app --reload
```

---

### 5. Open Frontend

Open:

```bash
frontend/index.html
```

---

## 🧪 Example Queries

* "What is this document about?"
* "List all students in CSE"
* "How many students are in AI & DS?"
* "Average marks of students"
* "Show records where department is CSE"

---

## 🚀 Deployment

* Backend → Render / Railway
* Frontend → Netlify / Static hosting

> Note: OCR features may require additional configuration in production.

---

## 🎯 Future Improvements

* 🔍 Table extraction from images/PDF
* ☁️ Cloud vector DB (Qdrant / Pinecone)
* 🧠 Better multimodal reasoning
* 🔐 Authentication & multi-user scaling

---

## 👨‍💻 Author

**Poovendhiran V S**
📧 [v.s.poovendhiran2006@gmail.com](mailto:v.s.poovendhiran2006@gmail.com)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!
