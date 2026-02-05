# Legal Document QnA System (RAG + ScaleDown)

This project is a Legal Document Question Answering system built using a Retrieval-Augmented Generation (RAG) architecture.
It supports multi-document legal analysis with citation extraction, precedent linking, and confidence scoring.

The system uses ScaleDown API to compress large legal documents before embedding in order to reduce token usage and improve latency.

---

## Tech Stack

* Backend: FastAPI (Python)
* Frontend: Next.js
* Vector Database: Pinecone
* LLM & Embeddings: API provided by Intel / GenAI platform
* Compression: ScaleDown API

---

## Key Features

* Upload large legal documents (PDF)
* ScaleDown compression before embedding
* Multi-document retrieval and analysis
* Citation extraction from source documents
* Precedent linking across documents
* Confidence score for each answer
* Compression metrics dashboard (token reduction)

---

## High Level Architecture

User → Next.js Frontend
→ FastAPI Backend
→ ScaleDown API (compression)
→ Embedding API
→ Pinecone Vector Database
→ LLM (answer generation)

---

## Project Structure

```
legal-doc-qna-rag-scaledown
│
├── backend
│   ├── main.py
│   ├── pdf_utils.py
│   ├── scaledown.py
│   ├── token_utils.py
│   └── ...
│
├── frontend
│   └── (Next.js app)
│
└── README.md
```

---

## API Endpoints (planned)

* POST /upload
  Upload legal documents, compress using ScaleDown, chunk, embed and store in vector DB.

* POST /chat
  Ask questions over uploaded legal documents.

* GET /metrics
  Returns compression and token reduction statistics.

---

## Deliverables Covered

* Working web application
* API endpoints for upload and chat
* Vector database with legal document chunks
* ScaleDown compression integration
* Compression metrics dashboard

