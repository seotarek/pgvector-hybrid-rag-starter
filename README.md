# pgvector Hybrid RAG Starter 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg?logo=postgresql)](https://github.com/pgvector/pgvector)

**A production-ready Enterprise RAG template combining PostgreSQL `pgvector` dense vector embeddings, full-text sparse search, and Reciprocal Rank Fusion (RRF) for accurate context retrieval.**

Developed by **Tarek Mohamed** ([@seotarek](https://github.com/seotarek))

---

## 📌 Architecture

```
User Query: "How does token budgeting prevent LLM out-of-memory errors?"
       │
       ├──► 1. Dense Vector Search (Cosine Similarity <=> pgvector)
       │         Rank 1: Doc A | Rank 2: Doc B | Rank 3: Doc C
       │
       ├──► 2. Sparse Keyword Search (Postgres to_tsvector @@ tsquery)
       │         Rank 1: Doc C | Rank 2: Doc A | Rank 3: Doc D
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Reciprocal Rank Fusion (RRF) Engine                         │
 │ Score = sum( 1 / (60 + Rank) )                              │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
               [ Top Reranked Context for LLM ]
```

---

## 🚀 Quickstart with Docker

```bash
git clone https://github.com/seotarek/pgvector-hybrid-rag-starter.git
cd pgvector-hybrid-rag-starter
docker-compose up -d
```

Access the interactive API docs at:  
👉 **`http://localhost:8000/docs`**

---

## 📜 License
Licensed under the [MIT License](LICENSE).
