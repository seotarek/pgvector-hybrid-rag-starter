from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
from typing import List, Optional
from .database import get_db, engine, Base
from .models import DocumentRecord
from .services.hybrid_search import reciprocal_rank_fusion

app = FastAPI(
    title="pgvector Hybrid RAG API",
    description="High-performance hybrid vector and full-text search using PostgreSQL pgvector and RRF",
    version="0.1.0"
)

class IngestRequest(BaseModel):
    title: str
    content: str
    embedding: List[float]

class SearchRequest(BaseModel):
    query: str
    query_embedding: List[float]
    top_k: int = 5

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

@app.get("/health")
def health():
    return {"status": "ok", "service": "pgvector-hybrid-rag"}

@app.post("/ingest")
async def ingest_document(req: IngestRequest, db: AsyncSession = Depends(get_db)):
    doc = DocumentRecord(title=req.title, content=req.content, embedding=req.embedding)
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return {"status": "success", "id": doc.id}

@app.post("/search/hybrid")
async def hybrid_search(req: SearchRequest, db: AsyncSession = Depends(get_db)):
    # 1. Dense Cosine Distance Query
    dense_sql = text("""
        SELECT id, title, content, 1 - (embedding <=> :emb) AS similarity
        FROM documents
        ORDER BY embedding <=> :emb
        LIMIT :top_k;
    """)
    dense_res = await db.execute(dense_sql, {"emb": str(req.query_embedding), "top_k": req.top_k * 2})
    dense_docs = [{"id": r.id, "title": r.title, "content": r.content, "dense_score": float(r.similarity)} for r in dense_res]

    # 2. Sparse Text Search
    sparse_sql = text("""
        SELECT id, title, content
        FROM documents
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
        LIMIT :top_k;
    """)
    sparse_res = await db.execute(sparse_sql, {"query": req.query, "top_k": req.top_k * 2})
    sparse_docs = [{"id": r.id, "title": r.title, "content": r.content} for r in sparse_res]

    # 3. Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(dense_docs, sparse_docs)
    return {"results": fused[:req.top_k]}
