from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector
from .database import Base

class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    # Default 768-dim embedding (compatible with Gemini embedding / bge-base)
    embedding = Column(Vector(768), nullable=True)
