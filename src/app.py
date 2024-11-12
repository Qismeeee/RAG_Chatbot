from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import os

app = FastAPI()
db_path = os.path.abspath("documents.db")


class Query(BaseModel):
    question: str
    session_id: str


class Document(BaseModel):
    file_id: str
    filename: str
    source: str
    chunk_number: int
    doc_id: str
    embedding: bytes


@app.post("/query/")
async def get_api_response(query: Query):
    response_text = f"Received query: {query.question} with session ID: {query.session_id}"
    return {
        "session_id": query.session_id,
        "answer": response_text,
        "model": "mock_model"
    }

# Thêm bản ghi vào bảng


@app.post("/embeddings/")
async def add_document(embedding: Document):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO documents (file_id, filename, source, chunk_number, doc_id, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (embedding.file_id, embedding.filename, embedding.source, embedding.chunk_number, embedding.doc_id, embedding.embedding))
    conn.commit()
    conn.close()
    return {"message": "Document added successfully"}

# Lấy tất cả bản ghi từ bảng


@app.get("/embeddings/")
async def get_all_embeddings():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents')
    rows = cursor.fetchall()
    conn.close()
    return {"Document": rows}

# Lấy một bản ghi theo file_id


@app.get("/embeddings/{file_id}")
async def get_embedding(file_id: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents WHERE file_id = ?', (file_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"embedding": row}
    else:
        raise HTTPException(status_code=404, detail="Document not found")

# Xóa một bản ghi theo file_id


@app.delete("/embeddings/{file_id}")
async def delete_embedding(file_id: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM documents WHERE file_id = ?', (file_id,))
    conn.commit()
    conn.close()
    return {"Message": "Document deleted successfully"}

# Cập nhật một bản ghi


@app.put("/embeddings/{file_id}")
async def update_document(file_id: str, embedding: Document):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE documents
        SET filename = ?, source = ?, chunk_number = ?, doc_id = ?, embedding = ?
        WHERE file_id = ?
    ''', (embedding.filename, embedding.source, embedding.chunk_number, embedding.doc_id, embedding.embedding, file_id))
    conn.commit()
    conn.close()
    return {"Message": "Document updated successfully"}
