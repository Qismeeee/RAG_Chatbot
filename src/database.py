import sqlite3
import os

db_path = os.path.abspath("data/documents.db")


def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE,
            filename TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def insert_document_record(doc_id, filename):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO documents (doc_id, filename)
        VALUES (?, ?)
    ''', (doc_id, filename))
    conn.commit()
    conn.close()


def delete_document_record(doc_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM documents WHERE doc_id = ?', (doc_id,))
    conn.commit()
    conn.close()


def get_all_documents():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT doc_id, filename FROM documents')
    documents = cursor.fetchall()
    conn.close()
    return [{"doc_id": doc[0], "filename": doc[1]} for doc in documents]
