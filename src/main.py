from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from src.preprocessing.preprocessing import process_uploaded_file
from src.database import init_db, insert_document_record, delete_document_record, get_all_documents
from src.chat_interface import generate_answer_stream
from src.embeddings.faiss_index import FaissIndex

import uvicorn

app = FastAPI()


@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "image/png",
        "image/jpeg",
        "audio/mpeg",
        "audio/wav"
    ]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_location = f"data/raw/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Xử lý file và lấy document_id
    document_id = process_uploaded_file(file_location)

    # Lưu thông tin tài liệu vào SQLite
    insert_document_record(document_id, file.filename)

    return {"message": "File uploaded successfully.", "document_id": document_id}


@app.get("/list_documents")
def list_documents():
    documents = get_all_documents()
    return {"documents": documents}


@app.delete("/delete_document/{doc_id}")
def delete_document(doc_id: str):
    success = delete_document_record(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")

    index = FaissIndex(dimension=384)
    index.load_index("models/faiss_index")
    index.remove_document_embeddings(doc_id)
    index.save_index("models/faiss_index")

    return {"message": "Document deleted successfully."}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question")
    session_id = data.get("session_id", None)
    model_name = data.get("model", "gpt-4o-mini")

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    return StreamingResponse(
        generate_answer_stream(question, session_id, model_name),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
