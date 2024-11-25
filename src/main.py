import json
import uvicorn
from database import init_milvus
from process_data import process_uploaded_file
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    temp_dir = "data/temp_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    file_location = f"{temp_dir}/{file.filename}"

    # Save the uploaded file to the temporary directory
    with open(file_location, "wb") as f:
        f.write(await file.read())

    print("File uploaded: ", file_location)

    # Xử lý file và lấy doc_ids
    doc_ids = process_uploaded_file(file_location)

    return {"message": "File uploaded successfully.", "doc_ids": doc_ids}


@app.get("/list_documents")
def list_documents():
    # Lấy tất cả document_id từ Milvus
    collection = init_milvus()
    all_entities = collection.query(expr="", output_fields=["id"])
    documents = [entity["id"] for entity in all_entities]
    return {"documents": documents}


@app.delete("/delete_document/{doc_id}")
def delete_document(doc_id: str):
    # Xóa document từ Milvus
    delete_embedding(doc_id)
    return {"message": "Document deleted successfully."}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question")
    session_id = data.get("session_id", None)
    model_name = data.get("model", "gpt-3.5-turbo")

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    return StreamingResponse(
        generate_answer_stream(question, session_id, model_name),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8010, reload=True)
