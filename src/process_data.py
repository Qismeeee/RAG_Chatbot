import os
import uuid
import shutil
import json
import openai
import hashlib
from dotenv import load_dotenv
from preprocessing.docsLoader import langchain_document_loader
from preprocessing.chunking import chunk_documents
from sentence_transformers import SentenceTransformer
from database import insert_embeddings, delete_embedding, collection

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

PROCESSED_FILES_PATH = "data/processed_files.json"


def compute_file_hash(file_path):
    """Tính toán hàm băm của tệp để xác định duy nhất."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()


def load_processed_files():
    """Tải danh sách các tệp đã xử lý từ tệp JSON."""
    print("LOAD processed files: ")
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_processed_files(processed_files):
    """Lưu danh sách các tệp đã xử lý vào tệp JSON."""
    with open(PROCESSED_FILES_PATH, 'w') as f:
        json.dump(processed_files, f, indent=4)


def correct_text_with_gpt(text):
    """Sửa lỗi chính tả, ký tự và ngữ pháp bằng GPT."""
    messages = [
        {
            "role": "system",
            "content": "Bạn là một trợ lý ngôn ngữ tiếng Việt, hãy sửa lỗi chính tả, ký tự và ngữ pháp cho đoạn văn sau."
        },
        {"role": "user", "content": text}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=2048,
        temperature=0.2
    )

    corrected_text = response.choices[0].message['content'].strip()
    return corrected_text


def process_file(file_path, output_dir):
    """Xử lý một tệp duy nhất: sửa lỗi và lưu lại."""
    base_filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, base_filename)

    if file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        corrected_content = correct_text_with_gpt(content)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(corrected_content)
        print(f"Đã sửa lỗi và lưu {output_path}")

    elif file_path.endswith(".json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            content = data["page_content"]
            metadata = data["metadata"]
        corrected_content = correct_text_with_gpt(content)

        output_data = {
            "page_content": corrected_content,
            "metadata": metadata
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"Đã sửa lỗi và lưu {output_path}")


def correct_all_files(input_dir, output_dir):
    """Sửa lỗi cho tất cả các tệp trong thư mục đầu vào."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.endswith(".txt") or filename.endswith(".json"):
            process_file(file_path, output_dir)


def process_uploaded_file(file_path):
    # processed_files = load_processed_files()
    processed_files = {}
    file_hash = compute_file_hash(file_path)
    if file_hash in processed_files and processed_files[file_hash].get("chunked", False):
        print(f"Tệp {file_path} đã được chunking trước đó. Bỏ qua bước chunking.")
    else:
        temp_dir = "data/temp_processing"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        filename = os.path.basename(file_path)
        temp_file_path = os.path.join(temp_dir, filename)
        shutil.copy(file_path, temp_file_path)

        # Bước 1: Trích xuất văn bản từ tệp
        # langchain_document_loader(temp_dir)

        # Bước 2: Sửa lỗi chính tả và ngữ pháp
        # input_dir = "data/processed"
        # output_dir_corrected = "data/corrected"
        # correct_all_files(input_dir, output_dir_corrected)

        # Bước 3: Chia nhỏ văn bản (chunking)
        # input_directory = output_dir_corrected
        input_directory = "data/processed"
        output_directory = "data/chunks"
        chunk_documents(input_directory, output_directory, PROCESSED_FILES_PATH)

        # Đánh dấu chunking đã hoàn thành
        if file_hash not in processed_files:
            processed_files[file_hash] = {}
        processed_files[file_hash]["chunked"] = True
        save_processed_files(processed_files)

        shutil.rmtree(temp_dir)

    if file_hash in processed_files and processed_files[file_hash].get("embedded", False):
        print(
            f"Tệp {file_path} đã được embedding trước đó. Bỏ qua bước embedding.")
        return processed_files[file_hash].get("doc_ids", [])

    input_directory = "data/chunks"
    model = SentenceTransformer('all-MiniLM-L6-v2')

    doc_ids = []

    for filename in os.listdir(input_directory):
        chunk_file_path = os.path.join(input_directory, filename)
        if not chunk_file_path.endswith(".json"):
            continue
        with open(chunk_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data["page_content"]
            metadata = data["metadata"]

        embedding = model.encode(text).tolist()

        doc_id = metadata.get("doc_id", str(uuid.uuid4()))
        metadata["doc_id"] = doc_id
        # insert_embeddings(doc_id, embedding)
        print(f"Đã chèn embedding cho doc_id: {doc_id}")

        doc_ids.append(doc_id)

    if file_hash not in processed_files:
        processed_files[file_hash] = {}
    processed_files[file_hash]["embedded"] = True
    processed_files[file_hash]["doc_ids"] = doc_ids
    save_processed_files(processed_files)

    return doc_ids
