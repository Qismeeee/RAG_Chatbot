import os
import uuid
import shutil
import json
import openai
import hashlib
from dotenv import load_dotenv
from preprocessing.docsLoader import langchain_document_loader
from src.preprocessing.chunking import chunk_documents
from src.preprocessing.embedding import process_embeddings
from src.embeddings.faiss_index import FaissIndex

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Đường dẫn cho file theo dõi các tệp đã xử lý
CHUNKING_PROCESSED_FILES_PATH = "data/processed_files_chunking.json"
EMBEDDING_PROCESSED_FILES_PATH = "data/processed_files_embedding.json"


def compute_file_hash(file_path):
    """Tính toán hàm băm của tệp để xác định duy nhất."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()


def load_processed_files(processed_files_path):
    """Tải danh sách các tệp đã xử lý từ tệp JSON."""
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            return json.load(f)
    return {}


def save_processed_files(processed_files, processed_files_path):
    """Lưu danh sách các tệp đã xử lý vào tệp JSON."""
    with open(processed_files_path, 'w') as f:
        json.dump(processed_files, f, indent=4)


def correct_text_with_gpt(text):
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
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.endswith(".txt") or filename.endswith(".json"):
            process_file(file_path, output_dir)


def process_uploaded_file(file_path):
    # Load danh sách tệp đã chunking và embedding
    chunking_processed_files = load_processed_files(
        CHUNKING_PROCESSED_FILES_PATH)
    embedding_processed_files = load_processed_files(
        EMBEDDING_PROCESSED_FILES_PATH)

    # Tính toán hash của tệp để kiểm tra xem nó đã được chunking hoặc embedding chưa
    file_hash = compute_file_hash(file_path)

    # Nếu tệp đã được chunking, bỏ qua bước chunking
    if file_hash in chunking_processed_files:
        print(f"Tệp {file_path} đã được chunking trước đó. Bỏ qua bước chunking.")
    else:
        # Tạo một thư mục tạm thời để xử lý file
        temp_dir = "data/temp_processing"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # Di chuyển file vào thư mục tạm thời
        filename = os.path.basename(file_path)
        temp_file_path = os.path.join(temp_dir, filename)
        shutil.copy(file_path, temp_file_path)

        # Bước 1: Trích xuất văn bản từ file
        langchain_document_loader(temp_dir)

        # Bước 2: Sửa lỗi chính tả và ngữ pháp
        input_dir = "data/processed"
        output_dir = "data/corrected"
        correct_all_files(input_dir, output_dir)

        # Bước 3: Chia nhỏ văn bản (chunking)
        input_directory = output_dir
        output_directory = "data/chunks"
        chunk_documents(input_directory, output_directory)

        # Cập nhật danh sách tệp đã chunking
        chunking_processed_files[file_hash] = True
        save_processed_files(chunking_processed_files,
                             CHUNKING_PROCESSED_FILES_PATH)

        # Xóa thư mục tạm thời
        shutil.rmtree(temp_dir)

    # Nếu tệp đã được embedding, bỏ qua bước embedding
    if file_hash in embedding_processed_files:
        print(
            f"Tệp {file_path} đã được embedding trước đó. Bỏ qua bước embedding.")
        return embedding_processed_files[file_hash]

    # Bước 4: Tạo embeddings
    input_directory = "data/chunks"
    output_embeddings_file = "data/embeddings/embeddings.json"
    document_id = str(uuid.uuid4())
    process_embeddings(input_directory, output_embeddings_file, document_id)

    # Bước 5: Lưu embeddings vào FAISS Index
    index = FaissIndex(dimension=384)
    index.load_index("models/faiss_index")
    index.add_embeddings_from_json(output_embeddings_file)
    index.save_index("models/faiss_index")

    # Cập nhật danh sách tệp đã embedding với document_id mới
    embedding_processed_files[file_hash] = document_id
    save_processed_files(embedding_processed_files,
                         EMBEDDING_PROCESSED_FILES_PATH)

    return document_id
