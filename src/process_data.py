import sys
import os
import json
import hashlib

# Đảm bảo thư mục gốc của dự án được thêm vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from embeddings.faiss_index import FaissIndex
from database import init_db, insert_document_record

def compute_file_hash(file_path):
    """Tính toán hàm băm của tệp để xác định nếu nội dung đã được xử lý."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()


def load_embeddings_to_faiss(embeddings_file, index_file):
    """Nạp embeddings từ tệp JSON vào FAISS Index và lưu Index."""
    index = FaissIndex(dimension=384)
    index.load_index(index_file)
    index.add_embeddings_from_json(embeddings_file)
    index.save_index(index_file)
    print(
        f"Embeddings from {embeddings_file} have been loaded into FAISS index at {index_file}.")


def load_metadata_to_db(embeddings_file):
    """Nạp metadata từ tệp JSON vào cơ sở dữ liệu SQLite."""
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            metadata = entry['metadata']
            doc_id = metadata['doc_id']
            filename = metadata.get('filename', 'unknown')
            insert_document_record(doc_id, filename)
    print(
        f"Metadata from {embeddings_file} has been loaded into documents.db.")


def check_and_update_processed_files(processed_files_path, embeddings_file):
    """Kiểm tra và cập nhật danh sách các embeddings đã xử lý."""
    if not os.path.exists(processed_files_path):
        return True  # Xử lý nếu danh sách không tồn tại

    # Tính toán hash của tệp embeddings
    embeddings_hash = compute_file_hash(embeddings_file)
    with open(processed_files_path, 'r') as f:
        processed_files = json.load(f)

    if processed_files.get(embeddings_file) == embeddings_hash:
        print(f"No new embeddings to process in {embeddings_file}.")
        return False  # Không cần xử lý nếu hash không thay đổi

    # Cập nhật hash mới vào danh sách tệp đã xử lý
    processed_files[embeddings_file] = embeddings_hash
    with open(processed_files_path, 'w') as f:
        json.dump(processed_files, f, indent=4)

    return True  # Cần xử lý nếu hash đã thay đổi


if __name__ == "__main__":
    embeddings_file = os.path.abspath("data/embeddings/embeddings.json")
    index_file = os.path.abspath("models/faiss_index")
    processed_files_path = os.path.abspath(
        "data/processed_files_embeddings.json")

    # Khởi tạo cơ sở dữ liệu
    init_db()

    # Kiểm tra và cập nhật nếu có embeddings mới
    if check_and_update_processed_files(processed_files_path, embeddings_file):
        # Load embeddings vào FAISS index
        load_embeddings_to_faiss(embeddings_file, index_file)

        # Load metadata vào documents.db
        load_metadata_to_db(embeddings_file)
    else:
        print("All embeddings are up-to-date. No processing needed.")
