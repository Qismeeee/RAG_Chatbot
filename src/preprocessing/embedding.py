# src/preprocessing/embedding.py

import os
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import hashlib

# Thêm hàm để tính hash của tệp


def compute_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()


def load_processed_files(processed_files_path):
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            return json.load(f)
    else:
        return {}


def save_processed_files(processed_files, processed_files_path):
    with open(processed_files_path, 'w') as f:
        json.dump(processed_files, f, indent=4)


model = SentenceTransformer('all-MiniLM-L6-v2')


def create_embedding(text):
    embedding = model.encode(text)
    return embedding.tolist()


def process_embeddings(input_dir, output_file, processed_files_path):
    embeddings_data = []
    processed_files = load_processed_files(processed_files_path)

    for filename in tqdm(os.listdir(input_dir), desc="Processing embeddings"):
        file_path = os.path.join(input_dir, filename)

        # Tính hash của tệp
        file_hash = compute_file_hash(file_path)

        # Kiểm tra xem tệp đã được xử lý chưa
        if filename in processed_files and processed_files[filename] == file_hash:
            print(f"File {filename} đã được embedding trước đó. Bỏ qua.")
            continue  # Bỏ qua tệp đã được xử lý

        if filename.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data["page_content"]
                metadata = data["metadata"]
                metadata["filename"] = metadata.get("filename", filename)
        else:
            continue
        embedding = create_embedding(text)
        embeddings_data.append({
            "id": metadata["doc_id"],  # Thêm id cho mỗi embedding
            "embedding": embedding,
            "metadata": metadata
        })

        # Cập nhật thông tin tệp đã xử lý
        processed_files[filename] = file_hash

    # Lưu lại embeddings mới
    if embeddings_data:
        # Kiểm tra xem tệp output có tồn tại không
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            embeddings_data = existing_data + embeddings_data

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=4)
        print(f"Saved all embeddings to {output_file}")
    else:
        print("Không có embeddings mới để lưu.")

    # Lưu lại danh sách tệp đã xử lý
    save_processed_files(processed_files, processed_files_path)


# Sử dụng
if __name__ == "__main__":
    input_directory = "data/chunks/"
    output_file = "data/embeddings/embeddings.json"
    processed_files_json = "data/processed_files_embedding.json"
    process_embeddings(input_directory, output_file, processed_files_json)
