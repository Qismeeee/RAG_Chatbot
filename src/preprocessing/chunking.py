import os
import json
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter


def generate_doc_id(text, source, chunk_number):
    """Tạo một doc_id duy nhất dựa trên nội dung, nguồn và số thứ tự chunk."""
    unique_string = f"{text}_{source}_{chunk_number}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def load_text_content(file_path):
    """Tải nội dung văn bản từ tệp .txt."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_json_content(file_path):
    """Tải nội dung và metadata từ tệp .json."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["content"], data["metadata"]


def chunk_text_with_splitter(text, chunk_size=512, chunk_overlap=50):
    """Chia văn bản thành các đoạn nhỏ bằng RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)


def chunk_documents(input_dir, output_dir, processed_files_path, chunk_size=512, chunk_overlap=50):
    os.makedirs(output_dir, exist_ok=True)
    processed_files = load_processed_files(processed_files_path)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        base_filename = os.path.splitext(filename)[0]

        # Tính hash của tệp
        file_hash = compute_file_hash(file_path)

        # Kiểm tra xem tệp đã được xử lý chưa
        if filename in processed_files and processed_files[filename] == file_hash:
            print(f"File {filename} đã được chunking trước đó. Bỏ qua.")
            continue  # Bỏ qua tệp đã được xử lý

        # Tiến hành xử lý tệp
        if filename.endswith(".txt"):
            text = load_text_content(file_path)
            metadata = {"source": filename, "original_text": text}
        elif filename.endswith(".json"):
            text, metadata = load_json_content(file_path)
            metadata["original_text"] = text
        else:
            continue

        chunks = chunk_text_with_splitter(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for i, chunk in enumerate(chunks, start=1):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_number": i,
                "doc_id": generate_doc_id(chunk, filename, i),
                "filename": filename  # Thêm tên tệp vào metadata
            })

            output_data = {
                "page_content": chunk,
                "metadata": chunk_metadata
            }

            output_path = os.path.join(
                output_dir, f"{base_filename}_chunk_{i}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)

            print(f"Saved chunk {i} to {output_path}")

        # Cập nhật thông tin tệp đã xử lý
        processed_files[filename] = file_hash

    # Lưu lại danh sách tệp đã xử lý
    save_processed_files(processed_files, processed_files_path)

# Các hàm hỗ trợ cho xử lý file đã được xử lý


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


# Sử dụng
if __name__ == "__main__":
    input_directory = "data/corrected"
    output_directory = "data/chunks"
    processed_files_json = "data/processed_files_chunking.json"
    chunk_documents(input_directory, output_directory, processed_files_json)
