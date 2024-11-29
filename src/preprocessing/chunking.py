import os
import sys
import json
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.docsLoader import langchain_document_loader

from database import seed_milvus


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


def chunk_documents(input_dir, output_dir, processed_files_path, collection_name="data_ctu", use_ollama_embeddings=False, chunk_size=512, chunk_overlap=50):
    os.makedirs(output_dir, exist_ok=True)
    processed_files = load_processed_files(processed_files_path)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        base_filename = os.path.splitext(filename)[0]

        file_hash = compute_file_hash(file_path)
        if filename in processed_files and processed_files[filename] == file_hash:
            print(f"File {filename} đã được chunking trước đó. Bỏ qua.")
            continue  

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
                "filename": filename  
            })
            for field in ['source', 'page_number', 'original_text', 'chunk_number', 'doc_id', 'filename']:
                if field not in chunk_metadata:
                    chunk_metadata[field] = 0
            print("Chunk metadata: ", chunk_metadata.keys())

            output_data = {
                "page_content": chunk,
                "metadata": chunk_metadata
            }

            output_path = os.path.join(
                output_dir, f"{base_filename}_chunk_{i}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)

            store = seed_milvus(
                'http://localhost:19530', [output_data], collection_name, use_ollama_embeddings)
            print(f"Saved chunk {i} to {output_path}")

        processed_files[filename] = file_hash
    save_processed_files(processed_files, processed_files_path)



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

def prepare_milvus_data():
    input_directory = "data/milvus_processed"
    output_directory = "data/milvus_chunks"
    langchain_document_loader("../data/downloads", input_directory)
    chunk_documents(input_directory, output_directory, "data/new_milvus_processed_files.json")

# Sử dụng
if __name__ == "__main__":
    prepare_milvus_data()
