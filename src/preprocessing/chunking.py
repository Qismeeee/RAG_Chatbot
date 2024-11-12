import os
import json
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter


def generate_doc_id(text, source, chunk_number):
    """Generate a unique doc_id based on content, source, and chunk number."""
    unique_string = f"{text}_{source}_{chunk_number}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def load_text_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_json_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["content"], data["metadata"]


def chunk_text_with_splitter(text, chunk_size=512, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)


def chunk_documents(input_dir, output_dir, chunk_size=512, chunk_overlap=50):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        base_filename = os.path.splitext(filename)[0]

        if filename.endswith(".txt"):
            text = load_text_content(file_path)
            metadata = {"source": filename, "original_text": text}
        elif filename.endswith(".json"):
            text, metadata = load_json_content(file_path)
            # Ensure original_text is available
            metadata["original_text"] = text
        else:
            continue

        chunks = chunk_text_with_splitter(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for i, chunk in enumerate(chunks, start=1):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_number": i,
                "doc_id": generate_doc_id(chunk, filename, i)
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


# Example usage
if __name__ == "__main__":
    input_directory = "D:/HK1_2024-2025/Chatbot/Chat/data/corrected"
    output_directory = "D:/HK1_2024-2025/Chatbot/Chat/data/chunks"
    chunk_documents(input_directory, output_directory)
