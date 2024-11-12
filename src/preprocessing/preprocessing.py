import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def correct_text_with_gpt(text):
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý ngôn ngữ tiếng Việt, hãy sửa lỗi chính tả, ký tự và ngữ pháp cho đoạn văn sau."},
        {"role": "user", "content": text}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
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
            content = data["content"]
            metadata = data["metadata"]
        corrected_content = correct_text_with_gpt(content)

        output_data = {
            "content": corrected_content,
            "metadata": metadata
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"Đã sửa lỗi và lưu {output_path}")


def process_all_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.endswith(".txt") or filename.endswith(".json"):
            process_file(file_path, output_dir)


if __name__ == "__main__":
    input_directory = "D:/HK1_2024-2025/Chatbot/Chat/data/processed"
    output_directory = "D:/HK1_2024-2025/Chatbot/Chat/data/corrected"
    process_all_files(input_directory, output_directory)
