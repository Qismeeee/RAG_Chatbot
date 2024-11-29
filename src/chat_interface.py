import openai
from openai import OpenAI
import os
import json
from sentence_transformers import SentenceTransformer

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_conversation(session_id):
    file_path = f"data/sessions/{session_id}.json"
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
    else:
        conversation = []
    return conversation


def save_conversation(session_id, conversation):
    os.makedirs("data/sessions", exist_ok=True)
    file_path = f"data/sessions/{session_id}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=4)


def create_embedding(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding.tolist()


def speech_to_text(audio_data): 
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript

def generate_answer(question, search_results, stream=True, session_id=None, model_name="gpt-4o-mini"): 
    conversation = []
    if session_id:
        conversation = load_conversation(session_id)

    conversation.extend([
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý ảo CTU, một trợ lý AI chuyên biệt hỏi đáp về các quy định, chính sách và "
                "thông tin liên quan của Đại học Cần Thơ. Nhiệm vụ của bạn là cung cấp câu trả lời chính xác, "
                "cô đọng, xúc tích cho sinh viên dựa trên thông tin được cung cấp. Dưới đây là các tài liệu liên quan: "
                f"TÀI LIỆU:\n{search_results}"
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Related documents:\n{search_results}"
            )
        }
    ])

    if stream:
        response = client.chat.completions.create(
            model=model_name,
            messages=conversation,
            stream=True,
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=conversation
        ).choices[0].message.content.strip()

    if session_id:
        conversation.append({"role": "assistant", "content": response})
        save_conversation(session_id, conversation)
    return response
