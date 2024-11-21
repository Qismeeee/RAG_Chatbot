import openai
from openai import OpenAI
import os
import json
from database import search_embeddings, collection
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

async def generate_answer_stream(question, session_id=None, model_name="gpt-3.5-turbo"):
    conversation = []
    if session_id:
        conversation = load_conversation(session_id)

    # Tạo embedding cho câu hỏi
    query_embedding = create_embedding(question)
    # Tìm kiếm các embeddings tương tự từ Milvus
    search_results = search_embeddings(query_embedding, top_k=5)
    retrieved_texts = []
    for result in search_results:
        doc_id = result.id
        retrieved_texts.append(f"Document ID: {doc_id}")

    # Kết hợp các đoạn văn bản truy xuất được vào cuộc hội thoại
    context = "\n".join(retrieved_texts)
    conversation.append(
        {"role": "user", "content": f"{question}\nContext:\n{context}"})

    response = client.chat.completions.create(
        model=model_name,
        messages=conversation,
        stream=True
    )

    # Gửi phản hồi dạng streaming
    assistant_reply = ""
    for chunk in response:
        if 'choices' in chunk:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                text = delta['content']
                assistant_reply += text
                yield f"data: {text}\n\n"

    # Lưu lại lịch sử hội thoại
    if session_id:
        conversation.append({"role": "assistant", "content": assistant_reply})
        save_conversation(session_id, conversation)


def generate_answer(question, search_results, stream = True,  session_id=None, model_name="gpt-3.5-turbo"):
    conversation = []
    if session_id:
        conversation = load_conversation(session_id)

    conversation.extend(
        [{"role": "system", "content": f"Dựa vào kết quả truy vấn tài liệu dưới đây để trả lời câu hỏi của người dùng. \nThông tin tài liệu: {search_results}"},
        {"role": "user", "content": f"Câu hỏi: {question}\n{search_results}"}])

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
    # Lưu lại lịch sử hội thoại
    if session_id:
        conversation.append({"role": "assistant", "content": response})
        save_conversation(session_id, conversation)
    return response

    
