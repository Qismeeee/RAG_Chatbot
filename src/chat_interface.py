import openai
import os
import json

openai.api_key = os.getenv("OPENAI_API_KEY")


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


async def generate_answer_stream(question, session_id=None, model_name="gpt-3.5-turbo"):
    conversation = []
    if session_id:
        conversation = load_conversation(session_id)

    conversation.append({"role": "user", "content": question})
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=conversation,
        stream=True
    )

    # Gửi phản hồi dạng streaming
    for chunk in response:
        if 'choices' in chunk:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                text = delta['content']
                yield f"data: {text}\n\n"

    # Lưu lại lịch sử hội thoại
    if session_id:
        conversation.append({"role": "assistant", "content": text})
        save_conversation(session_id, conversation)
