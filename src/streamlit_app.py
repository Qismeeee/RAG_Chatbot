# src/streamlit_app.py

import streamlit as st
import requests
import asyncio
import uuid
from streamlit_chat import message

st.set_page_config(
    page_title="Chatbot Hỏi Đáp Nội Quy Trường Học",
    page_icon=":robot_face:",
    layout="wide",
)

# Ẩn menu và footer của Streamlit
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Khởi tạo session_state nếu chưa có
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'input_question' not in st.session_state:
    st.session_state['input_question'] = ""

session_id = st.session_state['session_id']

# Sidebar để tải lên tài liệu và chọn mô hình
with st.sidebar:
    st.title("📁 Tải lên tài liệu")
    uploaded_file = st.file_uploader(
        "Chọn file",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "mp3", "wav"],
        help="Tải lên các tài liệu để chatbot có thể học và trả lời câu hỏi của bạn."
    )
    if uploaded_file is not None:
        with st.spinner('Đang xử lý...'):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(
                "http://localhost:8000/upload_document", files=files)
        if response.status_code == 200:
            st.success("✅ Tải lên và xử lý file thành công!")
            st.experimental_rerun()
        else:
            st.error("❌ Có lỗi xảy ra khi tải lên file.")

    st.title("⚙️ Cài đặt")
    model = st.selectbox(
        "Chọn mô hình",
        ["gpt-3.5-turbo", "gpt-4"],
        help="Chọn mô hình AI để sử dụng cho chatbot."
    )

# Tiêu đề chính của ứng dụng
st.title("🤖 Chatbot Hỏi Đáp Nội Quy Trường Học")

# Hàm hiển thị lịch sử chat


def display_chat_history():
    for chat in st.session_state['chat_history']:
        if chat['role'] == 'user':
            message(chat['content'], is_user=True,
                    key=str(uuid.uuid4()) + '_user')
        else:
            message(chat['content'], is_user=False,
                    key=str(uuid.uuid4()) + '_bot')

# Hàm gửi câu hỏi và nhận phản hồi từ API


async def stream_answer(question, session_id, model):
    headers = {'accept': 'application/json',
               'Content-Type': 'application/json'}
    data = {"question": question, "model": model, "session_id": session_id}
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            headers=headers,
            json=data,
            stream=True,
            timeout=500
        )
        if response.status_code == 200:
            content = ""
            bot_message_placeholder = st.empty()
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    text = chunk.decode()
                    content += text
                    bot_message_placeholder.markdown(f"**Trả lời:** {content}")
            st.session_state['chat_history'].append(
                {"role": "assistant", "content": content})
            st.experimental_rerun()
        else:
            st.error(
                f"API request failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Hiển thị lịch sử chat
display_chat_history()

# Hàm xử lý khi người dùng gửi câu hỏi


def on_send():
    question = st.session_state['input_question']
    if question:
        st.session_state['chat_history'].append(
            {"role": "user", "content": question})
        st.session_state['input_question'] = ""  # Xóa nội dung sau khi gửi
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(stream_answer(question, session_id, model))
    else:
        st.warning("⚠️ Vui lòng nhập câu hỏi.")


# Nhập câu hỏi với `on_change`
st.text_input("Nhập câu hỏi của bạn", key="input_question", on_change=on_send)

# Nút xóa lịch sử chat
if st.button("🧹 Xóa lịch sử chat"):
    st.session_state['chat_history'] = []
    st.experimental_rerun()
