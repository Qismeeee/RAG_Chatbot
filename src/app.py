import streamlit as st
import requests
from PIL import Image
import io

# Thiết lập tiêu đề ứng dụng
st.title("Chatbot Hỏi Đáp Nội Quy Trường")

# URL của FastAPI server (điều chỉnh tùy thuộc vào cấu hình server của bạn)
FASTAPI_URL = "http://localhost:8000/api"

# Phần phản hồi streaming


def stream_response(response):
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            st.write(chunk.decode("utf-8"))


# Chọn chế độ tương tác: Văn bản, Hình ảnh, hoặc Âm thanh
interaction_type = st.sidebar.selectbox(
    "Chọn kiểu tương tác",
    ["Text", "Image", "Audio"]
)

# Xử lý câu hỏi dạng văn bản
if interaction_type == "Text":
    st.header("Nhập câu hỏi của bạn dưới dạng văn bản")
    user_text = st.text_input("Câu hỏi:")
    if st.button("Gửi"):
        # Gửi yêu cầu đến FastAPI
        response = requests.post(
            f"{FASTAPI_URL}/text/search", json={"query": user_text}, stream=True)
        st.write("Đang tìm kiếm câu trả lời...")
        stream_response(response)

# Xử lý yêu cầu dạng hình ảnh
elif interaction_type == "Image":
    st.header("Tải lên hình ảnh để hỏi đáp")
    image_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Ảnh của bạn", use_column_width=True)
        if st.button("Gửi"):
            # Đọc ảnh thành bytes
            image_bytes = image_file.read()
            # Gửi yêu cầu đến FastAPI
            response = requests.post(
                f"{FASTAPI_URL}/image/search", files={"file": image_bytes}, stream=True)
            st.write("Đang xử lý hình ảnh...")
            stream_response(response)

# Xử lý yêu cầu dạng âm thanh
elif interaction_type == "Audio":
    st.header("Ghi âm hoặc tải lên tệp âm thanh")
    audio_file = st.file_uploader("Chọn tệp âm thanh", type=["wav", "mp3"])
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        if st.button("Gửi"):
            # Đọc âm thanh thành bytes
            audio_bytes = audio_file.read()
            # Gửi yêu cầu đến FastAPI
            response = requests.post(
                f"{FASTAPI_URL}/speech/search", files={"file": audio_bytes}, stream=True)
            st.write("Đang xử lý âm thanh...")
            stream_response(response)
