# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
import PyPDF2
import numpy as np
# Import the search function
from search_embeddings import search_milvus
from langchain_openai import OpenAIEmbeddings

from chat_interface import generate_answer, generate_answer_stream, speech_to_text
import asyncio
# === THIẾT LẬP GIAO DIỆN TRANG WEB ===


def setup_page():
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="💬",
        layout="wide"
    )

# === KHỞI TẠO ỨNG DỤNG ===


def initialize_app():
    load_dotenv()
    setup_page()

# === THANH CÔNG CỤ BÊN TRÁI ===


def display_sidebar():
    st.sidebar.title("⚙️ Cấu hình")

    # Embedding Model Selection
    st.sidebar.subheader("📚 Embeddings Model")
    embedding_model = st.sidebar.radio(
        "Chọn Embeddings Model:",
        options=["OpenAI", "Ollama"],
        index=1
    )

    # Data Source Configuration
    st.sidebar.subheader("📚 Nguồn dữ liệu")
    data_source = st.sidebar.radio(
        "Chọn nguồn dữ liệu:",
        options=["File Local", "URL trực tiếp"],
        index=0
    )

    # Input fields based on the selected data source
    if data_source == "File Local":
        uploaded_file = st.sidebar.file_uploader(
            "Chọn file:", type=["pdf", "docx", "html"])
        if uploaded_file and st.sidebar.button("Crawl dữ liệu"):
            with st.spinner("Đang tải lên..."):
                upload_response = add_embedding(
                    file_id="example_id",  # Replace with actual data
                    filename=uploaded_file.name,
                    source="uploaded",
                    chunk_number=1,
                    doc_id="example_doc_id",
                    embedding=uploaded_file.read()
                )
                if upload_response:
                    st.sidebar.success("Dữ liệu đã được tải lên.")
                    st.session_state.documents = get_all_embeddings()
    else:
        url = st.sidebar.text_input("Nhập URL:")
        collection_name = st.sidebar.text_input("Tên collection trong Milvus:")
        if url and collection_name and st.sidebar.button("Crawl dữ liệu"):
            st.sidebar.success(f"Dữ liệu từ URL: {url} đã được xử lý.")

    # AI Model Selection
    st.sidebar.subheader("🤖 Model AI")
    ai_model = st.sidebar.radio(
        "Chọn AI Model để trả lời:",
        options=["OpenAI GPT", "Ollama (Local)"],
        index=1
    )

# === XỬ LÝ TẢI LÊN PDF ===


def handle_pdf_upload():
    uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Đang xử lý file PDF..."):
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                st.success("Đã tải và xử lý file PDF thành công!")
                return text
            except Exception as e:
                st.error(f"Lỗi khi xử lý file PDF: {str(e)}")
    return None

# === GIAO DIỆN CHAT CHÍNH ===


def setup_chat_interface():
    st.title("💬 AI Assistant")
    st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và OpenAI GPT-4")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]


def handle_user_input(data_source):
    if data_source == "Câu hỏi":
        for msg in st.session_state.messages:
            role = "assistant" if msg["role"] == "assistant" else "human"
            st.chat_message(role).write(msg["content"])
        # Create footer container for the microphone
        # footer_container = st.container()
        # with footer_container:
        col1, col2 = st.columns([3, 1])
        with col1:
            prompt = st.chat_input("Hãy hỏi tôi bất cứ điều gì!")
        with col2:
            audio_bytes = audio_recorder(text="", icon_size="2x")
            print("AUDIO: ", audio_bytes)

        if audio_bytes:
            # Write the audio bytes to a file
            with st.spinner("Transcribing..."):
                webm_file_path = "temp_audio.mp3"
                with open(webm_file_path, "wb") as f:
                    f.write(audio_bytes)

                transcript = speech_to_text(webm_file_path)
                if transcript:
                    st.session_state.messages.append(
                        {"role": "user", "content": transcript})
                    with st.chat_message("user"):
                        st.write(transcript)
                    os.remove(webm_file_path)
        if prompt:
            st.session_state.messages.append(
                {"role": "human", "content": prompt})
            st.chat_message("human").write(prompt)

            # Search using the text directly
            results = search_milvus(prompt)
            print("PROMT: ", prompt)
            # print("RESULTS: ", results)

            # Format results
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_number": doc.metadata.get("chunk_number", "Unknown")
                })

            # Display results
            response = "Kết quả tìm kiếm:\n\n"
            for idx, result in enumerate(formatted_results, 1):
                response += f"{idx}. Nguồn: {result['source']}\n"
                response += f"   Đoạn số: {result['chunk_number']}\n"
                response += f"   Nội dung: {result['content'][:200]}...\n\n"

            st.session_state.messages.append(
                {"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            ai_response = generate_answer(prompt, result)
            # print("AI response: ", ai_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response})
            # st.chat_message("assistant").write(ai_response)
            st.chat_message("assistant").write_stream(
                ai_response)  # stream response

    elif data_source == "Tải lên PDF":
        pdf_text = handle_pdf_upload()
        if pdf_text:
            st.session_state.messages.append(
                {"role": "human", "content": pdf_text})
            st.chat_message("human").write(pdf_text)

            # Search using the PDF text directly
            results = search_milvus(pdf_text)

            # Display results
            response = f"Top results: {results}"
            st.session_state.messages.append(
                {"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
# === HÀM CHÍNH ===


def main():
    initialize_app()
    data_source = setup_sidebar()
    setup_chat_interface()
    handle_user_input(data_source)


# Chạy ứng dụng
if __name__ == "__main__":
    main()

# streamlit run streamlit_app.py
