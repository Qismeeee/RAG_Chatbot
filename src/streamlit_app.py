# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
import numpy as np
# Import the search function
from search_embeddings import search_milvus, connect_to_milvus
from langchain_openai import OpenAIEmbeddings

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


def setup_sidebar():
    with st.sidebar:
        st.title("⚙️ Cấu hình")

        st.header("📚 Nguồn dữ liệu")
        data_source = st.radio("Chọn nguồn dữ liệu:", [
                               "Câu hỏi", "Tải lên PDF"])

        return data_source

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

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

# === XỬ LÝ TIN NHẮN NGƯỜI DÙNG ===


def handle_user_input(data_source):
    if data_source == "Câu hỏi":
        prompt = st.chat_input("Hãy hỏi tôi bất cứ điều gì!")
        if prompt:
            st.session_state.messages.append(
                {"role": "human", "content": prompt})
            st.chat_message("human").write(prompt)

            # Search using the text directly
            results = search_milvus(prompt)

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
