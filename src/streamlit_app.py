import traceback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chat_interface import generate_answer, speech_to_text
from preprocessing.chunking import generate_doc_id
import streamlit as st
from dotenv import load_dotenv
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from crawl import crawl_multiple_urls
from langchain_community.document_loaders import RecursiveUrlLoader
from database import search_milvus, seed_milvus
from process_data import handle_upload_file
import PyPDF2
import os
from audio_recorder_streamlit import audio_recorder
import re

from bs4 import BeautifulSoup
from feedback import analyze_feedback, save_feedback_to_db
import time


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def setup_page():
    st.set_page_config(
        page_title="CTU Chatbot Assistant",
        page_icon="🎓",
        layout="wide"
    )


def initialize_app():
    load_dotenv()
    setup_page()


def setup_header():
    st.image("D:/HK1_2024-2025/Chatbot/Chat/images/logoCTU.png", width=300)
    st.markdown(
        """
        <div style="text-align: center; margin-top: -10px; font-size: 18px; color: black; background-color: #FFFF00; padding: 10px; border-radius: 10px;">
            <b>Đồng thuận – Tận tâm – Chuẩn mực – Sáng tạo</b>
        </div>
        """,
        unsafe_allow_html=True
    )


def setup_sidebar():
    with st.sidebar:
        setup_header()
        st.title("⚙️ Cấu hình CTU Chatbot Assistant")

        if st.button("Xem Dashboard"):
            st.session_state.page = "dashboard"

        st.header("🤖 Model AI")
        model_choice = st.radio(
            "Chọn AI Model để trả lời:",
            ["OpenAI GPT-4o", "Ollama (Local)"]
        )
        st.header("🔤 Embeddings Model")
        embeddings_choice = st.radio(
            "Chọn Embeddings Model:",
            ["OpenAI", "Ollama"]
        )
        use_ollama_embeddings = False

        st.header("📚 Nguồn dữ liệu")
        data_source = st.radio(
            "Chọn nguồn dữ liệu:",
            ["File Local", "URL trực tiếp"]
        )

        if data_source == "File Local":
            handle_local_file(use_ollama_embeddings)
        else:
            handle_url_input(use_ollama_embeddings)

        st.header("🔍 Collection để truy vấn")
        collection_to_query = st.text_input(
            "Nhập tên collection cần truy vấn:",
            "data_ctu",
            help="Nhập tên collection bạn muốn sử dụng để tìm kiếm thông tin"
        )

        return model_choice, collection_to_query


def handle_local_file(use_ollama_embeddings: bool):
    collection_name = st.text_input(
        "Tên collection trong Milvus:",
        "data_ctu",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    files = st.file_uploader("Tải lên PDF", type="pdf",
                             accept_multiple_files=True)
    print("Uploaded_file: ", files)
    for uploaded_file in files:
        with st.spinner("Đang xử lý file PDF..."):
            try:
                handle_upload_file(
                    uploaded_file, collection_name, use_ollama_embeddings)
                st.toast(
                    f"Đã tải và xử lý file {uploaded_file.name} thành công!", icon="✅")
            except Exception as e:
                print(f"Lỗi khi xử lý file PDF: {str(e)}")
                print("Error: ", traceback.format_exc())
                st.toast(f"Lỗi khi xử lý file PDF: {str(e)}")
    files = []


def handle_url_input(use_ollama_embeddings: bool):
    collection_name = st.text_input(
        "Tên collection trong Milvus:",
        "data_ctu",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    url = st.text_input("Nhập URL:", "https://www.ctu.edu.vn")

    if st.button("Crawl dữ liệu"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return

        with st.spinner("Đang crawl dữ liệu..."):
            try:
                loader = RecursiveUrlLoader(url,
                                            max_depth=2,
                                            extractor=bs4_extractor
                                            )
                documents = loader.load()

                filtered_documents = [
                    doc for doc in documents if len(doc.metadata.keys()) > 2]
                print("Check documents: ", len(filtered_documents))
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4096, chunk_overlap=200)
                all_splits = text_splitter.split_documents(filtered_documents)

                all_chunks = []
                for i, chunk in enumerate(all_splits, start=1):
                    # print("Chunk metadata: ", chunk.metadata.keys())
                    chunk_metadata = {
                        "source": chunk.metadata["source"], "original_text": ""}
                    chunk_metadata.update({
                        "chunk_number": i,
                        "doc_id": generate_doc_id(chunk, chunk_metadata["source"], i),
                        "filename": chunk.metadata["title"]
                    })

                    output_data = {
                        "page_content": chunk.page_content,
                        "metadata": chunk_metadata
                    }
                    all_chunks.append(output_data)
                print("Check chunked documents: ", len(all_splits))
                seed_milvus('http://localhost:19530', all_chunks,
                            collection_name, use_ollama_embeddings)

                st.success(
                    f"Đã crawl dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                print(f"Lỗi khi crawl dữ liệu: {str(e)}")
                st.error(f"Lỗi khi crawl dữ liệu: {str(e)}")


def handle_pdf_upload():
    files = st.file_uploader("Tải lên PDF", type="pdf",
                             accept_multiple_files=True)
    print("Uploaded_file: ", files)
    for uploaded_file in files:
        with st.spinner("Đang xử lý file PDF..."):
            try:
                handle_upload_file(uploaded_file)
                st.toast(
                    f"Đã tải và xử lý file {uploaded_file.name} thành công!", icon="✅")
            except Exception as e:
                st.toast(f"Lỗi khi xử lý file PDF: {str(e)}")

    return None


def setup_chat_interface(model_choice="OpenAI GPT-4o"):
    st.title("💬 CTU Chatbot Assistant")
    if model_choice == "OpenAI GPT-4o":
        st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và OpenAI GPT-4o")
    else:
        st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và Ollama LLaMA2")
    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
                "content": "Chào mừng bạn đến với CTU Chabot Assistant! Tôi có thể hỗ trợ gì cho bạn?"}
        ]
        # msgs.add_ai_message(
        #     "Chào mừng bạn đến với CTU Chabot Assistant! Tôi có thể hỗ trợ gì cho bạn?")

    # for msg in st.session_state.messages:
    #     role = "assistant" if msg["role"] == "assistant" else "human"
    #     st.chat_message(role).write(msg["content"])

    return msgs


def handle_user_input():
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.chat_input("Hãy hỏi tôi bất cứ điều gì!")
    with col2:
        audio_bytes = audio_recorder(text="", icon_size="2x")

    if audio_bytes:
        with st.spinner("Transcribing..."):
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)

            prompt = speech_to_text(webm_file_path)
            os.remove(webm_file_path)

    if prompt:
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)

        results = search_milvus(prompt)
        print("PROMT: ", prompt)

        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_number": doc.metadata.get("chunk_number", "Unknown")
            })

        response = "Kết quả tìm kiếm:\n\n"
        for idx, result in enumerate(formatted_results, 1):
            response += f"{idx}. Nguồn: {result['source']}\n"
            response += f"   Đoạn số: {result['chunk_number']}\n"
            response += f"   Nội dung: {result['content'][:200]}...\n\n"

        st.session_state.messages.append(
            {"role": "assistant", "content": response})

        with st.expander("Nhấn để xem kết quả tìm kiếm"):
            st.write(response)

        ai_response = generate_answer(prompt, results)
        st.session_state.messages.append(
            {"role": "assistant", "content": ai_response})
        st.chat_message("assistant").write_stream(ai_response)

        # Lưu trữ response để sử dụng cho feedback
        st.session_state.current_response = ai_response
        st.session_state.current_query = prompt

        # Tạo các nút feedback
        col_buttons = st.columns(2)

        with col_buttons[0]:
            if st.button("👍 Like", key="like_button", use_container_width=True):
                if save_feedback_to_db(st.session_state.current_query,
                                       st.session_state.current_response,
                                       'like'):
                    st.success("Cảm ơn bạn đã thích câu trả lời!")
                else:
                    st.error("Có lỗi xảy ra khi lưu phản hồi!")

        with col_buttons[1]:
            if st.button("👎 Dislike", key="dislike_button", use_container_width=True):
                if save_feedback_to_db(st.session_state.current_query,
                                       st.session_state.current_response,
                                       'dislike'):
                    st.error("Cảm ơn bạn đã phản hồi! Chúng tôi sẽ cải thiện.")
                else:
                    st.error("Có lỗi xảy ra khi lưu phản hồi!")


def setup_dashboard():
    st.title("Dashboard Phản Hồi")

    # Thêm hai nút vào cùng một hàng
    col1, col2 = st.columns([1, 1])  # Chia bố cục thành 2 cột đều nhau

    with col1:
        if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
            st.experimental_rerun()

    with col2:
        if st.button("← Quay lại", use_container_width=True):
            st.session_state.page = "chat"
            st.experimental_rerun()

    feedback_data = analyze_feedback()
    st.subheader("Thống Kê Phản Hồi")

    if not feedback_data:
        st.warning("Không có dữ liệu phản hồi nào để hiển thị.")
        return

    # Tính toán thống kê
    total_feedback = sum(count for _, count, _ in feedback_data)
    feedback_dict = {feedback: count for feedback, count, _ in feedback_data}
    like_count = feedback_dict.get('like', 0)
    dislike_count = feedback_dict.get('dislike', 0)

    # Hiển thị thống kê
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tổng số phản hồi", total_feedback)

    with col2:
        st.metric("Số lượng Like", like_count)

    with col3:
        st.metric("Số lượng Dislike", dislike_count)

    # Hiển thị tỷ lệ
    if total_feedback > 0:
        like_percentage = (like_count / total_feedback) * 100
        st.progress(like_percentage / 100)
        st.write(f"Tỷ lệ hài lòng: {like_percentage:.1f}%")


def main():
    initialize_app()
    model_choice, collection_to_query = setup_sidebar()

    if 'page' not in st.session_state:
        st.session_state.page = "chat"

    if st.session_state.page == "dashboard":
        setup_dashboard()
    else:
        setup_chat_interface(model_choice)
        handle_user_input()


if __name__ == "__main__":
    main()
