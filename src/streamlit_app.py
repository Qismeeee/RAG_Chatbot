import copy
import itertools
import matplotlib.pyplot as plt
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
import json
from bs4 import BeautifulSoup
import time
from collections import Counter
from analyze_question import get_suggestions
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import uuid
from feedback import save_chat_history, save_feedback, init_feedback_db
from dashboard import calculate_accuracy, get_top_questions, generate_feedback_charts


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def setup_page():
    st.set_page_config(
        page_title="CTU Chatbot Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton button {
            width: 100%;
            border-radius: 20px;
            padding: 10px 15px;
            background-color: #f0f2f6;
            border: 1px solid #e0e0e0;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #e0e2e6;
            transform: translateY(-2px);
        }
        .chat-container {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .suggestion-button {
            margin: 5px;
            padding: 8px 15px;
            border-radius: 15px;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            cursor: pointer;
        }
        .chat-input {
            border-radius: 20px;
            border: 2px solid #e0e0e0;
            padding: 10px 20px;
        }
        .feedback-button {
            padding: 5px 15px;
            border-radius: 15px;
            margin: 5px;
        }
        </style>
    """, unsafe_allow_html=True)


def initialize_app():
    load_dotenv()
    setup_page()

    # Initialize messages in session state if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []


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

        # st.header("🔍 Collection để truy vấn")
        # collection_to_query = st.text_input(
        #     "Nhập tên collection cần truy vấn:",
        #     "data_ctu",
        #     help="Nhập tên collection bạn muốn sử dụng để tìm kiếm thông tin"
        # )
        collection_to_query = "data_ctu"
        return model_choice, collection_to_query


def log_user_message_to_json(message):
    log_entry = {"message": message}
    log_file_path = "D:/HK1_2024-2025/Chatbot/Chat/data/chat_logs.json"

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Ghi log câu hỏi
    if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
        with open(log_file_path, "r+", encoding="utf-8") as log_file:
            data = json.load(log_file)
            data.append(log_entry)
            log_file.seek(0)
            json.dump(data, log_file, ensure_ascii=False, indent=4)
    else:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            json.dump([log_entry], log_file, ensure_ascii=False, indent=4)


def analyze_question_frequency(log_file_path):
    with open(log_file_path, "r", encoding="utf-8") as log_file:
        data = json.load(log_file)

    questions = [entry["message"] for entry in data]
    question_counter = Counter(questions)

    for question, count in question_counter.items():
        print(f"Câu hỏi: '{question}' xuất hiện {count} lần.")

    return question_counter


def handle_local_file(use_ollama_embeddings: bool):
    # collection_name = st.text_input(
    #     "Tên collection trong Milvus:",
    #     "data_ctu",
    #     help="Nhập tên collection bạn muốn lưu trong Milvus"
    # )
    collection_name = "data_ctu"
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
    # st.cache_resource.clear()


def handle_url_input(use_ollama_embeddings: bool):
    # collection_name = st.text_input(
    #     "Tên collection trong Milvus:",
    #     "data_ctu",
    #     help="Nhập tên collection bạn muốn lưu trong Milvus"
    # )
    collection_name = "data_ctu"
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
                    for field in ['source', 'page_number', 'original_text', 'chunk_number', 'doc_id', 'filename']:
                        if field not in chunk_metadata:
                            chunk_metadata[field] = 0

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


def setup_chat_interface(model_choice):
    # Create a clean container for chat
    with st.container():
        st.markdown("""
            <div class="chat-container">
                <h1 style='text-align: center; color: #1f77b4;'>💬 CTU Chatbot Assistant</h1>
                <p style='text-align: center; color: #666;'>
                    🚀 Powered by {model} - Hỗ trợ sinh viên 24/7
                </p>
            </div>
        """.format(model="OpenAI GPT-4" if model_choice == "OpenAI GPT-4o" else "Ollama LLaMA2"),
            unsafe_allow_html=True)

    # Suggestions with better styling
    st.markdown("""
        <style>
        .suggestion-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }
        .suggestion-button {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 15px;
            padding: 15px 20px;
            flex: 1;
            min-width: 200px;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .suggestion-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        }
        </style>
    """, unsafe_allow_html=True)

    # st.subheader("Câu hỏi phổ biến")
    # suggestions = get_suggestions(limit=4)
    # cols = st.columns(4)
    # for idx, suggestion in enumerate(suggestions):
    #     with cols[idx]:
    #         st.markdown(f"""
    #             <div class="suggestion-button" onclick="document.querySelector('#user-input').value='{suggestion}'">
    #                 {suggestion}
    #             </div>
    #             """, unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.messages:
        print(f"Check msg: {msg}")
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])
    st.session_state.question = None
    st.session_state.prompt = None

    # if st.session_state.last_question != question:
    # st.session_state.last_question = question
    #     st.session_state.messages.append(
    #         {"role": "human", "content": question})
    # handle_user_input(question)

    # Display chat history
    # for msg in st.session_state.messages:
    #     role = "assistant" if msg["role"] == "assistant" else "human"
    #     st.chat_message(role).write(msg["content"])


def handle_feedback(prompt, ai_response, feedback_type):
    chat_id = save_chat_history(st.session_state.user_id, prompt, ai_response)
    if chat_id:
        if save_feedback(chat_id, feedback_type):
            return True
    return False


def handle_user_input(prompt=None):
    clicked_question = None
    # print(" prompt: ", prompt)
    key=10000
    audio_bytes = None
    cached_voice = None

    if prompt is None:
        st.subheader("Câu hỏi phổ biến") #suggested questions
        faq_suggestions = get_suggestions(limit=4)
        cols = st.columns(4)
        for idx, question in enumerate(faq_suggestions):
            with cols[idx]:
                if st.button(question, key=f"faq_btn_{idx}_{question[:10]}"):

                    clicked_question = question
                    print(f"Câu hỏi được chọn: {clicked_question}")
                    st.session_state.prompt = question
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.chat_input(
                "Hãy hỏi tôi bất cứ điều gì!", key="user_input")
        with col2:
            timestamp = int(time.time())
            key+=1
             # Create a container for the audio recorder

            audio_bytes = audio_recorder(text="", icon_size="2x")
            print("Final check casched voice: ", cached_voice)
            if audio_bytes == cached_voice:
                audio_bytes = None
            
            print(f"Check audio bytes {key} is null?: ", audio_bytes==None)
        if user_input:
            prompt = user_input
            audio_bytes = None
        if clicked_question:
            prompt = clicked_question
            audio_bytes = None
        print("Check audio byte ??????????s: ", audio_bytes==None)
        if audio_bytes:
            cached_voice = audio_bytes
            with st.spinner("Đang chuyển đổi giọng nói thành văn bản..."):
                webm_file_path = "temp_audio.mp3"
                with open(webm_file_path, "wb") as f:
                    f.write(audio_bytes)
                prompt = speech_to_text(webm_file_path)
                os.remove(webm_file_path)
                audio_bytes = None
                
        print("Check audio bytes: ", audio_bytes)
        
    print(f"Check prompt: {st.session_state.prompt}")

    if prompt:
        # Thêm câu hỏi của người dùng vào session_state
        st.session_state.messages.append({"role": "human", "content": prompt})

        # Hiển thị câu hỏi của người dùng
        st.chat_message("human").write(prompt)

        # Tạo câu trả lời từ chatbot
        try:
            results = search_milvus(prompt)
            ai_response = generate_answer(prompt, results)
            # ai_response_copy = list(ai_response)
            ai_response, ai_response_copy = itertools.tee(ai_response)
            # Nếu không phải chuỗi, chuyển thành chuỗi
            # if not isinstance(ai_response, str):
            #     ai_response_str = str(ai_response)
            #     print(f"Check ai_response_str: {ai_response_str}")
        except Exception as e:
            ai_response = "Xin lỗi, tôi không thể xử lý câu hỏi của bạn ngay lúc này."
            print(f"Lỗi khi truy vấn hoặc sinh câu trả lời: {e}")
        print(f"Câu hỏi được chọn: {prompt}")
        # Hiển thị câu trả lời của chatbot
        # ai_response_copy = copy.deepcopy(ai_response)
        st.chat_message("assistant").write_stream(ai_response)
        
        # ai_response_copy = ai_response.copy()
        # create variables to collect the stream of chunks
        collected_messages = []
        # iterate through the stream of events
        for chunk in ai_response_copy:
            # extract the message
            chunk_message = chunk.choices[0].delta.content
            collected_messages.append(chunk_message)  # save the message

        # clean None in collected_messages
        collected_messages = [m for m in collected_messages if m is not None]
        full_reply_content = ''.join(collected_messages)
        print(f"Full conversation received: {full_reply_content}")
        # Thêm câu trả lời của chatbot vào session_state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_reply_content})

        # Lưu hội thoại vào database
        chat_id = None
        try:
            chat_id = save_chat_history(
                user_id=st.session_state.user_id,
                query=prompt,
                response=ai_response
            )
            if chat_id:
                st.session_state.last_chat_id = chat_id  # Lưu chat_id vào session_state
                print(f"Đã lưu chat_history với chat_id: {chat_id}")
            else:
                print("Lỗi khi lưu chat_history.")
        except Exception as e:
            print(f"Lỗi khi lưu chat_history: {e}")

        # Hiển thị chi tiết kết quả tìm kiếm
        with st.expander("Nhấn để xem chi tiết kết quả tìm kiếm"):
            response = "Kết quả tìm kiếm:\n\n"
            for idx, result in enumerate(results, 1):
                response += f"{idx}. Nguồn: {result.metadata.get('source', 'Unknown')}\n"
                response += f"   Đoạn số: {result.metadata.get('chunk_number', 'Unknown')}\n"
                response += f"   Nội dung: {result.page_content[:200]}...\n\n"
            st.write(response)


def setup_dashboard():
    st.title("Dashboard Phản Hồi")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("💬 Back to Chat", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

    gauge, timeline, top_chart = generate_feedback_charts()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gauge, use_container_width=True)
    with col2:
        st.metric("Overall Accuracy", f"{calculate_accuracy():.1f}%")

    st.plotly_chart(timeline, use_container_width=True)
    st.plotly_chart(top_chart, use_container_width=True)


def save_feedback_to_db(query, response, feedback_type):
    try:
        chat_id = save_chat_history(
            user_id=st.session_state.user_id,
            query=query,
            response=response
        )

        if chat_id:
            feedback_value = 1 if feedback_type == 'like' else 0
            if save_feedback(chat_id, feedback_value):
                return True
        return False
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False


def main():
    initialize_app()

    if 'page' not in st.session_state:
        st.session_state.page = "chat"
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    model_choice, collection_to_query = setup_sidebar()

    if st.session_state.page == "dashboard":
        setup_dashboard()
    else:
        setup_chat_interface(model_choice)
        handle_user_input()


if __name__ == "__main__":
    main()
