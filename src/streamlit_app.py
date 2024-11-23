import streamlit as st
from dotenv import load_dotenv
from local_ollama import get_retriever as get_ollama_retriever, get_llm_and_agent as get_ollama_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from crawl import crawl_multiple_urls
from local_ollama import get_retriever as get_ollama_retriever, get_llm_and_agent as get_ollama_agent
from search_embeddings import search_milvus
import PyPDF2
import os
from audio_recorder_streamlit import audio_recorder


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
    st.image("../images/logoCTU.png", width=150)
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

        st.header("🔤 Embeddings Model")
        embeddings_choice = st.radio(
            "Chọn Embeddings Model:",
            ["OpenAI", "Ollama"]
        )
        use_ollama_embeddings = (embeddings_choice == "Ollama")

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
            "data_test",
            help="Nhập tên collection bạn muốn sử dụng để tìm kiếm thông tin"
        )

        st.header("🤖 Model AI")
        model_choice = st.radio(
            "Chọn AI Model để trả lời:",
            ["OpenAI GPT-4o", "Ollama (Local)"]
        )

        st.header("🌐 Nhập URL để crawl")
        url_to_crawl = st.text_input(
            "Nhập URL muốn crawl:", "https://www.ctu.edu.vn")

        if st.button("Crawl dữ liệu từ URL"):
            if not url_to_crawl:
                st.error("Vui lòng nhập URL!")
            else:
                docs = crawl_multiple_urls([url_to_crawl])
                st.success(f"Đã crawl dữ liệu từ URL '{url_to_crawl}'!")

        return model_choice, collection_to_query


def handle_local_file(use_ollama_embeddings: bool):
    collection_name = st.text_input(
        "Tên collection trong Milvus:",
        "data_test",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    filename = st.text_input("Tên file JSON:", "ctu_data.json")
    directory = st.text_input("Thư mục chứa file:", "data")

    if st.button("Tải dữ liệu từ file"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return

        with st.spinner("Đang tải dữ liệu..."):
            try:
                seed_milvus(
                    'http://localhost:19530',
                    collection_name,
                    filename,
                    directory,
                    use_ollama=use_ollama_embeddings
                )
                st.success(
                    f"Đã tải dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu: {str(e)}")


def handle_url_input(use_ollama_embeddings: bool):
    collection_name = st.text_input(
        "Tên collection trong Milvus:",
        "data_test_live",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    url = st.text_input("Nhập URL:", "https://www.ctu.edu.vn")

    if st.button("Crawl dữ liệu"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return

        with st.spinner("Đang crawl dữ liệu..."):
            try:
                seed_milvus_live(
                    url,
                    'http://localhost:19530',
                    collection_name,
                    'ctu_data',
                    use_ollama=use_ollama_embeddings
                )
                st.success(
                    f"Đã crawl dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
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


def setup_chat_interface(model_choice):
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
        msgs.add_ai_message(
            "Chào mừng bạn đến với CTU Chabot Assistant! Tôi có thể hỗ trợ gì cho bạn?")

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

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
        print("AUDIO: ", audio_bytes)

    if audio_bytes:
        with st.spinner("Transcribing..."):
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)

            prompt = speech_to_text(webm_file_path)
            os.remove(webm_file_path)
            # if transcript:
            #     st.session_state.messages.append({"role": "user", "content": transcript})
            #     with st.chat_message("user"):
            #         st.write(transcript)
            #     os.remove(webm_file_path)
    if prompt:
        st.session_state.messages.append(
            {"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)

        results = search_milvus(prompt)
        print("PROMT: ", prompt)
        # print("RESULTS: ", results)

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


def main():
    initialize_app()
    data_source = setup_sidebar()
    setup_chat_interface()
    handle_user_input()


if __name__ == "__main__":
    main()
