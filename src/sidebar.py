import streamlit as st
from app import add_embedding, delete_embedding, get_all_embeddings


def display_sidebar():
    model_options = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    st.sidebar.selectbox("Select Model", options=model_options, key="model")

    # Document upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file", type=["pdf", "docx", "html"])
    if uploaded_file and st.sidebar.button("Upload"):
        with st.spinner("Uploading..."):
            upload_response = add_embedding(
                file_id="example_id",  # Replace with actual data
                filename=uploaded_file.name,
                source="uploaded",
                chunk_number=1,
                doc_id="example_doc_id",
                embedding=uploaded_file.read()
            )
            if upload_response:
                st.sidebar.success("File uploaded successfully.")
                st.session_state.documents = get_all_embeddings()

    # List and delete documents
    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        st.session_state.documents = get_all_embeddings()

    # Display document list and delete functionality
    if "documents" in st.session_state and st.session_state.documents:
        for doc in st.session_state.documents["embeddings"]:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['file_id']})")

        selected_file_id = st.sidebar.selectbox(
            "Select a document to delete", options=[doc['file_id'] for doc in st.session_state.documents["embeddings"]]
        )
        if st.sidebar.button("Delete Selected Document"):
            delete_response = delete_embedding(file_id=selected_file_id)
            if delete_response:
                st.sidebar.success("Document deleted successfully.")
                st.session_state.documents = get_all_embeddings()
