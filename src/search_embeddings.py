from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings


def connect_to_milvus(URI_link: str, collection_name: str) -> Milvus:
    if use_ollama:
        embeddings = OllamaEmbeddings(model="llama2:7b-chat")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name
    )
    return vectorstore


def search_milvus(query_text: str, collection_name: str = "langchain_milvus", top_k: int = 5):
    vectorstore = connect_to_milvus('http://localhost:19530', collection_name)
    # print("Milvus connection: ", vectorstore)
    results = vectorstore.similarity_search(query_text, k=top_k)
    return results


def seed_milvus(URI_link: str, documents: list, collection_name: str = "langchain_milvus", use_ollama: bool = False) -> Milvus:
    if use_ollama:
        embeddings = OllamaEmbeddings(model="llama2:7b-chat")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Milvus.from_documents(
        [Document(
            page_content=doc.get('page_content', ''),
            metadata=doc.get('metadata', '')
        )
            for doc in documents
        ],
        embeddings,
        collection_name=collection_name,
        connection_args={"uri": URI_link},
    )
    return vectorstore


def load_data(URI_link, collection_name: str = "langchain_milvus", use_ollama: bool = False):
    if use_ollama:
        embeddings = OllamaEmbeddings(model="llama2:7b-chat")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store_loaded = Milvus(
        embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vector_store_loaded
