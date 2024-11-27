import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from pymilvus import connections, Collection, utility
import os


def connect_to_milvus(URI_link: str, collection_name: str, use_ollama: bool = False) -> Milvus:
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


def search_milvus(query_text: str, collection_name: str = "data_ctu", top_k: int = 5):
    vectorstore = connect_to_milvus('http://localhost:19530', collection_name)
    # print("Milvus connection: ", vectorstore)
    results = vectorstore.similarity_search(query_text, k=top_k)
    return results


def seed_milvus(URI_link: str, documents: list, collection_name: str = "data_ctu", use_ollama: bool = False) -> Milvus:
    print("Seeding Milvus...")
    if use_ollama:
        embeddings = OllamaEmbeddings(model="llama2:7b-chat")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", embedding_ctx_length=4096)

   
    all_documents = []
    for doc in documents:
        if isinstance(doc, dict):
            new_doc = Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"]
            )
        else:
            new_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
        all_documents.append(new_doc)
    vectorstore = Milvus.from_documents(
        all_documents,
        embeddings,
        collection_name=collection_name,
        connection_args={"uri": URI_link},
    )
    return vectorstore


def load_data(URI_link, collection_name: str = "data_ctu", use_ollama: bool = False):
    if use_ollama:
        embeddings = OllamaEmbeddings(model="llama2:7b-chat")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", embedding_ctx_length=4096)
   
    vector_store_loaded = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vector_store_loaded


MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "data_ctu"


def init_milvus():
    # Kết nối đến Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    # Kiểm tra nếu collection đã tồn tại
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(name=COLLECTION_NAME)
    else:
        print("Collection does not exist. Creating new collection...")
        collection = Collection(name=COLLECTION_NAME)

    return collection
