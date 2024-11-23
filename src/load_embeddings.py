import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4
from crawl import crawl_multiple_urls
from langchain_ollama import OllamaEmbeddings


load_dotenv()


def load_data_from_local(filename: str, directory: str) -> list:
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f'Data loaded from {file_path}')
    return data


def seed_milvus(URI_link: str, collection_name: str, documents: list, use_ollama: bool=False) -> Milvus:
    if use_ollama:
        embeddings = OllamaEmbeddings(
            model="llama2:7b-chat"  
        )
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    documents = [
        Document(
            page_content=doc.get('page_content', ''),
            metadata={
                'source': doc['metadata'].get('source', ''),
                'title': doc['metadata'].get('title', ''),
                'description': doc['metadata'].get('description', ''),
                'doc_name': doc['metadata'].get('doc_name', '')
            }
        )
        for doc in documents
    ]

    print('documents: ', documents)

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        drop_old=True
    )
    vectorstore.add_documents(documents=documents, ids=uuids)
    print('vector: ', vectorstore)
    return vectorstore


def main():
    all_documents = load_data_from_local('ctu_data.json', 'data')

    # Seed Milvus with the loaded documents
    seed_milvus('http://localhost:19530', 'chatbot_collection', all_documents)


if __name__ == "__main__":
    main()
