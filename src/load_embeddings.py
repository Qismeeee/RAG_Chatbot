import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4

from preprocessing.chunking import chunk_documents
from preprocessing.docsLoader import langchain_document_loader

load_dotenv()


def load_data_from_local(filename: str, directory: str) -> list:
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f'Data loaded from {file_path}')
    return data


def seed_milvus(URI_link: str, collection_name: str, documents: list) -> Milvus:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Convert documents to the required format
    documents = [
        Document(
            page_content=doc.get('page_content', ''),
            metadata={
                'source': doc['metadata'].get('source', ''),
                'title': doc['metadata'].get('title', ''),
                'description': doc['metadata'].get('description', ''),
            }
        )
        for doc in documents
    ]

    # print('documents: ', documents)

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

def prepare_milvus_data():
    input_directory = "data/milvus_processed"
    output_directory = "data/milvus_chunks"
    langchain_document_loader("../data", input_directory)
    chunk_documents(input_directory, output_directory, "data/new_milvus_processed_files.json")
def main():
    # Load data from local JSON file
    # all_documents = load_data_from_local('ctu_data.json', 'data')
    all_documents = load_data_from_local('school_data.json', 'data')

    # Seed Milvus with the loaded documents
    seed_milvus('http://localhost:19530', 'school_data', all_documents)


if __name__ == "__main__":
    # main()
    prepare_milvus_data()
