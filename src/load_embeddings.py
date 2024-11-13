import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()


def load_data_from_local(filename: str, directory: str) -> tuple:
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f'Data loaded from {file_path}')
    return data, filename.rsplit('.', 1)[0].replace('_', ' ')


def seed_milvus(URI_link: str, collection_name: str, filename: str, directory: str, use_ollama: bool = False) -> Milvus:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    local_data, doc_name = load_data_from_local(filename, directory)

    documents = [
        Document(
            page_content=doc.get('page_content', ''),
            metadata={
                'source': doc['metadata'].get('source', ''),
                'chunk_number': doc['metadata'].get('chunk_number', 0),
                'doc_id': doc['metadata'].get('doc_id', ''),
                'filename': doc['metadata'].get('filename', ''),
                'doc_name': doc_name
            }
        )
        for doc in local_data
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
    seed_milvus('http://localhost:19530', 'data_test',
                'embeddings.json', 'data/embeddings', use_ollama=False)


if __name__ == "__main__":
    main()
