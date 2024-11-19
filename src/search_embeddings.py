from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document

def connect_to_milvus(URI_link: str, collection_name: str) -> Milvus:
    """
    Connect to Milvus and return a Milvus object.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name
    )
    return vectorstore


def search_milvus(query_text: str, collection_name: str = "langchain_milvus", top_k: int = 5):
    """
    Perform a search on Milvus using the query text.
    """
    vectorstore = connect_to_milvus('http://localhost:19530', collection_name)
    # print("Milvus connection: ", vectorstore)
    results = vectorstore.similarity_search(query_text, k=top_k)
    return results


def seed_milvus(URI_link: str, documents: list, collection_name: str = "langchain_milvus") -> Milvus:
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

def load_data(URI_link, collection_name: str = "langchain_milvus"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store_loaded = Milvus(
        embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vector_store_loaded
def main():
    # Example query
    # query_text = "Example query"
    # results = search_milvus(query_text)
    # print(f"Top results: {results}")
    data = load_data('http://localhost:19530')
    print("Check data from milvus: ", data)


if __name__ == "__main__":
    main()
