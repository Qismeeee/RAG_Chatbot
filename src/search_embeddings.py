from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus


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


def search_milvus(query_text: str, collection_name: str = "data_test", top_k: int = 5):
    """
    Perform a search on Milvus using the query text.
    """
    vectorstore = connect_to_milvus('http://localhost:19530', collection_name)
    results = vectorstore.similarity_search(query_text, k=top_k)
    return results


def main():
    # Example query
    query_text = "Example query"
    results = search_milvus(query_text)
    print(f"Top results: {results}")


if __name__ == "__main__":
    main()
