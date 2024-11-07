from .faiss_index import FAISSIndex
from .elasticsearch_index import ElasticsearchIndex
from .chroma_index import ChromaIndex
from .search_engine import ChatbotSearchEngine


def initialize_faiss_index(dimension=384):
    return FAISSIndex(dimension)


def initialize_elasticsearch_index(index_name="chatbot_embeddings"):
    return ElasticsearchIndex(index_name)


def initialize_chroma_index(collection_name="chatbot_collection"):
    return ChromaIndex(collection_name)
