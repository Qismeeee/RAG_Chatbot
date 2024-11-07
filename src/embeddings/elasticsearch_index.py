import os
import json
import hashlib
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm


class DocumentMetadata:
    def __init__(self, source, chunk_number, doc_id, original_text):
        self.source = source
        self.chunk_number = chunk_number
        self.doc_id = doc_id
        self.original_text = original_text


class ElasticSearchClient:
    def __init__(self, url, index_name):
        self.es_client = Elasticsearch(url)
        self.index_name = index_name
        self.create_index()

    def create_index(self):
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "original_text": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "text", "index": False}
                }
            }
        }
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(
                index=self.index_name, body=index_settings)

    def index_documents(self, documents_metadata):
        actions = [
            {
                "_index": self.index_name,
                "_id": metadata.doc_id,
                "_source": {
                    "doc_id": metadata.doc_id,
                    # Use part of the content if needed
                    "content": metadata.original_text[:512],
                    "original_text": metadata.original_text
                }
            }
            for metadata in documents_metadata
        ]
        success, _ = bulk(self.es_client, actions)
        self.es_client.indices.refresh(index=self.index_name)
        return success

    def search(self, query, k=5):
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "original_text"]
                }
            },
            "size": k
        }
        response = self.es_client.search(
            index=self.index_name, body=search_body)
        return [
            {
                "doc_id": hit["_source"]["doc_id"],
                "content": hit["_source"]["content"],
                "original_text": hit["_source"]["original_text"],
                "score": hit["_score"]
            }
            for hit in response["hits"]["hits"]
        ]

# Load actual chunks


def load_chunks(chunk_dir):
    documents = []
    for filename in tqdm(os.listdir(chunk_dir), desc="Loading chunks"):
        file_path = os.path.join(chunk_dir, filename)
        if filename.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                page_content = chunk_data["page_content"]
                metadata = chunk_data["metadata"]

                # Generate unique doc_id
                doc_id = hashlib.md5(
                    (metadata["source"] + str(metadata["chunk_number"])).encode()).hexdigest()

                documents.append(DocumentMetadata(
                    source=metadata["source"],
                    chunk_number=metadata["chunk_number"],
                    doc_id=doc_id,
                    original_text=metadata.get("original_text", page_content)
                ))
    return documents


# Usage
if __name__ == "__main__":
    es_url = "http://localhost:9200"
    index_name = "test_index"
    chunk_dir = "D:/HK1_2024-2025/Chatbot/Chat/data/chunks"

    es_client = ElasticSearchClient(es_url, index_name)

    # Load and index documents
    documents_metadata = load_chunks(chunk_dir)
    indexed = es_client.index_documents(documents_metadata)
    print(f"Indexed {indexed} documents.")

    # Example search
    query = "Học bổng khuyến khích học tập dành cho ai?"
    results = es_client.search(query)
    for result in results:
        print(f"Score: {result['score']}, Content: {result['content']}")
