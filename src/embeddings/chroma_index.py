from chromadb import Client
from chromadb.config import Settings


class ChromaIndex:
    def __init__(self, collection_name="my_collection"):
        self.client = Client(Settings())
        self.collection = self.client.get_or_create_collection(
            name=collection_name)

    def add_embeddings(self, embeddings, metadata):
        ids = [str(i) for i in range(len(embeddings))]
        self.collection.add(documents=metadata, embeddings=embeddings, ids=ids)

    def search(self, query_embedding, top_k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return [(score, doc) for score, doc in zip(results['scores'][0], results['documents'][0])]
