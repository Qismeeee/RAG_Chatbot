import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer


class FaissIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_embeddings_from_json(self, json_file):
        embeddings = []
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                embeddings.append(entry["embedding"])
                self.metadata.append(entry["metadata"])

        self.add_embeddings(embeddings)

    def add_embeddings(self, embeddings):
        """Thêm embeddings vào chỉ mục FAISS."""
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), top_k)

        results = []
        for i in range(top_k):
            result = {
                "score": distances[0][i],
                "metadata": self.metadata[indices[0][i]] if indices[0][i] < len(self.metadata) else None
            }
            results.append(result)
        return results


# # Ví dụ sử dụng
# if __name__ == "__main__":
#     dimension = 384
#     faiss_index = FAISSIndex(dimension)

#     embeddings_file = "D:/HK1_2024-2025/Chatbot/Chat/data/embeddings/embeddings.json"
#     faiss_index.add_embeddings_from_json(embeddings_file)

#     query = "Học bổng khuyến khích học tập dành cho ai?"
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#     query_embedding = model.encode(query)

#     results = faiss_index.search(query_embedding, top_k=5)
#     print("Kết quả tìm kiếm:")
#     for result in results:
#         print(f"Score: {result['score']}, Metadata: {result['metadata']}")
