import faiss
import numpy as np
import json
import os
import pickle


class FaissIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.document_ids = []

    def add_embeddings_from_json(self, json_file):
        embeddings = []
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                embeddings.append(entry["embedding"])
                self.metadata.append(entry["metadata"])
                self.document_ids.append(entry["metadata"]["doc_id"])
        self.add_embeddings(embeddings)

    def add_embeddings(self, embeddings):
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def save_index(self, index_file_path):
        faiss.write_index(self.index, index_file_path)
        with open(index_file_path + "_meta.pkl", "wb") as f:
            pickle.dump({"metadata": self.metadata,
                        "document_ids": self.document_ids}, f)

    def load_index(self, index_file_path):
        if os.path.exists(index_file_path):
            self.index = faiss.read_index(index_file_path)
            with open(index_file_path + "_meta.pkl", "rb") as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
                self.document_ids = data["document_ids"]
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            self.document_ids = []

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), top_k)
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            result = {
                "score": distances[0][i],
                "metadata": self.metadata[idx] if idx < len(self.metadata) else None
            }
            results.append(result)
        return results
    
    def remove_document_embeddings(self, doc_id):
        indices_to_remove = [i for i, d_id in enumerate(
            self.document_ids) if d_id == doc_id]
        if not indices_to_remove:
            print(f"No embeddings found for doc_id: {doc_id}")
            return

        # Remove embeddings and metadata
        self.index.remove_ids(faiss.IDSelectorArray(np.array(indices_to_remove, dtype='int64')))
        self.metadata = [m for i, m in enumerate(self.metadata) if i not in indices_to_remove]
        self.document_ids = [d for i, d in enumerate(self.document_ids) if i not in indices_to_remove]
        print(f"Removed embeddings for doc_id: {doc_id}")
