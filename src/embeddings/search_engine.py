class ChatbotSearchEngine:
    def __init__(self, faiss_index=None, es_index=None, chroma_index=None):
        self.faiss_index = faiss_index
        self.es_index = es_index
        self.chroma_index = chroma_index

    def search(self, query_embedding, top_k=5, engine="faiss"):
        if engine == "faiss" and self.faiss_index:
            return self.faiss_index.search(query_embedding, top_k)
        elif engine == "elasticsearch" and self.es_index:
            return self.es_index.search(query_embedding, top_k)
        elif engine == "chroma" and self.chroma_index:
            return self.chroma_index.search(query_embedding, top_k)
        else:
            raise ValueError(
                f"Engine '{engine}' is not supported or not initialized.")
