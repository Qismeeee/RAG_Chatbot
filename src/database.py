from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "chatbot_collection"


def init_milvus():
    # Kết nối đến Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    # Kiểm tra nếu collection đã tồn tại
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(name=COLLECTION_NAME)
    else:
        # Định nghĩa các field cho schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR,
                        max_length=64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        schema = CollectionSchema(fields, "Chatbot Embeddings")
        collection = Collection(name=COLLECTION_NAME, schema=schema)

        # Tạo index cho hiệu suất tìm kiếm
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)

    return collection


collection = init_milvus()


def insert_embeddings(doc_id, embedding):
    """Chèn một embedding mới vào Milvus."""
    data = [
        [doc_id],         # id
        [embedding]       # embedding
    ]
    collection.insert(data)
    collection.load()


def delete_embedding(doc_id):
    """Xóa một embedding khỏi Milvus dựa trên doc_id."""
    expr = f'id == "{doc_id}"'
    collection.delete(expr)


def search_embeddings(query_embedding, top_k=5):
    """Tìm kiếm các embeddings tương tự trong Milvus."""
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    results = collection.search(
        [query_embedding],
        "embedding",
        search_params,
        limit=top_k,
        output_fields=["id"]
    )
    return results[0]
