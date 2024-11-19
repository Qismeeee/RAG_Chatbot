from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient
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
    res0 = collection.insert(data)
    res1 = collection.load()
    print("RES0: ", res0)
    print("RES1: ", res1)


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

client = MilvusClient(
    uri="http://localhost:19530"
)
res = client.list_collections()
res = client.describe_collection(
    collection_name="langchain_milvus"
)
# res = client.get_load_state(
#     collection_name=COLLECTION_NAME
# )
# res = client.get(
#     collection_name=COLLECTION_NAME,
#     ids=["7c6a663592f3d8434cd1f848dc45a9c6"]
# )
print(res)

# collection = init_milvus()
# print(collection.schema)                # Return the schema.CollectionSchema of the collection.
# collection.description           # Return the description of the collection.
# print("Name: ", collection.name)                  # Return the name of the collection.
# print("Is empty: ", collection.is_empty)              # Return the boolean value that indicates if the collection is empty.
# collection.num_entities          # Return the number of entities in the collection.
# collection.primary_field         # Return the schema.FieldSchema of the primary key field.
# collection.partitions            # Return the list[Partition] object.
# print(collection.indexes)
# print(res)