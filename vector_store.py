from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import numpy as np
import os
from qdrant_client.http.exceptions import UnexpectedResponse

client = QdrantClient(host="localhost", port=6333, timeout=30.0)  # Use ":memory:" or replace with host/port

COLLECTION_NAME = "frames"

# def init_collection(vector_size):
#     client.recreate_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
#     )
def init_collection(vector_size):
    # Check if collection exists
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except UnexpectedResponse as e:
        if e.status_code == 404:  # Collection not found
            print(f"Collection '{COLLECTION_NAME}' not found, creating it.")
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Collection '{COLLECTION_NAME}' created.")
        else:
            raise e

def upload_vectors(vectors, image_paths):
    points = [
        PointStruct(id=i, vector=v.tolist(), payload={"path": image_paths[i]})
        for i, v in enumerate(vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def search_similar(vector, top_k=3):
    result = client.search(collection_name=COLLECTION_NAME, query_vector=vector.tolist(), limit=top_k)
    return result
