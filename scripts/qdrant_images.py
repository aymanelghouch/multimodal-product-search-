from qdrant_client import QdrantClient, models
import numpy


COLLECTION_NAME = "images_qdrant"

client = QdrantClient(path='/prueba')

    
client.create_collection(
)