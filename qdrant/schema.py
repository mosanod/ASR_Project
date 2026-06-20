"""Qdrant schema and collection setup for speaker embeddings."""
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantSchema:
    def __init__(self, url="localhost", port=6333):
        try:
            self.client = QdrantClient(url=url, port=port)
            print(f"Connected to Qdrant at {url}:{port}")
        except Exception as e:
            print(f"Connection failed: {e}")

    def setup_collections(self):
        """Create collections for speaker embeddings."""
        try:
            self.client.recreate_collection(
                collection_name="speakers_prod",
                vectors_config=models.VectorParams(size=256, distance=models.Distance.COSINE),
            )
            self.client.create_payload_index(
                collection_name="speakers_prod",
                field_name="speaker_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print("Collections and indexes setup successfully")
        except Exception as e:
            print(f"Error setting up collections: {e}")


if __name__ == "__main__":
    schema = QdrantSchema()
    schema.setup_collections()
