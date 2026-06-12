from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_COLLECTION_NAME = "safety_sentinel_kb"
DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class VectorStore:
    """
    Handles embedding generation and local ChromaDB storage.
    """

    def __init__(
        self,
        database_path: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.database_path = Path(database_path)

        self.database_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Loading embedding model: "
            f"{embedding_model_name}"
        )

        self.embedding_model = SentenceTransformer(
            embedding_model_name
        )

        print(
            f"Embedding device: "
            f"{self.embedding_model.device}"
        )

        self.client = chromadb.PersistentClient(
            path=str(self.database_path)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": (
                        "Safety Sentinel trusted "
                        "knowledge base"
                    ),
                    "embedding_model": (
                        embedding_model_name
                    ),
                },
            )
        )

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> None:
        """
        Generates embeddings and stores chunks in batches.

        upsert() allows the script to be executed multiple times
        without creating duplicate records.
        """
        if not chunks:
            raise ValueError(
                "No chunks were provided."
            )

        total_chunks = len(chunks)

        for start in range(
            0,
            total_chunks,
            batch_size,
        ):
            end = min(
                start + batch_size,
                total_chunks,
            )

            batch = chunks[start:end]

            ids = [
                item["id"]
                for item in batch
            ]

            documents = [
                item["text"]
                for item in batch
            ]

            metadatas = [
                item["metadata"]
                for item in batch
            ]

            embeddings = (
                self.embedding_model.encode(
                    documents,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            )

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings.tolist(),
            )

            print(
                f"Stored chunks: "
                f"{end}/{total_chunks}"
            )

    def count(self) -> int:
        """
        Returns the number of records stored in ChromaDB.
        """
        return self.collection.count()