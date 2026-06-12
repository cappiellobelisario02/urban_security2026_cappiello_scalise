from pathlib import Path
from typing import Any

from src.rag.vector_store import VectorStore


class Retriever:
    """
    Performs semantic search over the local ChromaDB knowledge base.
    """

    def __init__(
        self,
        database_path: str | Path,
        collection_name: str = "safety_sentinel_kb",
    ) -> None:
        self.vector_store = VectorStore(
            database_path=database_path,
            collection_name=collection_name,
        )

        if self.vector_store.count() == 0:
            raise RuntimeError(
                "The vector database is empty. "
                "Run scripts/build_knowledge_base.py first."
            )

    def search(
        self,
        question: str,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Returns the most relevant chunks for a user question.
        """
        clean_question = question.strip()

        if not clean_question:
            raise ValueError(
                "The question cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        available_records = self.vector_store.count()
        number_of_results = min(
            top_k,
            available_records,
        )

        query_embedding = (
            self.vector_store.embedding_model.encode(
                [clean_question],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        results = self.vector_store.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=number_of_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved_chunks: list[dict[str, Any]] = []

        for (
            chunk_id,
            document,
            metadata,
            distance,
        ) in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            retrieved_chunks.append(
                {
                    "id": chunk_id,
                    "text": document,
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_index": metadata[
                        "chunk_index"
                    ],
                    "distance": float(distance),
                }
            )

        return retrieved_chunks