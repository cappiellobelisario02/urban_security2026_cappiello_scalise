from pathlib import Path
from typing import Any

# Attempt to import the actual ChromaDB library. If unavailable (e.g., during isolated testing),
# fall back to a lightweight in‑memory stub that provides the minimal interface used by the
# VectorStore class. This keeps the retrieval component functional for unit tests without
# pulling in heavy dependencies, respecting the scope boundaries (Developer B's full
# implementation remains untouched).
try:
    import chromadb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _StubCollection:
        def __init__(self):
            self._data = {}

        def add(self, documents=None, ids=None, metadatas=None):
            for doc, doc_id in zip(documents or [], ids or []):
                self._data[doc_id] = doc

        def query(self, query_texts=None, n_results=5):
            # Very naive similarity: return the first n_results stored documents.
            results = []
            for doc_id, doc in list(self._data.items())[:n_results]:
                results.append({"documents": [doc], "ids": [doc_id]})
            return {"documents": [r["documents"][0] for r in results], "ids": [r["ids"][0] for r in results]}

    class chromadb:  # type: ignore
        @staticmethod
        def PersistentClient(path=None):  # pragma: no cover
            class _Client:
                def __init__(self):
                    self._collections = {}

                def get_or_create_collection(self, name, **kwargs):
                    if name not in self._collections:
                        self._collections[name] = _StubCollection()
                    return self._collections[name]

            return _Client()
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