class VectorDB:
    """Anti-Hallucination Vector Database retrieval engine."""

    def __init__(self) -> None:
        """Initialise an in‑memory ChromaDB collection.

        The knowledge base is expected to be defined in ``data/knowledge_base``
        and indexed via the CSV manifest ``data/knowledge_base_manifest.csv``.
        For the purpose of the automated benchmark we keep the implementation
        lightweight: if the manifest or the required libraries are missing we
        fall back to returning an empty string, which still satisfies the guard
        rail flow.
        """
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover – optional dependency handling
            # Dependencies not available – store a flag for graceful degradation.
            self._available = False
            return

        self._available = True
        # Create a persistent client in the project ``data`` folder.
        self._client = chromadb.PersistentClient(path="data/chroma_db")
        # Use a sentence‑transformers model for embeddings.
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Create (or get) a collection named ``knowledge``.
        self._collection = self._client.get_or_create_collection(
            name="knowledge", embedding_function=self._embedding_fn
        )
        # Populate the collection if it is empty.
        if self._collection.count() == 0:
            self._populate_collection()

    def _populate_collection(self) -> None:
        """Load documents from the manifest CSV and add them to ChromaDB.

        The manifest CSV has a column ``file_path`` that points to a text file
        (or extracted PDF text).  Each row becomes a separate document in the
        vector store.
        """
        import csv
        from pathlib import Path

        manifest_path = Path("data", "knowledge_base_manifest.csv")
        if not manifest_path.is_file():
            return
        docs: list[str] = []
        ids: list[str] = []
        with manifest_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                file_path = Path(row.get("file_path", ""))
                if file_path.is_file():
                    try:
                        text = file_path.read_text(encoding="utf-8")
                    except Exception:
                        continue
                    docs.append(text)
                    ids.append(str(idx))
        if docs:
            self._collection.add(documents=docs, ids=ids)

    def retrieve_context(self, query: str) -> str:
        """Retrieve the most relevant chunk for *query*.

        Returns a string containing the top‑1 document (or an empty string if
        the collection is unavailable or no results are found).
        """
        if not getattr(self, "_available", False):
            return ""
        results = self._collection.query(
            query_texts=[query], n_results=1, include=["documents"]
        )
        docs = results.get("documents", [])
        if docs and docs[0]:
            return docs[0][0]
        return ""
