class VectorDB:
    """Anti-Hallucination Vector Database retrieval engine."""

    def retrieve_context(self, query: str) -> str:
        """Retrieves relevant context from the vector database for a given query.

        Args:
            query (str): The query or search string.

        Returns:
            str: Retreived context to ground the response, empty string if none found.
        """
        pass
