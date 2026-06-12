from typing import Any


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    """
    Divide a text into overlapping word-based chunks.

    Args:
        text: Text to split.
        chunk_size: Maximum number of words per chunk.
        chunk_overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    words = text.split()

    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(words), step):
        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def create_chunks(
    pages: list[dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[dict[str, Any]]:
    """
    Creates chunks from extracted PDF pages while preserving metadata.

    Each chunk contains:
    - id
    - text
    - metadata:
        - source
        - page
        - chunk_index
    """
    chunks: list[dict[str, Any]] = []

    for page in pages:
        page_chunks = split_text(
            text=page["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(
            page_chunks
        ):
            source_stem = (
                page["source"]
                .replace(".pdf", "")
                .replace(" ", "_")
            )

            chunk_id = (
                f"{source_stem}"
                f"_page_{page['page']:04d}"
                f"_chunk_{chunk_index:03d}"
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks