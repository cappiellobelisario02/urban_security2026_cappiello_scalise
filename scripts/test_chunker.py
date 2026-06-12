from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunker import create_chunks
from src.rag.document_loader import load_pdf_pages


KNOWLEDGE_BASE_DIRECTORY = (
    PROJECT_ROOT / "data" / "knowledge_base"
)


def main() -> None:
    print("Loading PDF pages...")

    pages = load_pdf_pages(
        KNOWLEDGE_BASE_DIRECTORY
    )

    print()
    print("Creating chunks...")

    chunks = create_chunks(
        pages=pages,
        chunk_size=500,
        chunk_overlap=80,
    )

    print()
    print("=" * 70)
    print("CHUNKING COMPLETED")
    print("=" * 70)

    print(f"Extracted pages: {len(pages)}")
    print(f"Created chunks: {len(chunks)}")

    sources = sorted(
        {
            chunk["metadata"]["source"]
            for chunk in chunks
        }
    )

    print(f"Processed documents: {len(sources)}")

    for source in sources:
        source_chunks = [
            chunk
            for chunk in chunks
            if chunk["metadata"]["source"] == source
        ]

        print(
            f"- {source}: {len(source_chunks)} chunks"
        )

    print()
    print("=" * 70)
    print("FIRST CHUNK")
    print("=" * 70)

    first_chunk = chunks[0]

    print(f"ID: {first_chunk['id']}")
    print(
        f"Source: "
        f"{first_chunk['metadata']['source']}"
    )
    print(
        f"Page: "
        f"{first_chunk['metadata']['page']}"
    )
    print(
        f"Chunk index: "
        f"{first_chunk['metadata']['chunk_index']}"
    )
    print(
        f"Words: "
        f"{len(first_chunk['text'].split())}"
    )

    print("-" * 70)
    print(first_chunk["text"][:1200])


if __name__ == "__main__":
    main()