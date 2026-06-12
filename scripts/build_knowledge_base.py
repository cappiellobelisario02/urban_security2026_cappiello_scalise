from pathlib import Path
import shutil
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunker import create_chunks
from src.rag.document_loader import load_pdf_pages
from src.rag.vector_store import VectorStore


KNOWLEDGE_BASE_DIRECTORY = (
    PROJECT_ROOT / "data" / "knowledge_base"
)

DATABASE_DIRECTORY = (
    PROJECT_ROOT / "data" / "chroma_db"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def main() -> None:
    start_time = time.perf_counter()

    print("=" * 70)
    print("SAFETY SENTINEL - KNOWLEDGE BASE BUILDER")
    print("=" * 70)

    print()
    print("Step 1/3 - Loading PDF documents...")

    pages = load_pdf_pages(
        KNOWLEDGE_BASE_DIRECTORY
    )

    print(
        f"Extracted pages: {len(pages)}"
    )

    print()
    print("Step 2/3 - Creating text chunks...")

    chunks = create_chunks(
        pages=pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(
        f"Created chunks: {len(chunks)}"
    )

    print()
    print(
        "Step 3/3 - Generating embeddings "
        "and storing them in ChromaDB..."
    )

    vector_store = VectorStore(
        database_path=DATABASE_DIRECTORY
    )

    vector_store.add_chunks(
        chunks=chunks,
        batch_size=32,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    print()
    print("=" * 70)
    print("KNOWLEDGE BASE CREATED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Documents processed: "
        f"{len({page['source'] for page in pages})}"
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print(
        f"Records stored: {vector_store.count()}"
    )

    print(
        f"Database directory: "
        f"{DATABASE_DIRECTORY}"
    )

    print(
        f"Total execution time: "
        f"{elapsed_seconds:.2f} seconds"
    )


if __name__ == "__main__":
    main()