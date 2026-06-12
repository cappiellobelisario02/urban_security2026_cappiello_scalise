from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import Retriever


DATABASE_DIRECTORY = (
    PROJECT_ROOT / "data" / "chroma_db"
)


def main() -> None:
    print("=" * 70)
    print("SAFETY SENTINEL - RETRIEVAL TEST")
    print("=" * 70)

    print()
    print("Loading the embedding model and vector database...")

    retriever = Retriever(
        database_path=DATABASE_DIRECTORY
    )

    print(
        f"Available chunks: "
        f"{retriever.vector_store.count()}"
    )

    print()

    question = input(
        "Enter your question: "
    ).strip()

    if not question:
        print("The question cannot be empty.")
        return

    start_time = time.perf_counter()

    results = retriever.search(
        question=question,
        top_k=4,
    )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    print()
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(f"Question: {question}")
    print(f"Retrieved chunks: {len(results)}")
    print(
        f"Retrieval latency: "
        f"{elapsed_ms:.2f} ms"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print("=" * 70)
        print(f"RESULT {index}")
        print("=" * 70)

        print(f"Chunk ID: {result['id']}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(
            f"Chunk index: "
            f"{result['chunk_index']}"
        )
        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print("-" * 70)

        preview = result["text"][:1500]
        print(preview)

        if len(result["text"]) > 1500:
            print("...")


if __name__ == "__main__":
    main()