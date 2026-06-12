from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.document_loader import load_pdf_pages


KNOWLEDGE_BASE_DIRECTORY = (
    PROJECT_ROOT / "data" / "knowledge_base"
)


def main() -> None:
    pages = load_pdf_pages(KNOWLEDGE_BASE_DIRECTORY)

    print()
    print("=" * 70)
    print("PDF EXTRACTION COMPLETED")
    print("=" * 70)
    print(f"Extracted pages: {len(pages)}")

    sources = sorted(
        {page["source"] for page in pages}
    )

    print(f"Processed documents: {len(sources)}")

    for source in sources:
        source_pages = [
            page
            for page in pages
            if page["source"] == source
        ]

        print(
            f"- {source}: {len(source_pages)} pages with text"
        )

    print()
    print("=" * 70)
    print("FIRST EXTRACTED PAGE")
    print("=" * 70)

    first_page = pages[0]

    print(f"Source: {first_page['source']}")
    print(f"Page: {first_page['page']}")
    print("-" * 70)
    print(first_page["text"][:1000])


if __name__ == "__main__":
    main()