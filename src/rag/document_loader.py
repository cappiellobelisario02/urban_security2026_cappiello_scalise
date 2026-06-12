from pathlib import Path
from typing import Any

import pymupdf


def load_pdf_pages(pdf_directory: str | Path) -> list[dict[str, Any]]:
    """
    Extract text page by page from every PDF in a directory.

    Each returned dictionary contains:
    - text: extracted page text
    - source: PDF filename
    - page: human-readable page number, starting from 1
    """
    directory = Path(pdf_directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Knowledge-base directory not found: {directory}"
        )

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {directory}"
        )

    extracted_pages: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        try:
            with pymupdf.open(pdf_path) as document:
                for page_index, page in enumerate(document):
                    text = page.get_text(
                        "text",
                        sort=True,
                    ).strip()

                    if not text:
                        print(
                            f"  Empty page skipped: {page_index + 1}"
                        )
                        continue

                    extracted_pages.append(
                        {
                            "text": text,
                            "source": pdf_path.name,
                            "page": page_index + 1,
                        }
                    )

        except Exception as error:
            print(
                f"Error while reading {pdf_path.name}: {error}"
            )

    if not extracted_pages:
        raise RuntimeError(
            "No text was extracted from the PDF documents."
        )

    return extracted_pages