from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "red_team_dataset.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "red_team_results.csv"
)


class SentinelInterface(Protocol):
    """
    Common interface expected from SafetySentinel.
    """

    def process(self, prompt: str) -> dict[str, Any]:
        ...


class MockSafetySentinel:
    """
    Temporary implementation used to test the evaluation script.

    It will later be replaced by the real SafetySentinel class.
    """

    def process(self, prompt: str) -> dict[str, Any]:
        start_time = time.perf_counter()

        response = (
            "MOCK RESPONSE: the real SafetySentinel pipeline "
            "has not been connected yet."
        )

        total_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "blocked": False,
            "block_reason": "",
            "response": response,
            "sources": [],
            "input_guardrail_latency_ms": 0.0,
            "retrieval_latency_ms": 0.0,
            "generation_latency_ms": 0.0,
            "output_guardrail_latency_ms": 0.0,
            "total_latency_ms": total_latency_ms,
        }


def load_dataset(
    dataset_path: Path,
) -> list[dict[str, str]]:
    """
    Loads and validates the Red Team CSV dataset.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    with dataset_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        available_columns = set(
            reader.fieldnames or []
        )

        required_columns = {
            "Prompt",
            "Categoria",
        }

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:
            raise ValueError(
                "Missing required dataset columns: "
                + ", ".join(sorted(missing_columns))
            )

        dataset: list[dict[str, str]] = []

        for row_number, row in enumerate(
            reader,
            start=1,
        ):
            prompt = row["Prompt"].strip()
            category = row["Categoria"].strip()

            if not prompt:
                print(
                    f"Skipping empty prompt at CSV row "
                    f"{row_number + 1}"
                )
                continue

            dataset.append(
                {
                    "id": f"RT-{row_number:03d}",
                    "prompt": prompt,
                    "category": category,
                }
            )

    if not dataset:
        raise RuntimeError(
            "The Red Team dataset contains no valid prompts."
        )

    return dataset


def serialize_sources(
    sources: list[dict[str, Any]],
) -> str:
    """
    Converts retrieved source metadata into one CSV-safe string.
    """
    serialized_sources: list[str] = []

    for source in sources:
        filename = source.get(
            "source",
            "unknown",
        )

        page = source.get(
            "page",
            "unknown",
        )

        serialized_sources.append(
            f"{filename}:page_{page}"
        )

    return " | ".join(serialized_sources)


def evaluate_prompt(
    sentinel: SentinelInterface,
    item: dict[str, str],
) -> dict[str, Any]:
    """
    Sends one prompt to the selected pipeline.
    """
    result = sentinel.process(
        item["prompt"]
    )

    return {
        "id": item["id"],
        "prompt": item["prompt"],
        "category": item["category"],
        "blocked": result.get(
            "blocked",
            False,
        ),
        "block_reason": result.get(
            "block_reason",
            "",
        ),
        "response": result.get(
            "response",
            "",
        ),
        "sources": serialize_sources(
            result.get(
                "sources",
                [],
            )
        ),
        "input_guardrail_latency_ms": result.get(
            "input_guardrail_latency_ms",
            0.0,
        ),
        "retrieval_latency_ms": result.get(
            "retrieval_latency_ms",
            0.0,
        ),
        "generation_latency_ms": result.get(
            "generation_latency_ms",
            0.0,
        ),
        "output_guardrail_latency_ms": result.get(
            "output_guardrail_latency_ms",
            0.0,
        ),
        "total_latency_ms": result.get(
            "total_latency_ms",
            0.0,
        ),
        "error": "",
    }


def evaluate_dataset(
    sentinel: SentinelInterface,
    dataset: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Runs every prompt without stopping after individual errors.
    """
    results: list[dict[str, Any]] = []
    total_prompts = len(dataset)

    for index, item in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"[{index}/{total_prompts}] "
            f"{item['id']} - {item['category']}"
        )

        try:
            result = evaluate_prompt(
                sentinel=sentinel,
                item=item,
            )

        except Exception as error:
            print(
                f"  Error while processing "
                f"{item['id']}: {error}"
            )

            result = {
                "id": item["id"],
                "prompt": item["prompt"],
                "category": item["category"],
                "blocked": "",
                "block_reason": "",
                "response": "",
                "sources": "",
                "input_guardrail_latency_ms": "",
                "retrieval_latency_ms": "",
                "generation_latency_ms": "",
                "output_guardrail_latency_ms": "",
                "total_latency_ms": "",
                "error": str(error),
            }

        results.append(result)

    return results


def save_results(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Exports evaluation results to CSV.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "id",
        "prompt",
        "category",
        "blocked",
        "block_reason",
        "response",
        "sources",
        "input_guardrail_latency_ms",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "output_guardrail_latency_ms",
        "total_latency_ms",
        "error",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Safety Sentinel Red Team dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the Red Team CSV dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path of the generated result CSV.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of prompts "
            "to process."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    dataset = load_dataset(
        arguments.dataset
    )

    if arguments.limit is not None:
        if arguments.limit <= 0:
            raise ValueError(
                "--limit must be greater than zero."
            )

        dataset = dataset[
            : arguments.limit
        ]

    print("=" * 70)
    print("SAFETY SENTINEL - RED TEAM EVALUATION")
    print("=" * 70)

    print(
        f"Dataset: {arguments.dataset}"
    )
    print(
        f"Prompts loaded: {len(dataset)}"
    )
    print()

    # Use the real SafetySentinel pipeline instead of the mock implementation.
    from src.main import SafetySentinel
    sentinel = SafetySentinel()

    evaluation_start = time.perf_counter()

    results = evaluate_dataset(
        sentinel=sentinel,
        dataset=dataset,
    )

    total_execution_seconds = (
        time.perf_counter()
        - evaluation_start
    )

    save_results(
        results=results,
        output_path=arguments.output,
    )

    error_count = sum(
        bool(result["error"])
        for result in results
    )

    blocked_count = sum(
        result["blocked"] is True
        for result in results
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"Processed prompts: {len(results)}"
    )
    print(
        f"Blocked prompts: {blocked_count}"
    )
    print(
        f"Errors: {error_count}"
    )
    print(
        f"Execution time: "
        f"{total_execution_seconds:.2f} seconds"
    )
    print(
        f"Results saved to: "
        f"{arguments.output}"
    )


if __name__ == "__main__":
    main()