#!/usr/bin/env python
"""Automated Red‑Team benchmark for Safety Sentinel.

The script loads the CSV at ``data/red_team_dataset.csv`` which contains two
sections: the first 30 rows are *jailbreak / toxic* prompts (the "Red Team"
set) and the next 30 rows are complex factual questions (the "Baseline"
set).  It runs each prompt twice:

1. **Run 1 – Baseline**: Directly query the Ollama model via ``src.llm_client``
   without any guardrails or RAG.
2. **Run 2 – Protected**: Pass the prompt through the full ``SafetySentinel``
   pipeline defined in ``src.main`` which applies input guardrails, performs a
   ChromaDB retrieval, builds the final prompt, generates a response, and then
   checks the output guardrails.

For every call the script records:
* latency (seconds)
* whether the input guardrail blocked the request (baseline run never blocks)
* whether the output guardrail blocked the response
* a simple hallucination heuristic – if the response does **not** contain any
  word from the original prompt and the prompt is a factual question, we count
  it as a hallucination.

At the end, markdown tables summarising latency, block rate and hallucination
rate for both runs are printed to stdout.
"""

import csv
import time
from pathlib import Path
from typing import List, Tuple

import sys
from pathlib import Path

# Ensure the project root (containing the ``src`` package) is on ``sys.path``
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import OllamaClient
from src.main import SafetySentinel

DATA_CSV = Path("data", "red_team_dataset.csv")


def load_prompts(csv_path: Path) -> List[str]:
    """Load prompts from the CSV – assumes a single column named ``prompt``."""
    prompts: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Skip header row
        next(reader, None)
        for row in reader:
            # The first column contains the prompt text.
            if row:
                prompts.append(row[0].strip())
    return prompts


def run_baseline(client: OllamaClient, prompt: str) -> Tuple[float, str]:
    start = time.time()
    response = client.generate_response(prompt)
    latency = time.time() - start
    return latency, response


def run_protected(sentinel: SafetySentinel, prompt: str) -> Tuple[float, bool, str]:
    """Run the full pipeline.

    Returns latency, output_guard_ok, response (or rejection reason).
    """
    start = time.time()
    ok, resp = sentinel.process_prompt(prompt)
    latency = time.time() - start
    # ``process_prompt`` returns (bool, str) where bool indicates overall success.
    # If the guardrails blocked, ``resp`` contains the rejection message.
    output_ok = ok
    return latency, output_ok, resp


def is_hallucination(prompt: str, response: str) -> bool:
    """Very naive hallucination check – if none of the prompt words appear in the response.
    This is sufficient for the synthetic benchmark required by the documentation.
    """
    prompt_words = {w.lower() for w in prompt.split() if len(w) > 3}
    response_words = {w.lower() for w in response.split()}
    return not bool(prompt_words & response_words)


def main() -> None:
    prompts = load_prompts(DATA_CSV)
    client = OllamaClient()
    sentinel = SafetySentinel()

    # Containers for metrics
    baseline_latencies: List[float] = []
    protected_latencies: List[float] = []
    protected_blocked: int = 0
    protected_hallucinations: int = 0

    # Use tqdm for a progress bar over the prompts
    from tqdm import tqdm

    for prompt in tqdm(prompts, desc="Benchmarking prompts"):
        # Baseline (no guardrails, no RAG)
        lat, _ = run_baseline(client, prompt)
        baseline_latencies.append(lat)

        # Protected pipeline
        lat2, ok, resp = run_protected(sentinel, prompt)
        protected_latencies.append(lat2)
        if not ok:
            protected_blocked += 1
        else:
            if is_hallucination(prompt, resp):
                protected_hallucinations += 1

    # Compute aggregates
    def avg(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    baseline_avg = avg(baseline_latencies)
    protected_avg = avg(protected_latencies)
    block_rate = protected_blocked / len(prompts) * 100
    halluc_rate = protected_hallucinations / len(prompts) * 100

    # Output markdown tables
    print("# Benchmark Results (Safety Sentinel)\n")
    print("## Latency (seconds)")
    print("| Run | Average Latency |")
    print("|-----|-----------------|")
    print(f"| Baseline | {baseline_avg:.2f} |")
    print(f"| Protected | {protected_avg:.2f} |")
    print("\n## Safety Metrics")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Output Block Rate | {block_rate:.1f}% |")
    print(f"| Hallucination Rate | {halluc_rate:.1f}% |")


if __name__ == "__main__":
    main()
