# Safety Sentinel

Safety Sentinel is a local **SLM protection pipeline** built around **Ollama + Gemma 2B**.  
The project adds multiple external safety and grounding layers to a fixed small language model in order to:

- block malicious or unsafe prompts;
- reduce hallucinations through **Retrieval-Augmented Generation (RAG)**;
- require source-aware grounded answers;
- support safe abstention when the local knowledge base is insufficient.

The project is independent from model training: all improvements are implemented through **systems engineering, middleware orchestration, local retrieval, validation, and evaluation**.

---

## Repository Overview

Main components:

- `src/main.py` – `SafetySentinel` orchestration pipeline;
- `src/guardrails.py` – input/output guardrails;
- `src/llm_client.py` – Ollama wrapper for `gemma:2b`;
- `src/rag/` – document loading, chunking, vector storage, retrieval;
- `scripts/build_knowledge_base.py` – build local ChromaDB knowledge base;
- `scripts/run_red_team_evaluation.py` – structured protected evaluation runner;
- `scripts/run_benchmark.py` – baseline/protected benchmark summary script;
- `run_system.py` – one-click launcher for benchmark and interactive CLI;
- `tests/` – unit tests for guardrails, Ollama client, and SafetySentinel;
- `DOCUMENTATION.md` – full technical report.

---

## Final Benchmark Reference

The repository contains multiple iterative benchmark outputs produced during development:

- `results/red_team_results_current.csv`
- `results/red_team_results_current_v2.csv`
- `results/red_team_results_current_v3.csv`
- `results/red_team_results_current_v4.csv`

For the **final repository state**, the recommended protected benchmark reference is:

```text
results/red_team_results_current_v4.csv
```

This file best matches the final codebase because it includes:

- refined abstention handling for false-premise / unsupported / future prompts;
- safer fallback restrictions for security-sensitive prompts;
- protection against citation-repair prompt leakage.

---

## Prerequisites

- **Python 3.11.x** recommended for the final working environment;
- **Ollama** installed locally;
- **Gemma 2B** pulled in Ollama;
- macOS / Linux shell commands below assume `zsh` or `bash`.

Pull the model if needed:

```bash
ollama pull gemma:2b
```

Verify that Ollama is running:

```bash
ollama list
```

---

## Installation

```bash
git clone https://github.com/cappiellobelisario02/urban_security2026_cappiello_scalise.git
cd urban_security2026_cappiello_scalise

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

If your local `python3.11` command is unavailable, use the Python 3.11 interpreter installed on your machine.

---

## Build the Knowledge Base

The local knowledge base is built from verified PDF documents stored in:

```text
data/knowledge_base/
```

Run:

```bash
PYTHONPATH=. venv/bin/python scripts/build_knowledge_base.py
```

This process:

1. loads the PDF files with PyMuPDF;
2. extracts text page by page;
3. chunks the pages with overlap;
4. generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`;
5. stores the records in local ChromaDB.

Expected output directory:

```text
data/chroma_db/
```

---

## Run the System

### Option 1 — One-click launcher

```bash
PYTHONPATH=. venv/bin/python run_system.py
```

The launcher allows you to:

- run the automated Red Team benchmark;
- start the interactive protected CLI chat.

### Option 2 — Interactive CLI directly

```bash
PYTHONPATH=. venv/bin/python src/main.py
```

Example prompts:

```text
What is the purpose of the NIST AI Risk Management Framework?
```

```text
OWASP Top 10 LLM
```

```text
AI Risks
```

To exit the CLI cleanly, type:

```text
exit
```

or

```text
quit
```

---

## Run Evaluation and Benchmark Scripts

### Structured protected evaluation

```bash
PYTHONPATH=. venv/bin/python scripts/run_red_team_evaluation.py --output results/red_team_results_current_v4.csv
```

This generates a row-per-prompt CSV with:

- blocked status;
- block reason;
- protected response;
- retrieved sources;
- latency fields.

### Baseline vs protected benchmark summary

```bash
PYTHONPATH=. venv/bin/python scripts/run_benchmark.py
```

This prints summary tables for:

- average baseline latency;
- average protected latency;
- protected block rate;
- heuristic hallucination rate.

---

## Run Tests

Run the full suite:

```bash
venv/bin/python -m pytest tests -q
```

Run only the most relevant suites:

```bash
venv/bin/python -m pytest tests/test_guardrails.py -q
venv/bin/python -m pytest tests/test_safety_sentinel.py -q
```

The current repository state includes tests for:

- input guardrail blocking;
- output guardrail blocking;
- prompt leakage detection;
- repair-prompt leakage detection;
- citation validation;
- fabricated citation blocking;
- extractive fallback behavior;
- abstention behavior for unsupported questions.

---

## API

The repository includes an API skeleton in:

```text
src/api.py
```

This is part of the architecture, but the main tested interaction modes are currently:

- interactive CLI;
- benchmark/evaluation scripts.

---

## Logs and Outputs

Security and pipeline events are written to:

```text
data/security_audit.log
```

Benchmark and evaluation outputs are stored in:

```text
results/
```

---

## Known Limitations

Current limitations of the final project state:

- citation validation checks **syntax and source-label membership**, not full claim-level entailment;
- some factual prompts still rely on conservative extractive fallbacks instead of richer synthesized answers;
- some safety prompts may still produce safe refusal/meta behavior rather than ideal hard blocking;
- retrieval confidence thresholds are not yet fully calibrated;
- the benchmark requires partial manual interpretation for final scientific discussion.

These limitations are documented in `DOCUMENTATION.md` and should be discussed explicitly in the final report/presentation.

---

## Recommended Files for Final Review

Before delivery, review these files first:

- `DOCUMENTATION.md`
- `README.md`
- `src/main.py`
- `src/guardrails.py`
- `results/red_team_results_current_v4.csv`

---

## Final Delivery Recommendation

For the final submission, use:

- `results/red_team_results_current_v4.csv` as the official protected benchmark reference;
- `DOCUMENTATION.md` as the technical report;
- this `README.md` as the execution guide.
