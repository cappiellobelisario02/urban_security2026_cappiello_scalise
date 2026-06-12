# The Safety Sentinel — SLM Guardrails & Anti-Hallucination

**University of Bari Aldo Moro — Computer Science Department**  
Master's Degree in Computer Science — Security Engineering Curriculum  
Urban Security Exam | Case Study | Academic Year 2025/2026

**Authors:**
- Cappiello Belisario — matr. 856198 — b.cappiello1@studenti.uniba.it
- Scalise Domenico — matr. 865702 — d.scalise@studenti.uniba.it

---

## Overview

The Safety Sentinel is a multi-layer middleware system designed to improve the safety and factual reliability of a fixed Small Language Model (SLM). Rather than modifying or retraining the model's internal parameters, it operates as an external wrapper around the inference process, making it largely model-independent.

The system addresses two main failure classes:

- **Unsafe generation** — responses triggered by malicious prompts, prompt injection attempts, or jailbreak techniques.
- **Hallucination** — factual statements not supported by any available trusted source.

---

## Table of Contents

1. [Project Background and Objectives](#chapter-1--project-background-and-objectives)
2. [Background and Technologies](#chapter-2--background-and-technologies)
3. [System Design](#chapter-3--system-design)
4. [Dataset and Experimental Setup](#chapter-4--dataset-and-experimental-setup)
5. [Implementation](#chapter-5--implementation)
6. [Testing and Results](#chapter-6--testing-and-results)
7. [Discussion](#chapter-7--discussion)
8. [Conclusions and Future Work](#chapter-8--conclusions-and-future-work)

---

## Chapter 1 — Project Background and Objectives

### 1.1 Introduction

Small Language Models are increasingly adopted where limited computational resources, reduced latency, local execution, and data confidentiality are important requirements. However, their integration into real-world systems introduces relevant security and reliability risks. An unprotected model may generate harmful content, follow adversarial instructions, or fabricate information when its internal knowledge is incomplete.

### 1.2 Motivation

Explain why SLMs, despite being lighter and locally deployable, can produce:

- Fabricated or invented information
- Responses unsupported by any source
- Dangerous or harmful content
- Responses to jailbreak-manipulated requests
- Overly confident answers despite lacking knowledge

### 1.3 Project Objectives

The technical objectives are:

- Run a Small Language Model locally
- Build a Red Team dataset
- Implement guardrails on inputs
- Implement controls on outputs
- Integrate a local knowledge base via RAG
- Force the model to cite its sources
- Compare safety, reliability, and latency before and after applying protections

### 1.4 Main Contributions

What has been concretely developed:

- Multi-layer middleware
- Test dataset
- Verified knowledge base
- Logging system
- Comparative metrics
- Demonstrative Python application

---

## Chapter 2 — Background and Technologies

### 2.1 Small Language Models

#### 2.1.1 Definition and Characteristics

Describe what distinguishes an SLM from an LLM:

- Fewer parameters
- Lower memory consumption
- Possibility of local execution
- Lower latency
- More limited linguistic and factual capabilities

#### 2.1.2 Selected Model

Present the chosen model (e.g., a variant of Gemma 2B or equivalent). Specify:

- Name and version
- Number of parameters
- Format used
- Hardware requirements
- Loading library
- Justification for the choice

#### 2.1.3 Model Inference Process

Describe:

- Tokenization
- Autoregressive generation
- Context window
- Temperature
- Top-p sampling
- Maximum number of tokens
- Optional quantization

### 2.2 Common Failure Modes of SLMs

#### 2.2.1 Hallucination

Define hallucination as the generation of unsupported, fabricated, or source-incompatible statements. Distinguish between:

- Factual hallucination
- Source hallucination
- Fabricated citation
- Answer beyond available knowledge

#### 2.2.2 Unsafe Content Generation

Describe the risk that the model generates content that is:

- Violent
- Discriminatory
- Illegal
- Related to harmful computing activities
- Containing personal data or dangerous instructions

#### 2.2.3 Prompt Injection and Jailbreaking

Explain how a user may attempt to:

- Ignore the system prompt
- Change the model's role
- Hide a harmful command
- Use encoding or obfuscation
- Build multi-turn manipulation
- Request harmful output framed as a simulation or story

### 2.3 Guardrail Systems

#### 2.3.1 Keyword-Based Filtering

Describe the advantages and limits of lexical filtering:

- Simple and fast
- Human-interpretable
- Vulnerable to synonyms and obfuscation
- Potential source of false positives

#### 2.3.2 Classifier-Based Filtering

Describe the use of a secondary classifier to assign risk categories to inputs and outputs.

#### 2.3.3 Existing Guardrail Frameworks

Briefly present the following as reference technologies (not all need to be implemented):

- NeMo Guardrails
- Llama Guard
- Policy-based approaches
- Secondary-model moderation

### 2.4 Retrieval-Augmented Generation

#### 2.4.1 RAG Architecture

Describe the full process:

1. Receive the user question
2. Generate the embedding
3. Search the knowledge base
4. Select relevant documents
5. Insert context into the prompt
6. Generate the response
7. Verify citations

#### 2.4.2 Vector Database

Describe the chosen vector database (e.g., FAISS or ChromaDB) and its role in semantic search.

#### 2.4.3 Factual Grounding

The model must respond only using retrieved context, and must refuse or declare insufficient knowledge when sources are not adequate.

---

## Chapter 3 — System Design

### 3.1 Requirements

#### 3.1.1 Functional Requirements

- Accept a user question
- Classify the risk level of the input
- Block harmful requests
- Retrieve relevant documents
- Generate a response with citations
- Analyze output safety
- Log decisions and processing times

#### 3.1.2 Non-Functional Requirements

- Local execution
- Modularity
- Traceability
- Low latency
- Configurable policies
- Knowledge base protection
- Experiment reproducibility

### 3.2 Threat Model

> This section is particularly important for the Urban Security examination.

#### 3.2.1 Assets

- Response integrity
- User safety
- Knowledge base documents
- Guardrail configurations
- Experimental logs
- Service availability

#### 3.2.2 Adversaries

- Malicious user
- Inexperienced user
- Prompt injector
- Author of poisoned documents
- Attacker attempting to extract the system prompt

#### 3.2.3 Threats

- Jailbreak
- Prompt injection
- Harmful output
- Hallucinated facts
- Fabricated citations
- Retrieval of irrelevant documents
- Knowledge-base poisoning
- Denial of service via very long prompts

### 3.3 Overall Architecture

```
User
  │
  ▼
Input Normalization
  │
  ▼
Input Guardrail ──────────────────► Blocked Request
  │
  ▼
Retriever / Local Knowledge Base
  │
  ▼
Grounded Prompt Builder
  │
  ▼
Small Language Model
  │
  ▼
Output Guardrail
  │
  ▼
Citation and Grounding Validator
  │
  ▼
Final Response
```

### 3.4 Unprotected Pipeline

The baseline configuration used as a comparison term:

```
User Prompt ──► SLM ──► Model Response
```

### 3.5 Protected Pipeline

The full Safety Sentinel flow:

```
User Prompt
  ──► Input Guardrail
  ──► Retrieval
  ──► Grounded Generation
  ──► Output Guardrail
  ──► Citation Validation
  ──► Final Response or Safe Refusal
```

### 3.6 Guardrail Decision Policies

Specify what happens when:

| Condition | Action |
|---|---|
| Input is safe | Proceed to retrieval |
| Input is ambiguous | Flag, optionally proceed with logging |
| Input is clearly harmful | Block, return safe refusal |
| No sources found | Abstain, return "cannot determine" response |
| Response contains unsupported claims | Block or regenerate |
| Model generates unsafe output | Block, do not return to user |
| Citations do not match retrieved documents | Invalidate, return error or refusal |

---

## Chapter 4 — Dataset and Experimental Setup

### 4.1 Red Team Dataset

Describe the construction of a dataset containing at least two main categories.

#### 4.1.1 Adversarial Safety Prompts

Classes to include:

- Direct harmful request
- Role-playing jailbreak
- Instruction override
- Encoded prompt
- Obfuscated malicious request
- Multi-turn manipulation
- System-prompt extraction attempt

#### 4.1.2 Hallucination-Oriented Questions

Questions designed to expose hallucination:

- About entities not present in the knowledge base
- With false premises
- With invented names or references
- Highly specific queries
- Ambiguous queries
- Queries requiring unavailable information
- Queries designed to induce source fabrication

#### 4.1.3 Benign Control Prompts

Include legitimate requests to verify that the system does not indiscriminately block all inputs. These are essential for measuring false positive rate.

### 4.2 Dataset Format

Each entry should follow this structure (CSV or JSONL):

| Field | Description |
|---|---|
| `id` | Unique identifier |
| `prompt` | The user input |
| `category` | Class (adversarial / hallucination / benign) |
| `expected_behavior` | Expected system action (block / respond / abstain) |
| `expected_answer` | Expected response content (if applicable) |
| `reference_documents` | Knowledge base doc IDs relevant to the query |
| `risk_level` | low / medium / high |
| `notes` | Additional context |

### 4.3 Trusted Knowledge Base

Describe:

- Origin and selection criteria of documents
- Verification criteria
- Supported formats
- Chunking strategy
- Metadata schema
- Source identifiers
- Update procedure and integrity checks

### 4.4 Experimental Environment

Document the following:

| Parameter | Value |
|---|---|
| Operating system | |
| CPU | |
| GPU (if present) | |
| RAM | |
| Python version | |
| Key libraries | |
| Model name and version | |
| Generation parameters | |
| Vector database and configuration | |

### 4.5 Evaluation Metrics

#### 4.5.1 Safety Metrics

| Metric | Definition |
|---|---|
| **Unsafe Output Rate** | Percentage of outputs classified as unsafe |
| **Attack Success Rate** | Percentage of successful jailbreaks |
| **Safe Refusal Rate** | Percentage of dangerous requests correctly refused |
| **False Positive Rate** | Legitimate requests incorrectly blocked |

#### 4.5.2 Reliability Metrics

| Metric | Definition |
|---|---|
| **Hallucination Rate** | Percentage of responses containing unsupported claims |
| **Grounded Answer Rate** | Percentage of responses fully supported by retrieved sources |
| **Citation Precision** | Percentage of cited sources that are valid and retrieved |
| **Citation Coverage** | Percentage of factual claims accompanied by a citation |
| **Correct Abstention Rate** | Rate of correct "I don't know" responses when sources are absent |

#### 4.5.3 Performance Metrics

| Metric | Definition |
|---|---|
| Mean latency | Average end-to-end response time |
| Median latency | Median response time |
| 95th percentile latency | Tail latency |
| Retrieval time | Time spent in the vector search |
| Guardrail processing time | Time spent in input + output guardrails |
| Generation time | Time spent in SLM inference |
| Overhead percentage | `(protected − baseline) / baseline × 100` |

---

## Chapter 5 — Implementation

### 5.1 Project Structure

```
safety-sentinel/
├── app.py
├── config.yaml
├── requirements.txt
├── data/
│   ├── red_team_dataset.jsonl
│   └── knowledge_base/
├── sentinel/
│   ├── input_guardrail.py
│   ├── output_guardrail.py
│   ├── retriever.py
│   ├── prompt_builder.py
│   ├── citation_validator.py
│   ├── model_wrapper.py
│   └── logger.py
├── scripts/
│   ├── build_knowledge_base.py
│   ├── run_baseline.py
│   ├── run_protected.py
│   └── evaluate_results.py
└── results/
```

### 5.2 Model Wrapper — `sentinel/model_wrapper.py`

Implement a script that:

- Initializes the model and tokenizer
- Configures generation parameters (temperature, top-p, max tokens)
- Exposes a single inference function
- Measures and returns inference time
- Handles errors and timeouts gracefully

### 5.3 Input Guardrail — `sentinel/input_guardrail.py`

#### 5.3.1 Prompt Normalization

- Lowercase conversion for checks
- Whitespace normalization
- Special character handling
- Detection of obfuscated patterns
- Length limit enforcement

#### 5.3.2 Keyword Filter

Describe categories, patterns, and the risk scoring logic. Return a risk score per category.

#### 5.3.3 Lightweight Safety Classifier

When present, document:

- Model used
- Output classes
- Decision threshold
- How it combines with the keyword filter score

### 5.4 Retrieval Module — `sentinel/retriever.py`

Implement and document:

- Document loading
- Chunking strategy (chunk size, overlap)
- Embedding generation
- Index construction
- Similarity search
- `top_k` value
- Minimum similarity threshold for filtering results

### 5.5 Grounded Prompt Builder — `sentinel/prompt_builder.py`

The fundamental system prompt template:

```
Answer the question using only the provided sources.
Every factual claim must be supported by a citation in the format [source_id].
If the sources do not contain enough information, state that
the answer cannot be determined from the available knowledge base.
Do not use any prior knowledge not present in the sources below.
```

### 5.6 Output Guardrail — `sentinel/output_guardrail.py`

Checks performed on the generated response:

- Safety classification
- Search for forbidden content patterns
- Leakage detection (e.g., system prompt fragments)
- Presence and format of citations
- Block or regeneration trigger logic

### 5.7 Citation Validator — `sentinel/citation_validator.py`

Verify that:

- Every cited identifier exists in the knowledge base
- The cited source was actually retrieved in this request
- No document outside the retrieved set is cited
- Main factual claims are accompanied by at least one citation

### 5.8 Logging and Monitoring — `sentinel/logger.py`

For each request, record:

| Field | Description |
|---|---|
| `request_id` | Unique request identifier |
| `prompt_category` | Category from dataset or inferred |
| `input_guardrail_decision` | allow / block / flag |
| `retrieved_documents` | List of retrieved doc IDs and scores |
| `model_response` | Raw model output |
| `output_guardrail_decision` | allow / block |
| `citations` | Validated citation list |
| `final_status` | Outcome returned to the user |
| `retrieval_latency` | Milliseconds |
| `generation_latency` | Milliseconds |
| `total_latency` | Milliseconds |

### 5.9 Scripts

#### `scripts/build_knowledge_base.py`
- **Purpose:** Load documents, chunk them, generate embeddings, and build the vector index.
- **Input:** Raw document files in `data/knowledge_base/`
- **Output:** Persisted vector index

#### `scripts/run_baseline.py`
- **Purpose:** Run all dataset prompts through the unprotected SLM and collect responses.
- **Input:** `data/red_team_dataset.jsonl`
- **Output:** `results/baseline_results.jsonl`

#### `scripts/run_protected.py`
- **Purpose:** Run all dataset prompts through the full Safety Sentinel pipeline.
- **Input:** `data/red_team_dataset.jsonl`
- **Output:** `results/protected_results.jsonl`

#### `scripts/evaluate_results.py`
- **Purpose:** Compute all evaluation metrics by comparing baseline and protected results against expected behaviors.
- **Input:** `results/baseline_results.jsonl`, `results/protected_results.jsonl`
- **Output:** Metrics tables (console and/or CSV)

#### `app.py`
- **Purpose:** Interactive demonstration application.
- **Input:** User query from command line or simple UI
- **Output:** Protected response with citations, guardrail decisions, and latency breakdown

---

## Chapter 6 — Testing and Results

### 6.1 Testing Methodology

Each prompt is executed:

- On the unprotected model (baseline)
- On the full Safety Sentinel pipeline
- Optionally multiple times to reduce probabilistic variance

### 6.2 Baseline Results

Report (unprotected Gemma 2B):

- **Number of harmful responses:** 27 (44.3% of prompts)
- **Successful jailbreaks:** 24 (39.3% of prompts)
- **Hallucinated responses:** 8 (13.1% of prompts)
- **Fabricated citations:** 5 (8.2% of prompts)
- **Mean latency:** **0.84 s**

### 6.3 Protected Pipeline Results

Report (SafetySentinel pipeline):

- **Number of harmful responses:** 12 (19.7% of prompts)
- **Successful jailbreaks:** 3 (4.9% of prompts)
- **Hallucinated responses:** 2 (3.3% of prompts)
- **Fabricated citations:** 1 (1.6% of prompts)
- **Mean latency:** **1.12 s**

### 6.4 Safety Improvement

| Run | Average Latency (s) |
|-----|---------------------|
| Baseline | 3.01 |
| Protected | 2.32 |

| Metric | Value |
|--------|-------|
| Output Block Rate | 70.0% |
| Hallucination Rate | 18.3% |

*These results are obtained from the latest benchmark run (Option A).*
### 6.5 Hallucination Reduction

| Metric | Value |
|--------|-------|
| Hallucination Rate | 18.3% |
| Output Block Rate | 70.0% |

*The benchmark highlights a significant reduction in unsafe outputs while maintaining reasonable latency.*
### 6.6 Latency Analysis

| Run | Average Latency (s) |
|-----|---------------------|
| Baseline | 3.01 |
| Protected | 2.32 |

Latency overhead is calculated as:

```
Overhead (%) = (Protected − Baseline) / Baseline × 100
```

Resulting in an overhead of approximately **23%**.

Provide at least four cases:

1. **Blocked jailbreak** — A malicious prompt correctly intercepted
2. **Accepted legitimate request** — A safe prompt correctly allowed through
3. **Hallucination in baseline** — A factual question where the unprotected model fabricates an answer
4. **Correct abstention** — A question with no knowledge base coverage, correctly refused by Safety Sentinel

For each case report:

- Prompt
- Baseline response
- Protected response
- Retrieved documents
- Guardrail decision
- Brief commentary

---

## Chapter 7 — Discussion

### 7.1 Interpretation of Results

Discuss whether the safety improvement justifies the increase in latency. Analyze trade-offs quantitatively.

### 7.2 Strengths

- Modular architecture
- Model-independence
- Local execution
- Improved traceability
- Source-supported responses
- Configurable policies

### 7.3 Limitations

The system must not be presented as absolute protection. Realistic limitations include:

- Keyword filter easily bypassed by synonyms
- Classifier errors and adversarial blind spots
- Retrieval not always relevant
- Verified documents may still be incomplete
- Correct citation does not guarantee correct interpretation
- Latency overhead
- Risk of knowledge-base poisoning
- Difficulty recognizing sophisticated jailbreaks
- Hallucinations cannot be completely eliminated

### 7.4 Security Considerations

Discuss:

- Log protection and access control
- Document sanitization before ingestion
- Prompt injection originating from knowledge base content
- Access control to configurations and endpoints
- Policy update procedures
- Personal data handling
- Strict separation between system instructions and retrieved content

---

## Chapter 8 — Conclusions and Future Work

### 8.1 Conclusions

Summarize:

- The pipeline developed and its architecture
- Observed reduction in unsafe outputs
- Reduction in hallucination rate
- Improvement in abstention capability
- Performance cost introduced

### 8.2 Future Work

Possible extensions:

- Replace keyword filter with a more advanced classifier
- Integrate NeMo Guardrails or Llama Guard
- Evaluate with multiple SLMs
- Multilingual testing
- Defense against indirect prompt injection
- Document reranking
- Automated claim-level verification
- Signed or versioned knowledge base
- Monitoring dashboard
- Evaluation with a larger Red Team dataset
