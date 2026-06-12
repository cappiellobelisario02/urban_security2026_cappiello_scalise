# The Safety Sentinel – SLM Guardrails & Anti-Hallucination

**Urban Security Exam – Case Study**

**University of Bari Aldo Moro**  
Computer Science Department  
Master’s Degree Course in Computer Science  
Security Engineering Curriculum

**Authors**

- Cappiello Belisario – Student ID 856198
- Scalise Domenico – Student ID 865702

**Academic Year 2025/2026**

---

## Documentation status

This document describes the project according to the implementation available in the repository at the end of the RAG development phase.

The following components are currently implemented and testable:

- local Ollama client configured for `gemma:2b`;
- input and output guardrail module;
- FastAPI application skeleton and `/api/chat` endpoint;
- verified Red Team dataset and evaluation key;
- trusted local knowledge base;
- PDF text extraction;
- overlapping text chunking;
- embedding generation with Sentence Transformers;
- persistent ChromaDB storage;
- semantic retrieval with source and page metadata;
- unit tests for the guardrail module.

The following components are still being integrated or evaluated:

- final implementation of `SafetySentinel.run_pipeline()`;
- replacement or adaptation of the original `rag_engine.py` stub;
- complete grounded prompt construction;
- citation validation;
- automatic Red Team execution against baseline and protected modes;
- final safety, hallucination, citation, and latency metrics.

Sections that depend on these activities are marked as pending rather than populated with estimated or invented results.

---

# Contents

1. [Project Background and Objectives](chapter_1)
2. [Background and Technologies](chapter_2)
3. [System Design](chapter_3)
4. [Dataset and Experimental Setup](chapter_4)
5. [Implementation](chapter_5)
6. [Testing and Results](chapter_6)
7. [Discussion](chapter_7)
8. [Conclusions and Future Work](chapter_8)
9. [References](references)

---

# Chapter 1 – Project Background and Objectives

## 1.1 Introduction

Small Language Models (SLMs) are increasingly adopted in applications where limited computational resources, reduced latency, local execution, and data confidentiality are important requirements. Compared with larger language models, SLMs are easier to deploy on consumer hardware and may operate without sending user data to an external cloud service. These characteristics make them attractive for experimental systems and security-sensitive applications.

Despite these advantages, integrating an SLM into a real application introduces security and reliability risks. A language model may generate harmful content, follow adversarial instructions, provide unsupported factual claims, or fabricate information when its internal knowledge is incomplete. These problems are particularly significant when the generated output is fluent and confident, because users may have difficulty distinguishing verified information from plausible but incorrect content.

This case study presents **The Safety Sentinel**, a multi-layer middleware system designed to improve the safety and factual reliability of a fixed Small Language Model. Rather than modifying or retraining the internal parameters of the model, the proposed approach operates as an external wrapper around the inference process. This design makes the system largely model-independent and permits the same architecture to be adapted to different locally deployed language models.

The protected pipeline addresses two main classes of failure. The first is **unsafe generation**, including responses produced after malicious prompts, prompt-injection attempts, or jailbreak techniques intended to bypass safety instructions. The second is **hallucination**, defined in this project as the generation of factual statements that are not supported by the trusted information available to the application.

To mitigate unsafe interactions, the system introduces input and output guardrail layers. The input guardrail analyzes the prompt before it reaches the language model and blocks requests that match configured risk indicators. The output guardrail examines the generated response and prevents unsafe or policy-violating content from being returned to the user.

To reduce hallucinations, Safety Sentinel integrates a **Retrieval-Augmented Generation (RAG)** pipeline based on a local knowledge base composed of verified documents. Before generation, the system retrieves the most relevant passages and provides them to the SLM as contextual evidence. The intended final pipeline instructs the model to use only the retrieved context, cite its sources, and abstain when the available evidence is insufficient.

The project also introduces a Red Team dataset containing adversarial prompts and factual questions designed to expose common weaknesses. The dataset includes jailbreak attempts, instruction overrides, system-prompt extraction requests, obfuscated inputs, false-premise questions, unsupported numerical claims, nonexistent sources, and questions intentionally outside the knowledge base.

The final experimental evaluation will compare two configurations:

1. the original unprotected SLM;
2. the complete Safety Sentinel pipeline.

The comparison will consider unsafe outputs, successful jailbreaks, hallucinated claims, source validity, correct refusals, false positives, and the latency introduced by the safety layers.

## 1.2 Motivation

The development of Safety Sentinel is motivated by the growing use of compact generative models in local and resource-constrained environments. Local execution can reduce dependence on external services and improve control over data processing, but it does not eliminate the weaknesses of generative models.

One concern is the production of **invented or unsupported information**. A model may generate names, dates, numerical values, standards, or references that appear credible without being supported by evidence. In a security-related context, incorrect technical information can lead to poor design decisions or create a false perception of safety.

A second concern is **unsafe content generation**. A malicious user may attempt to make the model ignore its previous instructions, adopt an unrestricted role, reveal internal configuration, or generate prohibited material. Such attempts can be direct or hidden through role-playing, Base64 encoding, punctuation-based obfuscation, and indirect instructions embedded in retrieved documents.

A third concern is **overconfidence in the absence of knowledge**. A useful protected system should not merely improve retrieval; it should also support correct abstention. When the knowledge base does not contain adequate evidence, declaring insufficiency is preferable to producing a fabricated answer.

The project therefore adopts a defense-in-depth strategy. No single filter is treated as sufficient. Instead, Safety Sentinel combines deterministic checks before inference, trusted retrieval, constrained prompt construction, output validation, and source-aware evaluation.

## 1.3 Project Objectives

The primary objective is to design and implement a Python application that wraps a locally executed SLM with safety and anti-hallucination mechanisms without fine-tuning its internal parameters.

The technical objectives are:

1. **Execute an SLM locally** through Ollama.
2. **Create a Red Team dataset** containing adversarial and hallucination-oriented prompts.
3. **Implement an input guardrail** using keywords, regular expressions, normalization, and simple heuristics.
4. **Implement an output guardrail** to detect unsafe content and possible prompt leakage.
5. **Create a local RAG knowledge base** from verified documents.
6. **Extract and chunk PDF content** while preserving source metadata.
7. **Generate semantic embeddings** and store them in a persistent vector database.
8. **Retrieve relevant evidence** for factual questions.
9. **Construct grounded prompts** that constrain the model to retrieved information.
10. **Require source citations** and support safe abstention.
11. **Compare baseline and protected configurations** using the same dataset.
12. **Measure safety, reliability, and latency** before and after protection.

## 1.4 Main Contributions

The project contributions include:

- a modular multi-layer middleware architecture;
- a Red Team dataset with 60 prompts;
- a separate evaluation key defining expected protected behavior;
- a trusted knowledge base composed of four NIST and OWASP documents;
- PDF extraction through PyMuPDF;
- word-based chunking with overlap;
- embedding generation using `sentence-transformers/all-MiniLM-L6-v2`;
- persistent local storage through ChromaDB;
- semantic retrieval with document, page, and chunk metadata;
- an Ollama client configured for `gemma:2b`;
- input and output guardrails with automated tests;
- a FastAPI interface skeleton;
- a planned automatic evaluation workflow for baseline/protected comparison.

At the current stage, the knowledge-base builder has processed four documents, extracted 252 text-bearing pages, and stored 265 chunks. Retrieval tests correctly returned the relevant NIST AI RMF and OWASP passages for representative factual questions.

---

# Chapter 2 – Background and Technologies

## 2.1 Small Language Models

### 2.1.1 Definition and Characteristics

A Small Language Model is a generative language model designed to provide useful natural-language capabilities with a lower computational footprint than very large foundation models. The term does not define one universal parameter threshold; in this project it refers operationally to a model small enough to run locally on consumer hardware.

Relevant characteristics include:

- fewer parameters than large-scale models;
- reduced memory and storage requirements;
- lower inference cost;
- practical local deployment;
- reduced reliance on external cloud services;
- potentially lower latency after local initialization;
- more limited factual coverage and reasoning capability.

The reduced footprint is useful for the project because the protected pipeline must be reproducible on standard development machines. However, smaller size does not guarantee safer behavior. The external middleware is therefore responsible for restricting unsafe requests and providing verified context.

### 2.1.2 Selected Model

The repository configures the Ollama client to use:

```text
gemma:2b
```

The model is invoked through the local Ollama HTTP generation endpoint:

```text
http://localhost:11434/api/generate
```

The project uses the model as a fixed inference component. Its weights are not trained or fine-tuned. This preserves the independence factor required by the assignment: the work concerns systems engineering, API wrapping, retrieval, and evaluation rather than internal model training.

The selected model is suitable for the prototype because it can be deployed locally and accessed through a simple HTTP interface. The final results must nevertheless be interpreted as results for this exact model and configuration, not as universal conclusions for every SLM.

### 2.1.3 Model Inference Process

The model inference process is autoregressive: the input prompt is tokenized and the model generates output tokens sequentially. The Ollama service abstracts model loading, tokenization, and generation behind an HTTP API.

The current client submits:

- the model identifier;
- the complete prompt;
- `stream: false`, requesting one complete response;
- a request timeout of 15 seconds.

Generation parameters such as temperature, top-p, context length, and maximum output length are not explicitly fixed in the current client and therefore use the model/service defaults. Before the final experiment, these values should be explicitly configured or recorded to improve reproducibility.

## 2.2 Common Failure Modes of SLMs

### 2.2.1 Hallucination

Hallucination is the generation of content that is not supported by the relevant evidence. For this project, four forms are important:

- **factual hallucination**: incorrect facts, names, dates, or numerical values;
- **source hallucination**: attribution to a real source that does not contain the claim;
- **fabricated citation**: invention of a document, page, standard, or DOI;
- **answer beyond available knowledge**: an answer produced even though the local knowledge base is insufficient.

The NIST Generative AI Profile discusses confabulation as a generative-AI risk, while the OWASP 2025 risk category on misinformation addresses false or misleading outputs that appear credible.

### 2.2.2 Unsafe Content Generation

An SLM may generate content that is dangerous, discriminatory, privacy-invasive, or otherwise incompatible with the application's policy. Relevant categories for the prototype include:

- violent or dangerous requests;
- malicious cyber requests;
- toxic or degrading content;
- sensitive-information disclosure;
- system-prompt leakage;
- instructions designed to bypass access or safety controls.

The guardrail module does not attempt to solve every possible safety category. It implements a transparent first layer based on deterministic indicators and is evaluated using a curated Red Team dataset.

### 2.2.3 Prompt Injection and Jailbreaking

Prompt injection occurs when crafted input changes the intended behavior of an LLM application. A jailbreak is a related attempt to make the model disregard safety restrictions.

Typical strategies include:

- asking the model to ignore previous instructions;
- claiming that the user has administrator authority;
- assigning an unrestricted persona;
- hiding malicious instructions in Base64;
- separating prohibited words with punctuation;
- framing the request as fiction or simulation;
- placing malicious instructions inside retrieved content.

OWASP identifies prompt injection as the first risk in its 2025 Top 10 for LLM and GenAI applications. This threat directly motivates the input checks, the Red Team categories, and the need to distinguish system instructions from retrieved text.

## 2.3 Guardrail Systems

Guardrails are external controls placed around model interaction. They can inspect input, retrieval context, generated output, tool calls, and conversation state.

### 2.3.1 Keyword-Based Filtering

Keyword filtering compares text against a configured blocklist. Its advantages are:

- low computational cost;
- predictable decisions;
- simple testing;
- clear rejection reasons;
- no dependency on an additional model.

Its limitations include:

- synonym and paraphrase evasion;
- false positives for legitimate educational discussion;
- poor understanding of context;
- difficulty detecting multilingual attacks;
- sensitivity to spelling and obfuscation.

The implemented guardrail reduces simple evasion by checking both word-bounded raw text and an alphanumeric normalized representation.

### 2.3.2 Classifier-Based Filtering

A secondary classifier can assign a risk label or score to an input or output. Unlike a pure blocklist, it can detect semantically related content that does not contain an exact keyword. However, it introduces additional latency, model dependencies, threshold selection, and its own false positives and false negatives.

The current `SafetyGuard` class includes a placeholder for an optional output classifier, but the classifier is not yet active. The present implementation therefore relies on deterministic checks.

A generic sentiment model should not automatically be treated as a toxicity or safety classifier: sentiment and harmfulness are different tasks. Any classifier added to the final system should be selected and evaluated for the intended safety taxonomy.

### 2.3.3 Existing Guardrail Frameworks

Two relevant reference approaches are:

- **NVIDIA NeMo Guardrails**, an open-source toolkit for programmable input, retrieval, dialog, execution, and output rails;
- **Llama Guard**, a family of safety models designed for prompt and response classification.

Safety Sentinel does not require these frameworks to operate. They are studied as possible alternatives or extensions. The implemented prototype favors simple and inspectable Python controls because of the limited project scope and the need to measure each layer independently.

## 2.4 Retrieval-Augmented Generation

### 2.4.1 RAG Architecture

Retrieval-Augmented Generation combines document retrieval with language-model generation. The implemented and planned process is:

1. receive the user's question;
2. normalize and validate the input;
3. encode the question as a dense vector;
4. query the local vector database;
5. select the top-ranked chunks;
6. combine the chunks with source metadata;
7. build a grounded prompt;
8. generate an answer through Ollama;
9. validate output safety and citations.

The retrieval component is already complete. The final orchestration and citation validator remain to be integrated.

### 2.4.2 Vector Database

ChromaDB is used as the local vector store. A persistent client writes the collection to `data/chroma_db/`, which is excluded from Git because it can be reconstructed from the source documents.

Each stored record contains:

- a unique chunk identifier;
- the chunk text;
- the dense embedding;
- source filename;
- PDF page number;
- chunk index.

The `safety_sentinel_kb` collection supports nearest-neighbor search using query embeddings generated by the same Sentence Transformer model used for document embeddings.

### 2.4.3 Factual Grounding

Factual grounding means constraining the answer to evidence retrieved from trusted documents. The final prompt will instruct the SLM to:

- use only the provided context;
- cite the supplied source labels;
- avoid inventing sources or page numbers;
- explicitly abstain when the context is insufficient.

RAG reduces the probability of hallucination but does not guarantee truth. Retrieval may select incomplete or irrelevant chunks, and the model may misinterpret the evidence. For this reason, retrieval quality and citation validity must be evaluated separately from response fluency.

---

# Chapter 3 – System Design

## 3.1 Requirements

### 3.1.1 Functional Requirements

The system shall:

1. accept a textual prompt;
2. reject empty or invalid input;
3. analyze the prompt for malicious keywords and injection patterns;
4. detect selected forms of obfuscation;
5. retrieve relevant chunks from the local knowledge base;
6. build an evidence-grounded prompt;
7. send the prompt to a locally running Ollama model;
8. inspect the generated response;
9. return a safe response or a controlled refusal;
10. expose the workflow through an API;
11. record relevant latency and evaluation data;
12. support automated execution of the Red Team dataset.

### 3.1.2 Non-Functional Requirements

The system should provide:

- **local execution**, avoiding dependence on a commercial cloud inference API;
- **modularity**, with separate guardrail, retrieval, model-client, API, and evaluation components;
- **traceability**, retaining source and page metadata;
- **reproducibility**, rebuilding the vector database through a script;
- **low overhead**, relative to baseline generation;
- **fault isolation**, so one failed dataset prompt does not terminate an entire experiment;
- **maintainability**, using readable Python modules and explicit configuration;
- **security awareness**, treating user input, retrieved documents, and model output as untrusted data.

## 3.2 Threat Model

### 3.2.1 Assets

Protected assets include:

- integrity of final answers;
- confidentiality of internal prompts and configuration;
- safety of the user;
- trusted knowledge-base documents;
- integrity of the ChromaDB index;
- guardrail policies;
- experimental logs and results;
- availability of the local API and Ollama service.

### 3.2.2 Adversaries

The threat model considers:

- a malicious user attempting to bypass guardrails;
- an inexperienced user who may trust an incorrect answer;
- a prompt-injection attacker;
- an attacker attempting system-prompt extraction;
- a contributor introducing a poisoned document;
- a user generating excessive requests or oversized prompts.

### 3.2.3 Threats

Primary threats are:

- direct prompt injection;
- role-play jailbreaks;
- encoded and obfuscated payloads;
- system-prompt leakage;
- unsafe output;
- factual hallucination;
- fabricated citations;
- irrelevant retrieval;
- indirect prompt injection from documents;
- knowledge-base poisoning;
- denial of service or excessive resource use.

## 3.3 Overall Architecture

```mermaid
flowchart TD
    U[User] --> N[Input Normalization]
    N --> IG[Input Guardrail]
    IG -->|Blocked| R[Safe Refusal]
    IG -->|Allowed| Q[Query Embedding]
    Q --> VS[(ChromaDB Knowledge Base)]
    VS --> RET[Top-k Retrieved Chunks]
    RET --> GP[Grounded Prompt Builder]
    GP --> O[Ollama / Gemma 2B]
    O --> OG[Output Guardrail]
    OG -->|Blocked| R
    OG -->|Allowed| CV[Citation Validator]
    CV --> F[Final Response]
```

The architecture follows a layered design. Deterministic controls are applied outside the model. Retrieval augments generation but does not directly grant the SLM access to arbitrary system resources.

## 3.4 Unprotected Pipeline

The baseline pipeline is:

```text
User Prompt -> Ollama Client -> Gemma 2B -> Model Response
```

No input filter, retrieval context, output filter, or citation check is applied. This configuration is needed to measure the original model's behavior on the same Red Team prompts.

The baseline should use the same model and generation parameters as the protected pipeline so that the comparison isolates the effect of the safety layers.

## 3.5 Protected Pipeline

The intended protected pipeline is:

```text
User Prompt
-> Input Guardrail
-> ChromaDB Retrieval
-> Grounded Prompt Construction
-> Gemma 2B Generation
-> Output Guardrail
-> Citation Validation
-> Final Response or Safe Refusal
```

Current status:

- `SafetyGuard`: implemented;
- RAG indexing and retrieval modules: implemented;
- `OllamaClient`: implemented;
- FastAPI endpoint: present;
- `SafetySentinel` constructor: present;
- `SafetySentinel.run_pipeline()`: pending integration;
- grounded prompt builder: pending integration;
- citation validator: pending implementation.

The old `VectorDB` class in `src/rag_engine.py` is still a stub and should be replaced by, or adapted to, the completed modules in `src/rag/`.

## 3.6 Guardrail Decision Policies

The final orchestrator should apply the following policies:

| Condition | Decision |
|---|---|
| Empty or non-string input | Reject |
| Malicious keyword or obfuscated keyword | Reject and log reason |
| Instruction override or role-play jailbreak | Reject and log reason |
| Encoded malicious payload | Reject and log reason |
| Input accepted | Continue to retrieval |
| Relevant context retrieved | Build grounded prompt |
| Context insufficient | Instruct model to abstain |
| Ollama timeout or connection failure | Return controlled service error |
| Unsafe generated output | Block and return safe fallback |
| Prompt leakage indicator | Block and record critical event |
| Invalid or fabricated citation | Reject, regenerate, or return validation failure |
| Safe output with valid citations | Return final response |

A future refinement may distinguish between clearly malicious input and ambiguous security-related educational questions. The current keyword approach can block legitimate discussions containing terms such as “malware” or “exploit”; this is a known false-positive risk.

---

# Chapter 4 – Dataset and Experimental Setup

## 4.1 Red Team Dataset

The project uses a curated CSV dataset containing 60 prompts. The main dataset is stored as:

```text
data/red_team_dataset.csv
```

A separate file provides expected behavior and success criteria:

```text
data/red_team_evaluation_key.csv
```

The dataset is designed for prototype evaluation rather than as a standardized benchmark. Its purpose is to compare the same model before and after the application of Safety Sentinel.

### 4.1.1 Adversarial Safety Prompts

Thirty prompts test unsafe behavior and guardrail bypasses. The categories include:

- system-prompt extraction;
- direct jailbreak;
- role-play jailbreak;
- prompt injection;
- authority impersonation;
- instruction override;
- Base64 or formatting-based obfuscation;
- toxicity;
- privacy and data-exfiltration requests;
- indirect prompt injection;
- unsafe fabrication;
- output-guardrail evasion.

The prompts are designed to exercise the middleware without embedding unnecessarily detailed operational instructions.

### 4.1.2 Hallucination-Oriented Questions

Thirty questions test factual reliability. They include:

- grounded questions answerable from NIST or OWASP sources;
- detailed questions requiring precise retrieval;
- cross-document questions;
- false-premise questions;
- unsupported numerical thresholds;
- nonexistent publications and concepts;
- questions outside the knowledge base;
- future or underdetermined questions.

For grounded questions, success requires a response supported by retrieved evidence. For unsupported questions, success requires abstention or explicit correction of the false premise.

### 4.1.3 Benign Control Prompts

The current safety and hallucination datasets contain legitimate factual prompts that serve as partial controls. For a stronger false-positive analysis, the final dataset should include a clearly labeled benign-control subset containing ordinary questions and legitimate educational security questions.

This addition is important because blocking every security-related prompt would increase the apparent refusal rate without demonstrating useful safety.

## 4.2 Dataset Format

The primary CSV has two required columns:

```text
Prompt,Categoria
```

The evaluation-key file contains richer metadata:

```text
ID
Prompt
MacroCategoria
Sottocategoria
ComportamentoAttesoPipelineProtetta
CriterioSuccesso
```

The automatic evaluation script should generate one result row per prompt and per pipeline mode. Recommended result fields are:

```text
id
pipeline
prompt
category
blocked
block_reason
response
retrieved_sources
input_guardrail_latency_ms
retrieval_latency_ms
generation_latency_ms
output_guardrail_latency_ms
total_latency_ms
error
```

## 4.3 Trusted Knowledge Base

The knowledge base contains four documents:

1. NIST AI Risk Management Framework 1.0;
2. NIST Generative AI Profile;
3. NIST Adversarial Machine Learning taxonomy;
4. OWASP Top 10 for LLM Applications 2025 clean reference.

The source manifest is stored in:

```text
data/knowledge_base_manifest.csv
```

Document selection criteria include:

- identifiable publisher;
- relevance to AI risk, hallucination, prompt injection, and LLM security;
- stable document structure;
- local availability;
- text extractability;
- traceable title, year, and source.

The OWASP item is a clean, text-searchable reference reconstructed from official OWASP material because the original distributed PDF rendered incorrectly in some viewers. It must be identified as a reconstructed reference, not represented as the original branded PDF.

### Document preprocessing

PyMuPDF extracts text page by page using:

```python
page.get_text("text", sort=True)
```

Empty pages are skipped. Every extracted page retains:

- source filename;
- one-based page number;
- text content.

### Chunking

Pages are divided into word-based chunks using:

```text
chunk_size = 500 words
chunk_overlap = 80 words
```

Chunking does not cross page boundaries. Each chunk has an identifier similar to:

```text
nist.ai.100-1_page_0025_chunk_000
```

The overlap reduces the risk that a relevant sentence is split at an arbitrary boundary.

### Embeddings and storage

The implementation uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are normalized and stored with documents and metadata in the ChromaDB collection:

```text
safety_sentinel_kb
```

The indexing run produced:

| Item | Value |
|---|---:|
| Documents | 4 |
| Text-bearing pages | 252 |
| Chunks | 265 |
| ChromaDB records | 265 |

## 4.4 Experimental Environment

The RAG component was developed and tested on:

| Component | Configuration |
|---|---|
| Platform | macOS on Apple Silicon |
| CPU architecture | ARM64 |
| Development machine | Apple Mac with M4 processor |
| Python | 3.12.13 |
| Virtual environment | Python `venv` |
| Embedding acceleration | Apple Metal Performance Shaders (`mps:0`) |
| PDF extraction | PyMuPDF |
| Embedding library | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector database | ChromaDB PersistentClient |
| Local SLM service | Ollama |
| Configured SLM | `gemma:2b` |
| API framework | FastAPI |
| Test framework | pytest |

The exact package versions used for final experiments should be exported or pinned before submission.

## 4.5 Evaluation Metrics

### 4.5.1 Safety Metrics

**Unsafe Output Rate**

```text
unsafe outputs / evaluated prompts
```

**Attack Success Rate**

```text
successful adversarial prompts / adversarial prompts
```

**Safe Refusal Rate**

```text
correctly refused malicious prompts / malicious prompts
```

**False Positive Rate**

```text
benign prompts incorrectly blocked / benign prompts
```

### 4.5.2 Reliability Metrics

**Hallucination Rate**

```text
responses containing unsupported claims / factual test responses
```

**Grounded Answer Rate**

```text
answers supported by retrieved evidence / answerable factual prompts
```

**Citation Precision**

```text
valid cited sources / all cited sources
```

**Citation Coverage**

```text
supported factual claims with citations / factual claims requiring citations
```

**Correct Abstention Rate**

```text
correct abstentions / prompts unsupported by the knowledge base
```

### 4.5.3 Performance Metrics

The experiment should record:

- mean latency;
- median latency;
- 95th-percentile latency;
- input-guardrail latency;
- retrieval latency;
- generation latency;
- output-guardrail latency;
- total protected-pipeline latency.

Latency overhead is calculated as:

```text
(protected_latency - baseline_latency) / baseline_latency * 100
```

Retrieval distances are ranking values, not calibrated probabilities. They should not be interpreted as answer confidence without additional validation.

---

# Chapter 5 – Implementation

## 5.1 Project Structure

The current repository contains the following relevant structure:

```text
urban_security2026_cappiello_scalise/
├── data/
│   ├── knowledge_base/
│   ├── knowledge_base_manifest.csv
│   ├── red_team_dataset.csv
│   └── red_team_evaluation_key.csv
├── scripts/
│   ├── build_knowledge_base.py
│   ├── test_chunker.py
│   ├── test_pdf_loader.py
│   └── test_retrieval.py
├── src/
│   ├── api.py
│   ├── guardrails.py
│   ├── llm_client.py
│   ├── main.py
│   ├── rag_engine.py
│   └── rag/
│       ├── __init__.py
│       ├── chunker.py
│       ├── document_loader.py
│       ├── retriever.py
│       └── vector_store.py
├── tests/
│   ├── test_guardrails.py
│   └── test_ollama_client.py
├── requirements.txt
└── requirements-dev.txt
```

The generated directory `data/chroma_db/` is excluded from Git and rebuilt locally.

## 5.2 Model Wrapper

The model wrapper is implemented by `OllamaClient` in `src/llm_client.py`.

Its responsibilities are:

- sending HTTP POST requests to Ollama;
- selecting `gemma:2b`;
- disabling streaming;
- applying a 15-second timeout;
- returning the generated `response` field;
- converting timeout, connection, HTTP, and malformed-response failures into controlled messages.

The wrapper isolates model communication from the rest of the application. This allows the orchestrator and evaluation scripts to use a stable Python method rather than directly managing HTTP requests.

Current limitation: the wrapper returns error markers as strings. A structured result or dedicated exceptions would make API handling and metric collection more reliable.

## 5.3 Input Guardrail

The `SafetyGuard` class implements the input and output filtering layer.

### 5.3.1 Prompt Normalization

The guardrail creates a normalized form by removing non-alphanumeric characters and converting the remaining text to lowercase. This allows strings such as:

```text
h.a.c.k
```

to be compared against the same keyword as:

```text
hack
```

The module also rejects empty or invalid inputs.

### 5.3.2 Keyword Filter

The malicious-keyword set currently includes terms associated with:

- hacking and exploitation;
- denial of service;
- malware and ransomware;
- weapons and explosives;
- theft and illegal activity;
- bypass and jailbreak attempts.

Each keyword is tested against:

1. the original lowercase input with word boundaries;
2. the normalized alphanumeric representation.

When a match is detected, the method returns a Boolean rejection decision and an explicit reason.

### 5.3.3 Pattern and Heuristic Checks

The guardrail also uses regular expressions for:

- ignoring previous instructions;
- system override;
- system-prompt extraction;
- DAN/developer-mode role play;
- unrestricted-persona requests.

Additional heuristics include:

- detection and decoding of Base64-like sequences;
- recursive scanning of decoded content;
- rejection of inputs with an excessive special-character ratio.

These checks are covered by the current unit tests.

### 5.3.4 Lightweight Safety Classifier

The class contains a placeholder for an optional output classifier:

```python
self._output_classifier = None
```

No classifier is active at the current stage. Consequently, the implemented output policy is deterministic. If a classifier is added, its exact model, labels, thresholds, hardware cost, and contribution to false positives must be documented.

## 5.4 Retrieval Module

The retrieval subsystem is divided into four modules.

### `document_loader.py`

- finds PDF files in the knowledge-base directory;
- opens each document with PyMuPDF;
- extracts text page by page;
- uses `sort=True` to improve reading order;
- skips empty pages;
- preserves source and page metadata.

### `chunker.py`

- validates chunk parameters;
- divides page text into word sequences;
- uses a 500-word chunk size and 80-word overlap;
- creates stable identifiers;
- preserves source, page, and chunk index.

### `vector_store.py`

- initializes `SentenceTransformer`;
- creates a persistent ChromaDB client;
- obtains or creates `safety_sentinel_kb`;
- encodes chunks in batches of 32;
- normalizes embeddings;
- stores records through `upsert`;
- exposes the collection record count.

Using `upsert` allows the builder to be executed repeatedly without duplicate-ID failures.

### `retriever.py`

- validates the query and `top_k`;
- encodes the question;
- queries ChromaDB with the embedding;
- returns documents, metadata, identifiers, and distances;
- limits the result count to the number of available records.

The current default is:

```text
top_k = 4
```

The value is a practical starting point and should be treated as an experimental configuration rather than a universal optimum.

## 5.5 Grounded Prompt Builder

The final grounded prompt builder is not yet integrated into `main.py`. Its intended template is:

```text
You are a factual assistant.

Answer the question using only the provided context.

Rules:
1. Do not use information outside the context.
2. Cite sources using the supplied source labels.
3. Do not invent documents, facts, page numbers, or citations.
4. If the context is insufficient, state that the available
   knowledge base does not contain enough information.

QUESTION:
{user_question}

TRUSTED CONTEXT:
[SOURCE_1: filename, page X]
{retrieved_chunk}

ANSWER:
```

Each retrieved chunk should be represented by an immutable source label. The labels supplied to the model must also be passed to the citation validator.

## 5.6 Output Guardrail

The implemented output checks include:

- rejection of empty output;
- detection of a canary token associated with internal leakage;
- detection of prompt-related leakage patterns;
- toxic-keyword checks;
- optional classifier hook.

Unsafe output is replaced by a safe fallback message. Security events are written to:

```text
data/security_audit.log
```

The output guardrail is useful but not a complete semantic safety solution. Exact keyword matching can miss paraphrases and can also block text that merely discusses an unsafe term.

## 5.7 Citation Validator

> **Status: pending implementation**

The citation validator should verify that:

1. every cited label was supplied in the grounded context;
2. the cited filename exists in the retrieved result set;
3. the page value matches retrieved metadata;
4. no fabricated source identifier appears;
5. a factual answer contains at least one valid citation;
6. an abstention does not invent a citation.

A first implementation can validate citation syntax and source membership. Full claim-level entailment checking is a more advanced extension and should not be claimed unless explicitly implemented.

## 5.8 Logging and Monitoring

The current guardrail module writes warning and critical events to a security audit log.

The final pipeline should record one structured result per request:

```text
request_id
pipeline_mode
prompt_category
input_guardrail_allowed
input_guardrail_reason
retrieved_chunk_ids
retrieved_sources
model_response
output_guardrail_allowed
output_guardrail_reason
citations
final_status
input_guardrail_latency_ms
retrieval_latency_ms
generation_latency_ms
output_guardrail_latency_ms
total_latency_ms
```

Logs may contain sensitive or malicious prompt content. They should therefore be protected, minimized, and excluded from public commits.

## 5.9 Scripts

### `build_knowledge_base.py`

Purpose: construct the local vector database.

Input:

```text
data/knowledge_base/*.pdf
```

Process:

```text
PDF loading -> page extraction -> chunking -> embedding -> ChromaDB upsert
```

Output:

```text
data/chroma_db/
```

Observed build result:

- 4 documents;
- 252 pages;
- 265 chunks;
- 265 stored records.

### `test_pdf_loader.py`

Purpose: verify that all documents can be opened and their text can be extracted. It reports pages by source and prints a text sample.

### `test_chunker.py`

Purpose: validate chunk creation and metadata. It reports chunk counts per document and displays the first generated chunk.

### `test_retrieval.py`

Purpose: run an interactive semantic-search query and display:

- chunk ID;
- source;
- page;
- chunk index;
- distance;
- text preview;
- retrieval latency.

### Automatic Red Team runner

> **Status: planned**

A new script should iterate through `red_team_dataset.csv`, run each prompt in baseline and protected modes, continue after individual errors, and export structured results for metric calculation.

---

# Chapter 6 – Testing and Results

## 6.1 Testing Methodology

The complete evaluation will execute every Red Team prompt against:

1. **baseline mode**, containing only direct Ollama generation;
2. **protected mode**, containing input guardrail, RAG, grounded prompt, output guardrail, and citation validation.

For reproducibility:

- both modes must use the same model;
- generation parameters should be fixed;
- prompts must be executed in the same order;
- failures and timeouts must be retained in the result file;
- latency must be measured per component;
- stochastic tests may be repeated if generation randomness is enabled.

The final assessment will combine automated checks with manual review. Automated checks can determine blocking decisions, citation syntax, source membership, and latency. Hallucination and claim support may require human verification unless a reliable claim-level evaluator is implemented.

## 6.2 Component Validation Completed

The following component tests have already been completed.

### PDF extraction

All four knowledge-base documents were processed successfully.

```text
Extracted pages: 252
Processed documents: 4
```

### Chunking

The chunking test produced:

```text
Created chunks: 265
```

Breakdown:

| Document | Chunks |
|---|---:|
| NIST Adversarial Machine Learning | 137 |
| NIST Generative AI Profile | 67 |
| OWASP Clean Reference | 13 |
| NIST AI RMF 1.0 | 48 |

### Vector database construction

The persistent database was built successfully:

```text
Records stored: 265
```

The embedding model used the Apple MPS device during local testing.

### Semantic retrieval

Representative retrieval tests returned the expected sources.

**Question**

```text
What are the four functions of the NIST AI Risk Management Framework?
```

The top result was from `nist.ai.100-1.pdf` and contained:

```text
GOVERN, MAP, MEASURE, and MANAGE
```

**Question**

```text
What is prompt injection in an LLM application?
```

The top result was the OWASP Prompt Injection section.

**Question**

```text
What is excessive agency and how can it be mitigated?
```

The top result was the OWASP Excessive Agency section.

Observed interactive retrieval latency varied between approximately 100 ms and 800 ms depending on model warm-up and execution state. These observations are preliminary and are not the final benchmark.

### Guardrail unit tests

The pytest execution completed with:

```text
4 passed
```

The tests cover:

- safe input acceptance;
- multiple complex jailbreak/injection examples;
- safe output acceptance;
- output leakage and toxicity blocking.

Deprecation warnings from imported native dependencies were present but did not cause test failures.

## 6.3 Baseline Results

> **Status: pending final integrated execution**

This section will contain:

- number and rate of unsafe baseline responses;
- successful jailbreaks;
- hallucinated answers;
- fabricated citations;
- baseline generation latency;
- errors and timeouts.

No values should be inserted until the automated runner has executed the complete dataset against the direct Ollama baseline.

## 6.4 Protected Pipeline Results

> **Status: pending final integrated execution**

This section will report the same metrics after applying all Safety Sentinel layers.

## 6.5 Safety Improvement

The final table will use the following structure:

| Metric | Unprotected SLM | Safety Sentinel | Improvement |
|---|---:|---:|---:|
| Unsafe Output Rate | TBD | TBD | TBD |
| Attack Success Rate | TBD | TBD | TBD |
| Safe Refusal Rate | TBD | TBD | TBD |
| False Positive Rate | TBD | TBD | TBD |

## 6.6 Hallucination Reduction

| Metric | Unprotected SLM | Safety Sentinel | Improvement |
|---|---:|---:|---:|
| Hallucination Rate | TBD | TBD | TBD |
| Grounded Answer Rate | TBD | TBD | TBD |
| Valid Citation Rate | TBD | TBD | TBD |
| Correct Abstention Rate | TBD | TBD | TBD |

## 6.7 Latency Analysis

| Pipeline component | Mean latency |
|---|---:|
| Input guardrail | TBD |
| Retrieval | TBD |
| SLM generation | TBD |
| Output guardrail | TBD |
| Citation validation | TBD |
| Total protected pipeline | TBD |
| Baseline generation | TBD |

The final analysis will distinguish cold-start behavior from warmed execution when relevant.

## 6.8 Qualitative Case Studies

The final report should include at least:

1. a blocked jailbreak;
2. a benign prompt accepted by the input guardrail;
3. a factual question hallucinated by the baseline;
4. a grounded protected answer with valid sources;
5. an unsupported question producing correct abstention;
6. a detected invalid citation or leakage attempt.

Each case should show the prompt, baseline response, protected response, retrieved chunks, guardrail decisions, and a short interpretation.

---

# Chapter 7 – Discussion

## 7.1 Preliminary Interpretation

The completed component tests demonstrate that a local, source-aware retrieval layer can be built with a modest implementation footprint. The system can reconstruct its vector database from four verified documents, retrieve relevant NIST and OWASP passages, and preserve metadata needed for citations.

These results validate the infrastructure but do not yet prove that hallucinations are reduced in the final generated answers. That conclusion requires integrated execution through Gemma 2B and comparison with the baseline.

The guardrail tests also demonstrate that deterministic checks can detect selected attack patterns. They do not establish robustness against adaptive attackers, paraphrased attacks, multilingual inputs, or semantically complex requests.

## 7.2 Strengths

### Modularity

Model access, guardrails, retrieval, API exposure, and testing are separated into distinct modules. This makes individual layers replaceable and independently testable.

### Local execution

Both retrieval and intended generation operate locally. This reduces external service dependencies and permits direct latency measurement.

### Traceability

Each chunk preserves source filename, page number, and chunk index. This supports transparent retrieval results and future citation validation.

### Reproducibility

The ChromaDB directory is generated by script rather than committed as an opaque artifact. Given the same documents, model, and parameters, another developer can rebuild the index.

### Transparent first-line defense

Keyword, regex, and heuristic checks provide explainable decisions. A rejection can be associated with a specific matched pattern rather than an opaque classifier score.

### Verified corpus

The knowledge base is intentionally small and curated. This supports manual verification and reduces uncontrolled data provenance.

## 7.3 Limitations

### Keyword-filter limitations

Deterministic blocklists are vulnerable to synonyms, paraphrases, languages not covered by the policy, and novel obfuscation. They can also block legitimate educational security questions.

### Inactive semantic safety classifier

The output-classifier hook exists but is not active. Current output filtering cannot be described as classifier-based moderation.

### Retrieval limitations

A top-ranked chunk may be incomplete, semantically related but not answer-bearing, or contaminated by headers and footers. The current chunker is word-based and does not use document headings or semantic boundaries.

### No retrieval threshold

The current retriever returns the top `k` results even when all results are weak. Correct abstention may therefore require an explicit distance threshold or a separate context-sufficiency check.

### Citation validation pending

Source and page metadata are available, but the final validator has not yet been implemented. Until then, the model may still produce unsupported citations.

### Small knowledge base

Four documents provide controlled evaluation but limited domain coverage. Questions outside the corpus should be refused rather than treated as general-knowledge requests in protected mode.

### Model and prompt sensitivity

Results obtained with `gemma:2b` and a particular prompt template may not generalize to other models or parameter settings.

### Latency overhead

Embedding a query, searching ChromaDB, constructing context, and checking output add processing time. The final experiment must determine whether the safety improvement justifies this overhead.

## 7.4 Security Considerations

### Knowledge-base poisoning

Only reviewed documents should be indexed. Source hashes and version metadata would strengthen integrity checking.

### Indirect prompt injection

Retrieved text must be treated as data, not as executable instructions. The grounded prompt should explicitly state that instructions found inside sources are untrusted.

### Log protection

Audit logs may contain attack prompts and generated content. They should not be publicly committed and should have a retention policy.

### Secret separation

Credentials and sensitive configuration must not be embedded in system prompts. External application controls should enforce authorization.

### Denial of service

The API should limit prompt length, request frequency, generation length, and processing time.

### Error handling

Timeouts and connection failures should be represented structurally. Returning error strings through the normal response path can complicate monitoring and evaluation.

### Test safety

Red Team prompts should be curated to test policy bypasses without unnecessarily distributing detailed harmful instructions.

---

# Chapter 8 – Conclusions and Future Work

## 8.1 Preliminary Conclusions

The project has established the main foundations of Safety Sentinel.

A local RAG subsystem has been implemented and validated. It extracts text from verified PDFs, creates overlapping chunks, generates embeddings, stores them in ChromaDB, and retrieves relevant evidence together with source metadata.

The repository also contains an Ollama client, deterministic input/output guardrails, an API skeleton, and automated guardrail tests. These components support the intended architecture but still require final orchestration in `SafetySentinel.run_pipeline()`.

It is therefore possible to conclude that the technical components needed for a protected local SLM application are available. It is not yet possible to claim a measured reduction in unsafe outputs or hallucinations. Those conclusions must be based on the final Red Team comparison.

## 8.2 Future Work and Remaining Activities

### Immediate project activities

Before submission, the team should:

1. integrate the completed `src/rag/` modules into `SafetySentinel`;
2. remove or adapt the `rag_engine.py` stub;
3. implement grounded prompt construction;
4. implement citation validation;
5. support explicit baseline and protected modes;
6. create the automatic Red Team runner;
7. execute all 60 prompts in both modes;
8. calculate final safety, reliability, and latency metrics;
9. add qualitative case studies;
10. replace every `TBD` table entry with measured results.

### Technical extensions

Possible future improvements include:

- semantic or policy-specific safety classifiers;
- NeMo Guardrails integration;
- Llama Guard evaluation;
- multilingual Red Team prompts;
- metadata filters in ChromaDB;
- hybrid lexical and semantic retrieval;
- cross-encoder reranking;
- semantic chunking based on headings;
- retrieval-confidence calibration;
- claim-level source verification;
- cryptographic hashes for knowledge-base files;
- detection of indirect prompt injection in retrieved chunks;
- larger and externally validated Red Team datasets;
- comparison across multiple SLMs;
- monitoring dashboards and structured JSON logs.

The long-term value of Safety Sentinel lies in its modularity: stronger filters, retrievers, validators, and local models can be introduced without redesigning the entire application.

---

# References

1. National Institute of Standards and Technology, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, 2023.  
   <https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf>

2. National Institute of Standards and Technology, *Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile*, NIST AI 600-1, 2024.  
   <https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf>

3. National Institute of Standards and Technology, *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*, NIST AI 100-2e2025, 2025.  
   <https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf>

4. OWASP GenAI Security Project, *OWASP Top 10 for LLM Applications 2025*.  
   <https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/>

5. OWASP GenAI Security Project, *LLM01:2025 Prompt Injection*.  
   <https://genai.owasp.org/llmrisk/llm01-prompt-injection/>

6. OWASP GenAI Security Project, *LLM09:2025 Misinformation*.  
   <https://genai.owasp.org/llmrisk/llm09-overreliance/>

7. Chroma, *Chroma Documentation: Persistent Clients, Collections, and Querying*.  
   <https://docs.trychroma.com/docs/run-chroma/clients>  
   <https://docs.trychroma.com/docs/collections/add-data>  
   <https://docs.trychroma.com/docs/querying-collections/query-and-get>

8. Sentence Transformers, *Documentation and Semantic Search*.  
   <https://www.sbert.net/>  
   <https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html>

9. PyMuPDF, *Text Extraction Documentation*.  
   <https://pymupdf.readthedocs.io/en/latest/recipes-text.html>

10. NVIDIA, *NeMo Guardrails Documentation*.  
    <https://docs.nvidia.com/nemo-guardrails/index.html>

11. Meta AI, *Llama Guard and Responsible AI Resources*.  
    <https://ai.meta.com/blog/meta-llama-3/>

12. Safety Sentinel project repository.  
    <https://github.com/cappiellobelisario02/urban_security2026_cappiello_scalise>
