# Safety Sentinel: LLM Guardrails & Anti-Hallucination RAG

## Introduction

The Safety Sentinel project provides a robust middleware pipeline designed to enhance the safety and reliability of interactions with Small Language Models (SLMs). It integrates essential components like advanced I/O Guardrails to prevent LLM jailbreaks and toxic outputs, and an Anti-Hallucination RAG (Retrieval-Augmented Generation) layer to ground responses in factual knowledge, thereby reducing the risk of generating inaccurate or misleading information.

This system acts as a crucial defense layer, ensuring that user prompts are safe and compliant before reaching the SLM, and that the SLM's responses adhere to safety policies and remain accurate. By combining proactive input filtering with reactive output validation and factual retrieval, Safety Sentinel aims to create a more secure and trustworthy SLM application.

## Prerequisites

To set up and run the Safety Sentinel locally, you will need:

*   **Python 3.11** (or newer) installed on your system.
*   **Ollama**, a platform for running large language models locally.
*   **FastAPI** and **Uvicorn** (included in `requirements.txt`).

## Local Environment Setup

Follow the instructions below for your specific operating system to set up the environment and install Ollama.

### Windows

1.  **Open PowerShell or Command Prompt.**
2.  **Create a Python virtual environment:**
    ```bash
    py -3.11 -m venv venv
    ```
3.  **Activate the virtual environment:**
    ```bash
    .\venv\Scripts\activate
    ```
4.  **Install Ollama:**
    *   Download the official Windows installer from the [Ollama website](https://ollama.com/download/windows).
    *   Run the installer and follow the on-screen instructions.
    *   Verify installation by running `ollama run llama2` in a new terminal.

### macOS

1.  **Open Terminal.**
2.  **Create a Python virtual environment:**
    ```bash
    python3.11 -m venv venv
    ```
3.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```
4.  **Install Ollama:**
    *   **Using Homebrew (recommended):**
        ```bash
        brew install ollama
        ```
    *   **Manual Download:** Download and install the application from the [Ollama website](https://ollama.com/download/mac).
    *   Verify installation by running `ollama run llama2`.

### Linux

1.  **Open Terminal.**
2.  **Create a Python virtual environment:**
    ```bash
    python3.11 -m venv venv
    ```
3.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```
4.  **Install Ollama:**
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
    *   Verify installation by running `ollama run llama2`.

## Dependencies

Once your virtual environment is activated, install the required Python dependencies, including FastAPI and Uvicorn:

```bash
pip install -r requirements.txt
```

## Model Initialization

Before running the application, you need to pull and run the local language model using Ollama. We recommend `gemma:2b`:

```bash
ollama run gemma:2b
```

Ensure this model is running in the background or in a separate terminal before executing the `Safety Sentinel`.

## How to Run

To start the Safety Sentinel REST API development server, ensure your virtual environment is activated and run:

```bash
uvicorn src.api:app --reload
```

Output logs, including security audit warnings and errors, will be appended to `data/security_audit.log`.

## API Endpoints

The Safety Sentinel API provides the following endpoint:

### `POST /api/chat`

Processes a user prompt through the Safety Sentinel pipeline, applies guardrails, and retrieves RAG-augmented responses.

**Request Body (`application/json`):**

```json
{
  "user_prompt": "Your question or statement here."
}
```

**Example Response (`application/json`):**

```json
{
  "status": "success",
  "response": "The model's safe and augmented response.",
  "latency_ms": 123.45
}
```

### Swagger UI

Once the API server is running, you can access the interactive API documentation and test the endpoints directly in your browser by navigating to:

`http://127.0.0.1:8000/docs`
