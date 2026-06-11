from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from src.main import SafetySentinel


# [SchemaAgent]: Define Pydantic models for request and response validation
class PromptRequest(BaseModel):
    """Request model for processing a user prompt."""

    user_prompt: str


class PipelineResponse(BaseModel):
    """Response model for the output of the Safety Sentinel pipeline."""

    status: str
    response: str
    latency_ms: float


# [RouteAgent]: Initialize FastAPI app
app = FastAPI(
    title="Safety Sentinel API",
    description="Robust REST API for LLM Guardrails and Anti-Hallucination RAG pipeline.",
    version="1.0.0",
)

# [RouteAgent]: Instantiate the SafetySentinel orchestrator at startup
safety_sentinel = SafetySentinel()


@app.post("/api/chat", response_model=PipelineResponse)
async def chat_endpoint(request: PromptRequest) -> PipelineResponse:
    """Processes a user prompt through the Safety Sentinel pipeline.

    Args:
        request (PromptRequest): The incoming request containing the user prompt.

    Returns:
        PipelineResponse: The structured response from the pipeline, including status,
                          the LLM's response, and execution latency.

    Raises:
        HTTPException:
            - 400 if the input is blocked by guardrails.
            - 500 for internal errors like LLM timeouts or connection issues.
    """
    start_time = time.perf_counter()
    try:
        # Run the SafetySentinel pipeline
        result = safety_sentinel.run_pipeline(request.user_prompt)

        # [RouteAgent]: Implement robust error handling
        if (
            "Jailbreak attempt detected." in result
            or "Malicious keyword detected." in result
            or "Obfuscated malicious keyword detected." in result
            or "Instruction override attempt detected." in result
            or "Roleplay/Persona jailbreak attempt detected." in result
            or "Encoded payload violation:" in result
            or "Excessive special characters/obfuscation pattern detected." in result
        ):
            raise HTTPException(status_code=400, detail=result)
        elif (
            "[TIMEOUT]" in result
            or "[CONNECTION_ERROR]" in result
            or "[REQUEST_ERROR]" in result
            or "[API_ERROR]" in result
        ):
            raise HTTPException(status_code=500, detail=result)

        latency_ms = (time.perf_counter() - start_time) * 1000
        return PipelineResponse(
            status="success", response=result, latency_ms=latency_ms
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
