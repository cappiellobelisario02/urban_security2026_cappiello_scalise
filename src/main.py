from src.llm_client import OllamaClient
from src.guardrails import SafetyGuard
from src.rag_engine import VectorDB


class SafetySentinel:
    """Orchestrator class coordinating the safety guardrails and RAG pipeline."""

    def __init__(self) -> None:
        """Initializes the SafetySentinel orchestrator and its pipeline stages."""
        self.client = OllamaClient()
        self.guard = SafetyGuard()
        self.rag = VectorDB()

    def run_pipeline(self, user_prompt: str) -> str:
        """Runs the entire middleware pipeline on the user prompt.

        Includes input guardrails, RAG context retrieval, LLM response generation,
        and output guardrails.

        Args:
            user_prompt (str): The initial prompt submitted by the user.

        Returns:
            str: The final safe model response or a fallback violation message.
        """
        pass
