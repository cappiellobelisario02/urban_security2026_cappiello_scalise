import unittest
from src.main import SafetySentinel


class TestSafetySentinel(unittest.TestCase):
    """Integration test for the full SafetySentinel pipeline.

    The test ensures that a typical user prompt passes the input guardrail,
    retrieves context from the knowledge base, generates a response via the
    Ollama client, and passes the output guardrail.  It also checks that the
    orchestrator returns a non‑empty string and that the logging file is created.
    """

    def setUp(self) -> None:
        self.sentinel = SafetySentinel()

    def test_run_pipeline_success(self) -> None:
        prompt = "What is the purpose of the AI Risk Management Framework?"
        response = self.sentinel.run_pipeline(prompt)
        # The response should be a non‑empty string and not contain a guardrail error.
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertNotIn("Input blocked", response)
        self.assertNotIn("Output blocked", response)

        # Verify that the audit log file was created and contains at least one entry.
        import os
        log_path = os.path.join("data", "security_audit.log")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Pipeline completed", content)


if __name__ == "__main__":
    unittest.main()