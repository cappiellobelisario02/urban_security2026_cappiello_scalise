import unittest
from src.llm_client import OllamaClient


class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()

    def test_generate_response_basic(self):
        prompt = "Hello, how are you?"
        response = self.client.generate_response(prompt)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        # Basic check to ensure it's not an error message
        self.assertNotIn("[TIMEOUT]", response)
        self.assertNotIn("[CONNECTION_ERROR]", response)
        self.assertNotIn("[REQUEST_ERROR]", response)
        self.assertNotIn("[API_ERROR]", response)

    def test_generate_response_empty_prompt(self):
        prompt = ""
        response = self.client.generate_response(prompt)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertNotIn("[TIMEOUT]", response)
        self.assertNotIn("[CONNECTION_ERROR]", response)
        self.assertNotIn("[REQUEST_ERROR]", response)
        self.assertNotIn("[API_ERROR]", response)


if __name__ == "__main__":
    unittest.main()
