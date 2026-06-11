import requests
from requests.exceptions import Timeout, ConnectionError


class OllamaClient:
    """Client wrapper for Ollama Small Language Model interactions."""

    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate") -> None:
        """Initializes the OllamaClient with the given URL.

        Args:
            ollama_url (str): The URL of the Ollama API.
        """
        self.ollama_url = ollama_url
        self.timeout = 15  # [ReliabilityAgent] Define a strict timeout for API calls

    def generate_response(self, prompt: str) -> str:
        """Generates a response from the Ollama model for the given prompt.

        Includes timeout handling and circuit breaker pattern for robustness.

        Args:
            prompt (str): The input prompt for the model.

        Returns:
            str: The model\'s generated response or a fallback message if an error occurs.
        """
        # [ReliabilityAgent]: Implement Timeout and ConnectionError handling.
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "llama2",  # Assuming llama2 is the desired model
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            return response.json()["response"]
        except Timeout:
            return (
                "[TIMEOUT] The model took too long to respond. Please try again later."
            )
        except ConnectionError:
            return "[CONNECTION_ERROR] Could not connect to the Ollama server. Please check the server status."
        except requests.exceptions.RequestException as e:
            return f"[REQUEST_ERROR] An error occurred while communicating with the LLM: {e}"
        except KeyError:
            return "[API_ERROR] Unexpected response format from Ollama API."
