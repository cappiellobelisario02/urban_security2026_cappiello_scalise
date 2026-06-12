#!/usr/bin/env python
"""
run_system.py – One‑click entry point for the Safety Sentinel project.

Features
--------
1. Starts the local Ollama server (if not already running).
2. Ensures required Python dependencies are installed.
3. Provides an interactive menu with two options:
   A) Run the automated Red‑Team benchmark (baseline vs. protected).
   B) Start an interactive CLI chat that runs the full SafetySentinel pipeline.

The script is cross‑platform and relies only on the Python standard library.
"""

import subprocess
import sys
import time
from pathlib import Path


def is_ollama_running() -> bool:
    """Check whether the Ollama HTTP endpoint is reachable."""
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def start_ollama() -> None:
    """Start Ollama in the background if it is not already running."""
    if is_ollama_running():
        print("[+] Ollama server already running.")
        return

    print("[*] Starting Ollama server...")
    try:
        flags = 0
        if sys.platform.startswith("win"):
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    except FileNotFoundError:
        print("[-] Ollama executable not found. Install Ollama and ensure it is on PATH.")
        sys.exit(1)

    # Wait for the server to become ready (max 10 seconds).
    for _ in range(10):
        if is_ollama_running():
            print("[+] Ollama server started.")
            return
        time.sleep(1)
    print("[-] Ollama did not start within the expected time.")
    sys.exit(1)


def ensure_dependencies() -> None:
    """Install missing packages from requirements.txt if they are not present."""
    try:
        import pip  # noqa: F401
    except Exception:
        print("[-] pip not available. Ensure a working Python installation.")
        sys.exit(1)

    req_path = Path("requirements.txt")
    if not req_path.is_file():
        print("[-] requirements.txt not found.")
        sys.exit(1)

    missing = []
    try:
        import transformers  # noqa: F401
    except Exception:
        missing.append("transformers")
    try:
        import chromadb  # noqa: F401
    except Exception:
        missing.append("chromadb")

    if missing:
        print(f"[*] Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
    else:
        print("[+] All required dependencies are installed.")


def run_benchmark() -> None:
    """Execute the benchmark script that evaluates baseline vs. protected pipeline."""
    # Execute the benchmark as a module to ensure proper package imports
    module_name = "scripts.run_benchmark"
    print("[*] Running Red‑Team benchmark (baseline vs. protected)...")
    subprocess.run([sys.executable, "-m", module_name], check=False)


def start_interactive_chat() -> None:
    """Launch the interactive CLI that uses the full SafetySentinel pipeline."""
    # Launch the interactive CLI as a module to respect package imports
    module_name = "src.main"
    print("[*] Starting interactive SafetySentinel chat. Type 'exit' to quit.")
    subprocess.run([sys.executable, "-m", module_name], check=False)


def menu() -> None:
    while True:
        print("\n=== Safety Sentinel – One‑Click Launcher ===")
        print("[A] Run automated Red‑Team benchmark")
        print("[B] Start interactive CLI chat")
        print("[Q] Quit")
        choice = input("Select an option (A/B/Q): ").strip().upper()
        if choice == "A":
            run_benchmark()
        elif choice == "B":
            start_interactive_chat()
        elif choice == "Q":
            print("Good‑bye!")
            break
        else:
            print("Invalid choice – please select A, B, or Q.")


def main() -> None:
    # Verify Python version (must be 3.11.9 as required by the project)
    required_version = (3, 11, 9)
    if sys.version_info[:3] != required_version:
        print(
            f"[ERROR] Python {required_version[0]}.{required_version[1]}.{required_version[2]} "
            f"is required. Current version: {sys.version.split()[0]}"
        )
        sys.exit(1)

    start_ollama()
    ensure_dependencies()
    menu()


if __name__ == "__main__":
    main()
