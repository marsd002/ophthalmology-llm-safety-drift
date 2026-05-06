"""
Quick connectivity check for the ophthalmology LLM study.

Sends a trivial one-word test prompt to each of the four model providers
(OpenAI, Anthropic, Google, Llama via Ollama) and reports back which
work and which fail. Run this once after .env is set up. If any provider
fails, the printed error will tell us what to fix.
"""

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

TEST_SYSTEM = "You are a helpful assistant."
TEST_PROMPT = "Say the single word 'ok' and nothing else."

# ANSI colours for nicer terminal output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def check_openai():
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": TEST_SYSTEM},
            {"role": "user", "content": TEST_PROMPT},
        ],
        max_tokens=10,
    )
    return response.choices[0].message.content.strip(), response.model


def check_anthropic():
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        system=TEST_SYSTEM,
        messages=[{"role": "user", "content": TEST_PROMPT}],
    )
    return response.content[0].text.strip(), response.model


def check_google():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=TEST_PROMPT,
        config=types.GenerateContentConfig(
            system_instruction=TEST_SYSTEM,
            max_output_tokens=10,
        ),
    )
    return (response.text or "").strip(), "gemini-2.5-pro"


def check_ollama():
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.1:8b",
            "messages": [
                {"role": "system", "content": TEST_SYSTEM},
                {"role": "user", "content": TEST_PROMPT},
            ],
            "stream": False,
            "options": {"num_predict": 10},
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip(), "llama3.1:8b"


CHECKS = [
    ("OpenAI",         check_openai),
    ("Anthropic",      check_anthropic),
    ("Google Gemini",  check_google),
    ("Llama (Ollama)", check_ollama),
]


def main() -> None:
    print()
    print("API connectivity check")
    print("=" * 60)

    results = []
    for name, fn in CHECKS:
        print(f"\nTesting {name} …", end=" ", flush=True)
        try:
            t0 = time.time()
            reply, model_str = fn()
            elapsed_ms = int((time.time() - t0) * 1000)
            print(f"{GREEN}OK{RESET}")
            print(f"  model:  {model_str}")
            print(f"  reply:  {reply!r}")
            print(f"  time:   {elapsed_ms} ms")
            results.append((name, True, None))
        except Exception as e:
            print(f"{RED}FAIL{RESET}")
            err_str = f"{type(e).__name__}: {e}"
            print(f"  error:  {err_str}")
            results.append((name, False, err_str))

    print()
    print("=" * 60)
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"Summary: {n_ok} of {len(results)} providers reachable.")
    if n_ok < len(results):
        print("\nFailed providers:")
        for name, ok, err in results:
            if not ok:
                print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("All providers ready. You can safely move on to the collection script.")


if __name__ == "__main__":
    main()
