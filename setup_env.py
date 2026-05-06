"""
One-time setup: write the three API keys to a .env file in this folder.

Run this once. After that, all study scripts read keys from .env automatically.
The .env file is set to owner-only read permissions and should never be committed
to GitHub or shared.
"""

from pathlib import Path
import os
import stat

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"


def mask(value: str) -> str:
    if len(value) < 12:
        return "(too short — check that you pasted the full key)"
    return value[:8] + "…" + value[-4:]


def main() -> None:
    print()
    print("Ophthalmology LLM study — API key setup")
    print("=" * 60)
    print()
    print(f"This will create or overwrite: {ENV_FILE}")
    print()

    if ENV_FILE.exists():
        response = input("A .env file already exists here. Overwrite? (yes/no): ").strip().lower()
        if response != "yes":
            print("Aborted. No changes made.")
            return

    print()
    print("Paste each API key when prompted, then press Enter.")
    print("Don't worry that the text shows on screen — the file we save is")
    print("readable only by you, and we'll never commit it anywhere.")
    print()

    openai_key = input("OpenAI API key (starts with sk-proj-): ").strip()
    anthropic_key = input("Anthropic API key (starts with sk-ant-): ").strip()
    google_key = input("Google AI Studio API key (starts with AIza): ").strip()

    if not (openai_key and anthropic_key and google_key):
        print("\nError: one or more keys were empty. Nothing was saved.")
        return

    content = (
        "# API keys for the ophthalmology LLM safety-netting drift study.\n"
        "# DO NOT commit this file to GitHub or share it with anyone.\n"
        f"OPENAI_API_KEY={openai_key}\n"
        f"ANTHROPIC_API_KEY={anthropic_key}\n"
        f"GOOGLE_API_KEY={google_key}\n"
    )

    ENV_FILE.write_text(content, encoding="utf-8")
    # Owner read/write only (chmod 600)
    os.chmod(ENV_FILE, stat.S_IRUSR | stat.S_IWUSR)

    print()
    print(f"Saved {ENV_FILE}")
    print("File permissions set to owner-only read/write.")
    print()
    print("Verification (each key shown masked):")
    print(f"  OPENAI_API_KEY    = {mask(openai_key)}")
    print(f"  ANTHROPIC_API_KEY = {mask(anthropic_key)}")
    print(f"  GOOGLE_API_KEY    = {mask(google_key)}")
    print()
    print("All set. You can close this window.")


if __name__ == "__main__":
    main()
