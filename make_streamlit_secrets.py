"""
Helper that prints the exact TOML block you'll paste into Streamlit Cloud's
Secrets configuration. Reads gcp_service_account.json from the project folder,
combines it with the password and sheet key you provide on the command line,
and prints a ready-to-paste TOML block.

Usage:
    python3.13 make_streamlit_secrets.py "<the-password-for-raters>" "<the-google-sheet-key>"

Example:
    python3.13 make_streamlit_secrets.py "ophth-2026-spring" "1AbCDeF1g_2hIjKlMnOpQrStUvWxYz"

Print the output, copy it from terminal, and paste it into the Secrets editor
in your Streamlit Cloud app settings.
"""

import json
import sys
from pathlib import Path


def quote_toml(value: str) -> str:
    """Quote a string for TOML. Use triple-quoted form when the value contains
    line breaks (e.g., the PEM private key)."""
    if "\n" in value:
        # Triple-quoted multi-line string. Escape any embedded triple-quotes.
        escaped = value.replace('"""', '\\"\\"\\"')
        return f'"""\n{escaped}\n"""'
    # Single-line: escape backslashes and double quotes.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    password = sys.argv[1]
    sheet_key = sys.argv[2]

    project_dir = Path(__file__).resolve().parent
    json_path = project_dir / "gcp_service_account.json"

    if not json_path.exists():
        sys.exit(f"ERROR: {json_path} not found. Make sure the service-account "
                 "key is in your project folder.")

    data = json.loads(json_path.read_text())

    print()
    print("# === COPY EVERYTHING BELOW THIS LINE INTO STREAMLIT CLOUD SECRETS ===")
    print()
    print(f"app_password = {quote_toml(password)}")
    print(f"gs_sheet_key = {quote_toml(sheet_key)}")
    print()
    print("[gcp_service_account]")
    for key, value in data.items():
        if isinstance(value, str):
            print(f"{key} = {quote_toml(value)}")
        else:
            print(f"{key} = {json.dumps(value)}")
    print()
    print("# === COPY EVERYTHING ABOVE THIS LINE ===")


if __name__ == "__main__":
    main()
