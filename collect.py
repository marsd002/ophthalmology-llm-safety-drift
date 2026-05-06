"""
Data collection script for the ophthalmology LLM safety-netting drift study.

Reads the locked corpus, queries each model N times per question at the
configured timepoint, and saves every response immediately to
data/responses_<timepoint>.jsonl.

The script is RESUMABLE: if you stop it (Ctrl+C) and re-run with the same
timepoint, it picks up where it left off, skipping cells already saved
successfully. Failures are also logged so you can re-run later to retry them.

Usage examples
--------------
# Tiny smoke test
python3.13 collect.py --timepoint SMOKE --reps 1 --models llama-3.1-8b --max-questions 2

# Bigger pilot
python3.13 collect.py --timepoint PILOT --reps 3 --models llama-3.1-8b --max-questions 5

# Full T0 collection (all 6 models, all 48 questions, 10 reps each = 2,880 calls)
python3.13 collect.py --timepoint T0 --reps 10

# Just one model at T0
python3.13 collect.py --timepoint T0 --reps 10 --models gpt-4o
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
CORPUS_FILE = PROJECT_DIR / "locked_corpus_v1.json"
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")

# ----- Locked study parameters -----
TEMPERATURE = 0.7
MAX_TOKENS = 1024
# Reasoning models (gpt-5, o-series) consume tokens for internal "thinking"
# from the same budget as their visible answer. We give them a much larger
# budget so the visible answer has room to be substantive after reasoning.
MAX_COMPLETION_TOKENS_REASONING = 8192
TIMEOUT_SECONDS = 300  # reasoning models can take a while
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

MODELS = {
    "gpt-4o":            {"family": "openai",    "model_id": "gpt-4o"},
    "gpt-5":             {"family": "openai",    "model_id": "gpt-5"},
    "claude-sonnet-4-6": {"family": "anthropic", "model_id": "claude-sonnet-4-6"},
    "claude-opus-4-6":   {"family": "anthropic", "model_id": "claude-opus-4-6"},
    "gemini-2.5-pro":    {"family": "google",    "model_id": "gemini-2.5-pro"},
    "llama-3.1-8b":      {"family": "ollama",    "model_id": "llama3.1:8b"},
}


def is_openai_reasoning_model(model_id: str) -> bool:
    """Reasoning models (gpt-5, o-series) have restricted parameters:
    - require max_completion_tokens instead of max_tokens
    - only accept temperature = 1.0 (default)
    """
    prefixes = ("gpt-5", "o1", "o3", "o4")
    return any(model_id.startswith(p) for p in prefixes)


def effective_temperature(family: str, model_id: str) -> float:
    """Return the temperature actually used for this model.

    Most models use the protocol value (TEMPERATURE = 0.7). OpenAI reasoning
    models force temperature = 1.0 — this is a vendor-imposed constraint,
    documented as a deviation in the OSF addendum.
    """
    if family == "openai" and is_openai_reasoning_model(model_id):
        return 1.0
    return TEMPERATURE


def effective_max_tokens(family: str, model_id: str) -> int:
    """Return the token budget actually used for this model.

    Most models use MAX_TOKENS (1024). OpenAI reasoning models need a much
    larger budget because they consume tokens for internal reasoning before
    producing the visible answer.
    """
    if family == "openai" and is_openai_reasoning_model(model_id):
        return MAX_COMPLETION_TOKENS_REASONING
    return MAX_TOKENS


# ----- API callers -----

def call_openai(model_id, system_prompt, user_prompt):
    from openai import OpenAI
    client = OpenAI()
    t0 = time.time()

    kwargs = dict(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    # Reasoning models (gpt-5, o-series) restrict temperature to 1.0 (default)
    # and require max_completion_tokens. Older chat models accept both.
    if is_openai_reasoning_model(model_id):
        kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS_REASONING
        # Do not pass temperature; OpenAI rejects any value other than 1.0
    else:
        kwargs["max_tokens"] = MAX_TOKENS
        kwargs["temperature"] = TEMPERATURE

    r = client.chat.completions.create(**kwargs)
    return {
        "response_text": r.choices[0].message.content,
        "response_tokens": r.usage.completion_tokens if r.usage else None,
        "model_version_string": r.model,
        "latency_ms": int((time.time() - t0) * 1000),
        "api_response_id": r.id,
    }


def call_anthropic(model_id, system_prompt, user_prompt):
    from anthropic import Anthropic
    client = Anthropic()
    t0 = time.time()
    r = client.messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return {
        "response_text": r.content[0].text,
        "response_tokens": r.usage.output_tokens if r.usage else None,
        "model_version_string": r.model,
        "latency_ms": int((time.time() - t0) * 1000),
        "api_response_id": r.id,
    }


def call_google(model_id, system_prompt, user_prompt):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    t0 = time.time()
    r = client.models.generate_content(
        model=model_id,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        ),
    )
    return {
        "response_text": r.text or "",
        "response_tokens": (r.usage_metadata.candidates_token_count
                            if r.usage_metadata else None),
        "model_version_string": model_id,
        "latency_ms": int((time.time() - t0) * 1000),
        "api_response_id": None,
    }


def call_ollama(model_id, system_prompt, user_prompt):
    t0 = time.time()
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        },
        timeout=TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    data = r.json()
    return {
        "response_text": data["message"]["content"],
        "response_tokens": data.get("eval_count"),
        "model_version_string": model_id,
        "latency_ms": int((time.time() - t0) * 1000),
        "api_response_id": None,
    }


CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
    "ollama": call_ollama,
}


# ----- Resume support -----

def get_completed_cells(jsonl_path: Path) -> set:
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("response_text") and not row.get("error"):
                    done.add((row["model_label"], row["prompt_id"], row["repetition"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ----- Main collection loop -----

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_row(fout, row):
    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    fout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect LLM responses for the ophthalmology safety-netting study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--timepoint", required=True,
                        help="Timepoint label, e.g., SMOKE, PILOT, T0, T+3, T+6, MV-GPT5")
    parser.add_argument("--reps", type=int, default=10,
                        help="Repetitions per (model, question) cell. Default 10.")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help=f"Subset of models. Choices: {', '.join(MODELS.keys())}")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit to first N questions of corpus (smoke testing).")
    parser.add_argument("--questions", nargs="+", default=None,
                        help="Specific prompt_ids to query (e.g., GLA_T1 COR_T2).")
    args = parser.parse_args()

    if not CORPUS_FILE.exists():
        sys.exit(f"ERROR: corpus file not found: {CORPUS_FILE}")

    corpus = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
    system_prompt = corpus["system_prompt"]["text"]
    system_prompt_sha = corpus["system_prompt"]["sha256"]
    questions = corpus["questions"]
    if args.questions:
        questions = [q for q in questions if q["prompt_id"] in args.questions]
    if args.max_questions:
        questions = questions[: args.max_questions]

    for m in args.models:
        if m not in MODELS:
            sys.exit(f"ERROR: unknown model '{m}'. Valid: {', '.join(MODELS.keys())}")

    out_path = DATA_DIR / f"responses_{args.timepoint}.jsonl"
    completed = get_completed_cells(out_path)
    total_cells = len(args.models) * len(questions) * args.reps
    todo_cells = total_cells - len(completed)

    print()
    print(f"Timepoint:      {args.timepoint}")
    print(f"Models:         {', '.join(args.models)}")
    print(f"Questions:      {len(questions)}")
    print(f"Repetitions:    {args.reps}")
    print(f"Total cells:    {total_cells}")
    print(f"Already done:   {len(completed)}")
    print(f"Cells to do:    {todo_cells}")
    print(f"Output file:    {out_path}")
    print(f"Corpus SHA-256: {corpus['corpus_sha256']}")
    print("=" * 70)

    if todo_cells == 0:
        print("Nothing to do — all cells already collected for this timepoint.")
        return

    new_count, fail_count = 0, 0

    with out_path.open("a", encoding="utf-8") as fout:
        for model_label in args.models:
            cfg = MODELS[model_label]
            caller = CALLERS[cfg["family"]]

            for q in questions:
                for rep in range(1, args.reps + 1):
                    cell = (model_label, q["prompt_id"], rep)
                    if cell in completed:
                        continue

                    last_error = None
                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            result = caller(cfg["model_id"], system_prompt, q["prompt_text"])
                            row = {
                                "response_id": str(uuid.uuid4()),
                                "timepoint": args.timepoint,
                                "collection_datetime_utc": utc_now(),
                                "model_label": model_label,
                                "model_family": cfg["family"],
                                "model_id_requested": cfg["model_id"],
                                "model_version_string": result["model_version_string"],
                                "system_prompt_sha256": system_prompt_sha,
                                "temperature": effective_temperature(cfg["family"], cfg["model_id"]),
                                "max_tokens": effective_max_tokens(cfg["family"], cfg["model_id"]),
                                "prompt_id": q["prompt_id"],
                                "subspecialty_code": q["subspecialty_code"],
                                "subspecialty_name": q["subspecialty_name"],
                                "question_type": q["question_type"],
                                "prompt_sha256": q["prompt_sha256"],
                                "repetition": rep,
                                "response_text": result["response_text"],
                                "response_tokens": result.get("response_tokens"),
                                "latency_ms": result["latency_ms"],
                                "api_response_id": result.get("api_response_id"),
                                "attempts": attempt,
                                "error": None,
                            }
                            write_row(fout, row)
                            new_count += 1
                            print(f"  [{model_label:18}] {q['prompt_id']:8} rep {rep:>2}  "
                                  f"{result['latency_ms']:>5} ms  ok")
                            break
                        except Exception as e:
                            last_error = f"{type(e).__name__}: {e}"
                            if attempt < MAX_RETRIES:
                                wait = RETRY_BACKOFF_SECONDS * attempt
                                print(f"  [{model_label:18}] {q['prompt_id']:8} rep {rep:>2}  "
                                      f"attempt {attempt} failed; waiting {wait}s")
                                time.sleep(wait)
                            else:
                                row = {
                                    "response_id": str(uuid.uuid4()),
                                    "timepoint": args.timepoint,
                                    "collection_datetime_utc": utc_now(),
                                    "model_label": model_label,
                                    "model_family": cfg["family"],
                                    "model_id_requested": cfg["model_id"],
                                    "model_version_string": None,
                                    "system_prompt_sha256": system_prompt_sha,
                                    "temperature": effective_temperature(cfg["family"], cfg["model_id"]),
                                    "max_tokens": MAX_TOKENS,
                                    "prompt_id": q["prompt_id"],
                                    "subspecialty_code": q["subspecialty_code"],
                                    "subspecialty_name": q["subspecialty_name"],
                                    "question_type": q["question_type"],
                                    "prompt_sha256": q["prompt_sha256"],
                                    "repetition": rep,
                                    "response_text": None,
                                    "response_tokens": None,
                                    "latency_ms": None,
                                    "api_response_id": None,
                                    "attempts": attempt,
                                    "error": last_error,
                                }
                                write_row(fout, row)
                                fail_count += 1
                                print(f"  [{model_label:18}] {q['prompt_id']:8} rep {rep:>2}  "
                                      f"FAILED after {attempt} attempts: {last_error[:80]}")

    print("=" * 70)
    print(f"Done. New successful: {new_count}. Failed (logged): {fail_count}.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
