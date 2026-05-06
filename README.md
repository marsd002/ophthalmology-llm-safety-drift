# Ophthalmology LLM Safety-Netting Drift Study

This repository contains the code and locked corpus for a prospective,
longitudinal, multi-model surveillance study of safety-netting behaviour in
patient-facing large language model (LLM) responses to vision-threatening
ophthalmic symptoms.

The study is pre-registered on the Open Science Framework. The pre-registration,
the locked patient question corpus, and all generated raw responses are stored
on OSF.

## Structure

| File | Purpose |
|---|---|
| `locked_corpus_v1.json` | The 48 patient questions and one system prompt that drive the study. Hashed and locked at T0. |
| `collect.py` | Data collection script. Sends each (model × question) cell N times to all six study models and saves responses to `data/responses_<timepoint>.jsonl`. |
| `verify_apis.py` | Connectivity check for the four model providers. Run after any environment change. |
| `setup_env.py` | One-time interactive script that writes the `.env` API-key file. |
| `archive_pilot_data.py` | Moves pre-T0 SMOKE / PILOT / VERIFY / TEST data into `archive/` before launching T0. |
| `rate.py` | Streamlit rating interface for two masked ophthalmologist raters and a senior adjudicator. |

## Folders excluded from version control

`data/`, `ratings/`, `archive/`, `logs/`, and the `.env` API-key file are
deliberately not tracked by Git. The raw responses and rater judgements are
stored on OSF; secrets stay on the lead investigator's machine.

## Models

| Family | Models |
|---|---|
| OpenAI | gpt-4o, gpt-5 |
| Anthropic | claude-sonnet-4-6, claude-opus-4-6 |
| Google | gemini-2.5-pro |
| Meta (open-weights, frozen control) | llama-3.1-8b via Ollama |

## Reproducing the pipeline

1. `python3.13 -m pip install --user openai anthropic google-genai requests python-dotenv streamlit`
2. Install Ollama and pull `llama3.1:8b` (one-time, ~5 GB).
3. `python3.13 setup_env.py` — paste your three API keys when prompted.
4. `python3.13 verify_apis.py` — confirm all four providers reachable.
5. `python3.13 collect.py --timepoint T0 --reps 10` — full T0 collection.
6. `python3.13 -m streamlit run rate.py` — launch the rating app.

## Methodological deviations

- **OSF Addendum 1:** GPT-5 cannot be queried at temperature 0.7 or with
  max_tokens 1024. Vendor-imposed constraint. GPT-5 calls use
  temperature=1.0 (the only value OpenAI permits) and
  max_completion_tokens=8192. All other models use the originally pre-registered
  parameters. The deviation is logged transparently in every JSONL response row.

## Licence

Code: MIT. Data and corpus: CC-BY 4.0.
