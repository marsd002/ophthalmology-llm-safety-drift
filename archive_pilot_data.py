"""
One-time archival utility: moves all pre-T0 SMOKE / PILOT / VERIFY / TEST
data and ratings out of the active study folders and into archive/, so the
real T0 collection and rating queue start from a clean slate.

Run this ONCE, immediately before launching T0 collection:
    python3.13 archive_pilot_data.py

The script is conservative: it only moves files whose timepoint label or
rater ID matches a known pre-T0 label, and it never deletes anything.
A summary of what was moved is printed at the end, and the original
filenames are preserved inside archive/.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RATINGS_DIR = PROJECT_DIR / "ratings"
ARCHIVE_DIR = PROJECT_DIR / "archive"

# Timepoint labels considered pre-T0 / non-study data.
PRE_T0_TIMEPOINT_LABELS = ("SMOKE", "PILOT", "PILOT2", "VERIFY_GPT5")

# Rater IDs considered pre-T0 / non-study ratings.
PRE_T0_RATER_IDS = ("TEST",)


def archive_file(src: Path, archive_subdir: str) -> Path:
    """Move src into archive/<archive_subdir>/, preserving its name."""
    target_dir = ARCHIVE_DIR / archive_subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / src.name
    # If a file with the same name already exists, append a timestamp.
    if dest.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = target_dir / f"{src.stem}.{ts}{src.suffix}"
    shutil.move(str(src), str(dest))
    return dest


def main() -> None:
    print()
    print("Archiving pre-T0 data and ratings")
    print("=" * 60)

    moved = []

    # Data files
    if DATA_DIR.exists():
        for label in PRE_T0_TIMEPOINT_LABELS:
            f = DATA_DIR / f"responses_{label}.jsonl"
            if f.exists():
                dest = archive_file(f, "data")
                moved.append(("data", f.name, dest))

    # Ratings files
    if RATINGS_DIR.exists():
        for rater_id in PRE_T0_RATER_IDS:
            f = RATINGS_DIR / f"ratings_{rater_id}.csv"
            if f.exists():
                dest = archive_file(f, "ratings")
                moved.append(("ratings", f.name, dest))

    print()
    if not moved:
        print("Nothing to archive — your data/ and ratings/ folders are already clean.")
        return

    print(f"Moved {len(moved)} file(s) into {ARCHIVE_DIR}:")
    for kind, name, dest in moved:
        print(f"  {kind:8s}  {name:40s}  ->  {dest.relative_to(PROJECT_DIR)}")

    print()
    print("Active study folders are now clean. You can safely launch T0 collection.")


if __name__ == "__main__":
    main()
