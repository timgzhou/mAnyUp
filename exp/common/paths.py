"""Where the repo, its data and the feature caches live.

Data paths elsewhere stay relative ("data/..."): every entry point runs from the repo root
(`python -m exp...`). Feature caches are the exception -- they run to terabytes, so they
live in project space rather than in the repo. Override with $MANYUP_FEATURES.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
FEATURES = Path(os.environ.get("MANYUP_FEATURES", "~/projects/aip-gpleiss/timz/features")).expanduser()
ANYUP_REPO = REPO / "third_party" / "anyup"
