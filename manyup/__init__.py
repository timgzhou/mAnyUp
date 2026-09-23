"""mAnyUp / timAnyUp upsamplers, built on the vendored upstream AnyUp in third_party/anyup."""
import sys
from pathlib import Path

_ANYUP_REPO = str(Path(__file__).resolve().parents[1] / "third_party" / "anyup")
if _ANYUP_REPO not in sys.path:
    sys.path.insert(0, _ANYUP_REPO)
