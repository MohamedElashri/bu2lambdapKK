"""Root conftest.py — ensures analysis/ is on sys.path for test imports."""

import sys
from pathlib import Path

_analysis_dir = str(Path(__file__).resolve().parent)
if _analysis_dir not in sys.path:
    sys.path.insert(0, _analysis_dir)
