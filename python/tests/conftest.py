"""Pytest configuration: put the package on the path without an install step.

Keeping the repo runnable straight from a clone — no ``pip install -e .``
required — means the reproducibility instructions in the README are two
commands rather than five.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
