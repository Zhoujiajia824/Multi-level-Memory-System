"""Run from the project root, using the normal EP-Driver Python 3.12 environment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ep_driver_navsim.launch import main

if __name__ == "__main__":
    raise SystemExit(main())
