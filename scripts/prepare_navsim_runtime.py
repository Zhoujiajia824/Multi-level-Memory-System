"""Generate isolated A/B/C runtime configurations; no installs or model calls."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ep_driver_navsim.runtime_config import main

if __name__ == "__main__":
    main()
