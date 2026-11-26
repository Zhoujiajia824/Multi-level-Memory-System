"""Preview or manually start our isolated service using read-only shared weights."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ep_driver_navsim.shared_model_launch import main

if __name__ == "__main__":
    raise SystemExit(main())
