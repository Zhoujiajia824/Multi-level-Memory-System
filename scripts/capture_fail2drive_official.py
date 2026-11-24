"""Collect expert demonstrations in unmodified original Fail2Drive scenarios."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from ep_driver_fail2drive.official_demo_launch import main


if __name__=='__main__':
    raise SystemExit(main())
