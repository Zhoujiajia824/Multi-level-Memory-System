"""Collect key interactions using the original Fail2Drive routes and scenario runner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ep_driver_fail2drive import official_demo_launch


def main(argv=None):
    return official_demo_launch.main(argv, collection_entry=True)


if __name__ == '__main__':
    raise SystemExit(main())
