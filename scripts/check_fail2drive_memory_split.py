"""Read-only check of the fixed 34-demo/17-test route partition; no CARLA."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ep_driver_fail2drive.memory_split import load_transfer_split,route_ids,validate_transfer_source


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-config')
    parser.add_argument('--fail2drive-root',required=True)
    parser.add_argument('--ids',choices=('collection','evaluation'),help='Print only validated numeric IDs for existing launchers')
    args=parser.parse_args(argv)
    try:
        split=load_transfer_split(args.split_config)
        report=validate_transfer_source(split,args.fail2drive_root)
        if args.ids:print(' '.join(map(str,route_ids(split,args.ids))))
        else:print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,KeyError,TypeError,OSError) as error:
        print(json.dumps(dict(valid=False,error=str(error)),ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
