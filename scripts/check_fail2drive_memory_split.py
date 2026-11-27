"""Validate an official collection/evaluation partition and optionally save the chosen pairs."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ep_driver_fail2drive.memory_split import (
    ALL_FAMILIES_SPLIT,RESERVE_PAIRS,load_transfer_split,route_ids,validate_transfer_source,with_reserve_pairs)


def _save(path,split):
    path.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        tmp=Path(stream.name)
    try:
        tmp.write_text(json.dumps(split,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        try:
            # Windows rename refuses an existing target. POSIX rename would
            # replace it, so publish the complete same-directory inode by link.
            if os.name=='nt':
                tmp.rename(path)
            else:
                os.link(tmp,path)
        except FileExistsError:
            if load_transfer_split(path)!=split:
                raise ValueError('The output split already contains a different selection: '+str(path))
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-config',default=ALL_FAMILIES_SPLIT,
                        help='Partition JSON; defaults to the 17-family official collection')
    parser.add_argument('--fail2drive-root',required=True)
    parser.add_argument('--reserve-pair',action='append',choices=sorted(RESERVE_PAIRS),default=[],
                        help='Use both official reserve partners for this family in the final selection. Repeat by family.')
    parser.add_argument('--write-split',type=Path,help='Save the validated selection for check/prepare; existing selections are preserved.')
    parser.add_argument('--ids',choices=('collection','evaluation'),help='Print only validated numeric IDs for existing launchers')
    args=parser.parse_args(argv)
    try:
        split=load_transfer_split(args.split_config)
        if args.reserve_pair:
            split=with_reserve_pairs(split,args.reserve_pair)
        report=validate_transfer_source(split,args.fail2drive_root)
        if args.write_split:
            _save(args.write_split,split)
            report['split_path']=str(args.write_split.resolve())
        if args.ids:print(' '.join(map(str,route_ids(split,args.ids))))
        else:print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,KeyError,TypeError,OSError) as error:
        print(json.dumps(dict(valid=False,error=str(error)),ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
