"""Finalize official expert captures, check evidence, or prepare offline tasks."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    finish=commands.add_parser('finalize')
    finish.add_argument('--capture','--run-dir',dest='capture',required=True)
    finish.add_argument('--checkpoint',required=True)
    finish.add_argument('--launch',required=True)
    for name in ('check','prepare'):
        command=commands.add_parser(name)
        command.add_argument('--captures',required=True,action='append')
        command.add_argument('--anchor-frame',action='append')
        command.add_argument('--require-all-families',action='store_true')
        command.add_argument('--collection-split',help='Fixed 34-route memory partition; reject held-out test routes and partners')
        if name=='prepare':
            command.add_argument('--workbench-root',required=True)
            command.add_argument('--output-dir',required=True)
            command.add_argument('--pipeline-config')
            command.add_argument('--dino-model')
            command.add_argument('--device',default='cpu')
    args=parser.parse_args(argv)
    from ep_driver_fail2drive.official_demo_memory import (finalize_official_capture,
        check_official_demonstrations,prepare_official_demonstrations)
    try:
        if args.command=='finalize':
            result=finalize_official_capture(args.capture,args.checkpoint,args.launch)
        elif args.command=='check':
            result=check_official_demonstrations(args.captures,anchor_frames=args.anchor_frame,
                require_all_families=args.require_all_families,collection_split=args.collection_split)
        else:
            result=prepare_official_demonstrations(args.captures,args.output_dir,args.workbench_root,
                pipeline_config=args.pipeline_config,dino_model=args.dino_model,device=args.device,
                anchor_frames=args.anchor_frame,require_all_families=args.require_all_families,
                collection_split=args.collection_split)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result['valid'] else 2
    except (ValueError,KeyError,TypeError,OSError,ImportError) as exc:
        print(json.dumps(dict(valid=False,error=str(exc),memory_published=False),ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
