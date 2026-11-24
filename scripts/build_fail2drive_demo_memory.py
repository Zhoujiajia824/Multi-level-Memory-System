"""Check real targeted demonstrations or prepare them for the offline workbench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'prepare'):
        command = commands.add_parser(name)
        command.add_argument('--captures', action='append', required=True,
                             help='One episode or a dedicated capture root; repeat to select explicit episodes')
        command.add_argument('--anchor-frame', action='append', help='One to three actual frame IDs; single episode only')
        command.add_argument('--require-all-families', action='store_true')
        if name == 'prepare':
            command.add_argument('--workbench-root', required=True, help='Existing offline_memory_builder project root')
            command.add_argument('--pipeline-config')
            command.add_argument('--output-dir', required=True, help='New build directory, separate from source/workbench')
            command.add_argument('--dino-model', help='Explicit real DINO weights; omit to extract later')
            command.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    from ep_driver_fail2drive.demo_memory import check_demonstrations, prepare_demonstrations
    try:
        if args.command == 'check':
            report = check_demonstrations(args.captures, anchor_frames=args.anchor_frame,
                                         require_all_families=args.require_all_families)
        else:
            report = prepare_demonstrations(args.captures, args.output_dir, args.workbench_root,
                pipeline_config=args.pipeline_config, dino_model=args.dino_model, device=args.device,
                anchor_frames=args.anchor_frame, require_all_families=args.require_all_families)
            # These are instructions, not executed calls.  Review requires the
            # actual human reviewer identity and each specific task ID.
            report['next_steps'] = [
                'Use the explicit workbench with this build; extract real DINO if not supplied.',
                'annotate --build <build> --dry-run; inspect the actual teacher request.',
                'annotate --build <build> --call-model; retain the real raw response.',
                'validate --build <build>; inspect actual driving evidence and text.',
                'review --build <build> --task <ID> --decision approve --reviewer <human>',
                'publish --build <build> --library-dir <new-dedicated-frozen-library>',
                'verify-readers against this Fail2Drive branch before reflection retrieval.',
            ]
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['valid'] else 2
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(json.dumps({'valid': False, 'error': str(error), 'memory_published': False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
