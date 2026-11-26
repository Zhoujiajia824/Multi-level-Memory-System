"""List/check designs, or explicitly execute one independent CARLA demonstration.

Use the existing CARLA SDK environment here and the existing numeric environment
for --worker-python. This never starts CARLA or modifies an official evaluator.
"""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('list','check','capture'))
    parser.add_argument('--catalog')
    parser.add_argument('--scenario',help='One catalog id or family, never a formal route id')
    parser.add_argument('--map',choices=('Town01','Town02','Town03','Town04','Town05','Town10HD_Opt','Town13'),
                        help='Map for an independently selected road')
    parser.add_argument('--collection-route-id', type=int,
                        help='Recreate the interaction near this collection route on its original map')
    parser.add_argument('--fail2drive-root', help='Official devkit containing fail2drive_split/')
    parser.add_argument('--seed',type=int,help='Explicit alternative development site seed')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=39000)
    parser.add_argument('--output-dir',help='New episode directory under branch output/')
    parser.add_argument('--worker-python',help='Existing numerical Python; required for actual capture/export')
    parser.add_argument('--allow-map-load',action='store_true',help='Allow required map load only after idle-world check')
    parser.add_argument('--allow-similar-road',action='store_true',
                        help='Allow a same-map site with the required topology if the collection anchor is unsuitable')
    parser.add_argument('--no-video',action='store_true',help='Video is on by default')
    parser.add_argument('--interaction-only',action='store_true',
                        help='End after the observed interaction, return to navigation and real three-second recovery images')
    args = parser.parse_args(argv)
    from ep_driver_fail2drive.demo_catalog import load_catalog,get_demonstration,catalog_digest,validate_catalog
    from nine_grid.paths import filesystem_path
    try:
        catalog = load_catalog(args.catalog)
        if args.command == 'list':
            print(json.dumps([dict(id=s['id'],family=s['family'],map=s['map'],
                kind=s['interaction_kind'],maximum_seconds=s['max_duration_s'],
                proxy=s['proxy']) for s in catalog['demonstrations']],ensure_ascii=False,indent=2))
            return 0
        if not args.scenario:
            parser.error('--scenario is required for check/capture')
        spec = get_demonstration(catalog,args.scenario)
        if args.map is not None or args.seed is not None:
            for item in catalog['demonstrations']:
                if item['id'] == spec['id']:
                    if args.map is not None:
                        item['map'] = args.map
                    if args.seed is not None:
                        item['seed'] = args.seed
            catalog = validate_catalog(catalog)
            spec = get_demonstration(catalog,args.scenario)
        if (args.collection_route_id is None) != (args.fail2drive_root is None):
            parser.error('--collection-route-id and --fail2drive-root must be supplied together')
        if args.allow_similar_road and args.collection_route_id is None:
            parser.error('--allow-similar-road requires a collection route anchor')
        if args.collection_route_id is not None:
            if args.map is not None:
                parser.error('The collection route supplies the map; omit --map')
            from ep_driver_fail2drive.demo_catalog import with_collection_route_anchor
            spec = with_collection_route_anchor(spec, args.fail2drive_root, args.collection_route_id)
        if args.command == 'check':
            print(json.dumps(dict(valid_design=True,spec=spec,catalog_sha256=catalog_digest(catalog),
                capture_mode='interaction_segment' if args.interaction_only else 'full_route',
                server_connected=False,executed=False,successful_memory_created=False),ensure_ascii=False,indent=2))
            return 0
        if not args.worker_python:
            parser.error('--worker-python must select the existing numerical environment')
        output = filesystem_path(args.output_dir or ROOT/'output'/'demonstrations'/spec['id']/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        if not output.is_relative_to(filesystem_path(ROOT/'output')):
            parser.error('demonstration artifacts must be under this branch output/ directory')
        if output.exists():
            parser.error('output directory must be new; previous data is never overwritten')
        # Check the selected interpreter without creating an environment or
        # touching the simulator. Fail early before collecting unusable data.
        check = subprocess.run([args.worker_python,'-c',
            'import sys;sys.path.insert(0,sys.argv[1]);from ep_driver_fail2drive.demo_export import export_capture',str(ROOT/'src')],
            cwd=str(ROOT),check=False)
        if check.returncode:
            raise ValueError('Selected numerical environment cannot import the capture converter')
        from ep_driver_fail2drive.demo_capture import capture
        result = capture(spec,output,host=args.host,port=args.port,allow_map_load=args.allow_map_load,
                         video=not args.no_video,interaction_only=args.interaction_only,
                         allow_similar_road=args.allow_similar_road)
        command = [args.worker_python,'-c',
            'import sys;sys.path.insert(0,sys.argv[1]);from ep_driver_fail2drive.demo_export import export_capture;export_capture(sys.argv[2])',
            str(ROOT/'src'),str(output)]
        converted = subprocess.run(command,cwd=str(ROOT),check=False)
        if converted.returncode:
            raise ValueError('Raw collection retained; numerical conversion failed: '+str(output))
        print(json.dumps(dict(output=str(output),evaluation=result,memory_published=False),ensure_ascii=False,indent=2))
        return 0 if result['passed'] else 2
    except (ValueError,RuntimeError,OSError,KeyError,ImportError) as exc:
        print(json.dumps(dict(error=type(exc).__name__+': '+str(exc),memory_published=False),ensure_ascii=False),file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
