"""Collect the ten selected Fail2Drive interaction pairs, or resume the same batch."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ep_driver_fail2drive.demo_catalog import capture_implementation, get_demonstration, load_catalog, with_collection_route_anchor
from ep_driver_fail2drive.memory_split import load_transfer_split, validate_transfer_source
from nine_grid.paths import filesystem_path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def write_text(path, content):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(content, encoding='utf-8')
    tmp.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def build_plan(source, similar):
    split = load_transfer_split()
    verified = validate_transfer_source(split, source)
    catalog = load_catalog()
    routes = {}
    for row in split['rows']:
        for record in row['collection']:
            rid = record['route_id']
            spec = get_demonstration(catalog, row['family'])
            spec['seed'] = split['collection_seed']
            spec = with_collection_route_anchor(spec, source, rid)
            routes[f'{rid:04d}'] = dict(family=row['family'], variant=record['variant'],
                                        description=spec['interaction_reconstruction']['description'],
                                        spec_sha256=digest(spec))
    if len(routes) != 20 or len(split['rows']) != 10:
        raise ValueError('Expected the frozen ten-family, twenty-demonstration collection')
    return dict(schema='fail2drive-interaction-batch-1', split_sha256=verified['split_sha256'],
                implementation_sha256=capture_implementation(), allow_similar_road=similar,
                seed=split['collection_seed'], routes=routes)


@contextmanager
def batch_lock(root):
    # OS releases the lock even on process death; the empty lock file may remain.
    with (root / '.collection.lock').open('a+b') as lock:
        try:
            if os.name == 'nt':
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('Another collector is using this batch directory') from exc
        yield


def check_capture(episode, record, worker, audit_path):
    if not (episode / 'capture_manifest.json').is_file():
        return False
    try:
        run = read_json(episode / 'run.json')
        if (digest(run.get('parameters')) != record['spec_sha256']
                or run.get('implementation_sha256') != capture_implementation()
                or run.get('capture_mode') != 'interaction_segment' or
                (episode / 'planning' / 'video.mp4').stat().st_size == 0):
            return False
    except (OSError, ValueError):
        return False
    with audit_path.open('w', encoding='utf-8') as audit:
        result = subprocess.run([worker, str(ROOT / 'scripts/build_fail2drive_demo_memory.py'),
                                 'check', '--captures', str(episode)], stdout=audit, cwd=ROOT)
    return result.returncode == 0


def collect_one(tag, record, plan, root, args):
    previous = sorted((root / 'clips').glob(tag + '_try_*'), reverse=True)
    def check(episode):
        return check_capture(episode, record, args.worker_python, root / 'logs' / (episode.name + '.check.json'))
    for path in previous:
        if check(path):
            return dict(status='ready_for_video_review', path='clips/' + path.name)
    attempted = previous + list((root / 'logs').glob(tag + '_try_*.log'))
    index = 1 + max((int(p.stem.rsplit('_', 1)[1]) for p in attempted), default=0)
    episode = root / 'clips' / f'{tag}_try_{index:03d}'
    command = [sys.executable, str(ROOT / 'scripts/capture_fail2drive_demo.py'), 'capture',
               '--scenario', record['family'], '--collection-route-id', str(int(tag)),
               '--seed', str(plan['seed']), '--fail2drive-root', str(args.fail2drive_root),
               '--host', args.host, '--port', str(args.port), '--worker-python', args.worker_python,
               '--interaction-only', '--output-dir', str(episode)]
    if args.allow_map_load:
        command.append('--allow-map-load')
    if plan['allow_similar_road']:
        command.append('--allow-similar-road')
    log = root / 'logs' / (episode.name + '.log')
    with log.open('w', encoding='utf-8') as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    if result.returncode in (-signal.SIGINT, -signal.SIGTERM):
        raise RuntimeError('Capture process terminated by signal: ' + str(-result.returncode))
    passed = result.returncode == 0 and check(episode)
    return dict(status='ready_for_video_review' if passed else 'failed',
                path='clips/' + episode.name, capture_exit=result.returncode, log='logs/' + log.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fail2drive-root', required=True)
    parser.add_argument('--output-dir', required=True, help='Dedicated resumable batch under this branch output/')
    parser.add_argument('--worker-python')
    parser.add_argument('--route-ids', nargs='+', type=int, help='One or more frozen collection IDs, default all twenty')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=39000)
    parser.add_argument('--allow-map-load', action='store_true')
    parser.add_argument('--allow-similar-road', action='store_true')
    parser.add_argument('--run', action='store_true', help='Execute; otherwise only verify source and print plan')
    args = parser.parse_args(argv)
    root = filesystem_path(args.output_dir)
    base = filesystem_path(ROOT / 'output')
    if root == base or not root.is_relative_to(base):
        parser.error('Choose a dedicated subdirectory under branch output/')
    plan = build_plan(args.fail2drive_root, args.allow_similar_road)
    routes = plan['routes']
    selected = list(routes) if args.route_ids is None else [f'{n:04d}' for n in args.route_ids]
    if len(set(selected)) != len(selected) or set(selected) - set(routes):
        parser.error('Use distinct collection route IDs; held-out evaluation IDs are ineligible')
    if not args.run:
        for tag in selected:
            item = routes[tag]
            print(tag, item['family'], item['variant'], item['description'])
        print(f'{len(selected)} selected / {len(routes)} planned')
        return 0
    if not args.worker_python:
        parser.error('--run requires --worker-python (existing numerical Python)')
    if not shutil.which('ffmpeg'):
        raise ValueError('ffmpeg is required on PATH for the review videos')
    import carla  # Verify SDK before starting any attempt.
    subprocess.run([args.worker_python, '-c', 'import sys;sys.path.insert(0,sys.argv[1]);'
                    'from ep_driver_fail2drive.demo_export import export_capture;'
                    'from ep_driver_fail2drive.demo_memory import check_demonstrations', str(ROOT / 'src')], check=True)
    root.mkdir(parents=True, exist_ok=True)
    with batch_lock(root):
        binding = root / 'batch_plan.json'
        if binding.exists():
            if read_json(binding) != plan:
                raise ValueError('Collection designs or site policy changed; select a new batch directory')
        else:
            if any(p.name != '.collection.lock' for p in root.iterdir()):
                raise ValueError('This nonempty directory belongs to another run; select a new batch directory')
            write_json(binding, plan)
        (root / 'clips').mkdir(exist_ok=True)
        (root / 'logs').mkdir(exist_ok=True)
        summary = root / 'collection.json'
        results = read_json(summary) if summary.exists() else {}
        for tag in selected:
            print(f"[{tag}] {routes[tag]['family']} {routes[tag]['variant']}", flush=True)
            item = collect_one(tag, routes[tag], plan, root, args)
            results[tag] = item
            write_json(summary, results)
            ready = [results[t]['path'] for t in routes if results.get(t, {}).get('status') == 'ready_for_video_review']
            write_text(root / 'review_paths.txt', ''.join(p + '\n' for p in ready))
            print(f"[{tag}] {item['status']} ({len(ready)}/{len(routes)})  {item.get('log', item['path'])}", flush=True)
        return 0 if all(results[t]['status'] == 'ready_for_video_review' for t in selected) else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
