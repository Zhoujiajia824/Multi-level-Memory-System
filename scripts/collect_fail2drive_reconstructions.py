"""Collect authored short driving interactions, one Base/Generalization family pair at a time."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ep_driver_fail2drive.demo_catalog import FAMILIES, capture_implementation, get_demonstration, load_catalog, with_collection_route_anchor
from ep_driver_fail2drive.memory_split import ALL_FAMILIES_SPLIT, load_transfer_split, validate_transfer_source
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


def build_plan(source, similar, split_path=None):
    split = load_transfer_split(split_path or ALL_FAMILIES_SPLIT)
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
    if len(routes) != 2*len(split['rows']):
        raise ValueError('Each family needs its two frozen collection references')
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
    try:
        if not (episode / 'capture_manifest.json').is_file():
            raise ValueError('缺少 capture_manifest.json')
        run = read_json(episode / 'run.json')
        if digest(run.get('parameters')) != record['spec_sha256']:
            raise ValueError('采集参数与此变体不一致')
        if run.get('implementation_sha256') != capture_implementation():
            raise ValueError('采集代码指纹与当前版本不一致')
        if run.get('capture_mode') != 'interaction_segment':
            raise ValueError('采集模式与交互片段不一致')
        if not (episode / 'planning' / 'video.mp4').stat().st_size:
            raise ValueError('planning/video.mp4 为空')
    except (OSError, ValueError) as error:
        write_json(audit_path, dict(valid=False, error=str(error)))
        return False
    with audit_path.open('w', encoding='utf-8') as audit:
        result = subprocess.run([worker, str(ROOT / 'scripts/build_fail2drive_demo_memory.py'),
                                 'check', '--captures', str(episode)], stdout=audit, stderr=subprocess.PIPE,
                                encoding='utf-8', errors='replace', cwd=ROOT)
    if result.returncode and result.stderr:
        write_text(audit_path.with_suffix('.stderr.log'), result.stderr)
    if result.returncode and not audit_path.stat().st_size:
        reason = result.stderr.strip() if result.stderr else f'验收进程退出码 {result.returncode}'
        try:
            reason = json.loads(reason)['error']
        except (ValueError, KeyError, TypeError):
            reason = reason.splitlines()[-1] if reason else f'验收进程退出码 {result.returncode}'
        write_json(audit_path, dict(valid=False, error=reason, check_exit=result.returncode))
    return result.returncode == 0


def failure_reason(episode, evidence, stage):
    paths = [evidence] if stage == 'check' else [episode / 'capture_failure.json', episode / 'demo_evaluation.json']
    for path in paths:
        try:
            report = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict):
            continue
        reasons = [report.get('error'), *(report.get('errors') or []),
                   *(entry.get('reason') for entry in report.get('rejected', []) if isinstance(entry, dict))]
        if report.get('passed') is False:
            reasons.extend(key + '=false' for key in ('interaction_complete', 'recovery_complete',
                           'projection_valid', 'world_settings_restored') if report.get(key) is False)
            reasons.extend(f'{key}={report[key]}' for key in ('collision_count', 'offroad_count',
                           'rule_violation_count') if report.get(key))
        message = ' / '.join(str(value) for value in reasons if value)
        if message:
            return ' '.join(message.split())[:500]
    # A startup/import error can precede all episode artifacts.
    try:
        with evidence.open('rb') as handle:
            handle.seek(max(0, evidence.stat().st_size - 8192))
            lines = handle.read().decode('utf-8', errors='replace').strip().splitlines()
        if lines:
            line = lines[-1]
            try:
                line = json.loads(line)['error']
            except (KeyError, TypeError, ValueError):
                pass
            return str(line)[:500]
    except OSError:
        pass
    return '子进程未提供错误文本'


def last_startup_step(episode):
    try:
        if (episode / 'execution.jsonl').is_file():
            return None
        stage = read_json(episode / 'startup_stage.json')
    except (OSError, ValueError):
        return None
    if not isinstance(stage, dict) or stage.get('status') != 'entered':
        return None
    operation = stage.get('operation')
    if not isinstance(operation, str) or not operation:
        return None
    step = dict(operation=operation)
    if isinstance(stage.get('timeout_s'), (int, float)) and stage['timeout_s'] > 0:
        step['timeout_s'] = stage['timeout_s']
    return step


def collect_one(tag, record, plan, root, args):
    prefix = tag + '_try_'
    previous = sorted((p for p in (root / 'clips').glob(prefix + '*')
                       if p.is_dir() and p.name[len(prefix):].isdecimal()), reverse=True)
    def check(episode):
        return check_capture(episode, record, args.worker_python, root / 'logs' / (episode.name + '.check.json'))
    for path in previous:
        if check(path):
            return dict(status='ready_for_video_review', path='clips/' + path.name)
    attempted = previous + list((root / 'logs').glob(prefix + '*'))
    # Audit artifacts share the attempt prefix (e.g. 0070_try_002.check.stderr.log).
    numbers = [p.name[len(prefix):].partition('.')[0] for p in attempted]
    index = 1 + max((int(n) for n in numbers if n.isdecimal()), default=0)
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
    item = dict(path='clips/' + episode.name, capture_exit=result.returncode, log='logs/' + log.name)
    if result.returncode == 0 and check(episode):
        return dict(item, status='ready_for_video_review')
    stage = 'capture' if result.returncode else 'check'
    evidence = log if result.returncode else log.with_suffix('.check.json')
    reason = failure_reason(episode, evidence, stage)
    item.update(status='failed', failure_stage=stage, reason=reason, failure_log='logs/' + evidence.name)
    if result.returncode < 0 or (os.name == 'nt' and result.returncode >= 0xC0000000):
        item['batch_stop_reason'] = 'capture_process_terminated'
        step = last_startup_step(episode)
        if step:
            item['last_startup_step'] = step
            detail = step['operation']
            if 'timeout_s' in step:
                detail += f"（RPC 时限 {step['timeout_s']:g} 秒）"
            item['reason'] += '。最后记录的初始化步骤为 ' + detail
    elif stage == 'capture':
        try:
            failure = read_json(episode / 'capture_failure.json')
        except (OSError, ValueError):
            failure = {}
        if ((isinstance(failure, dict) and failure.get('failure_code') == 'world_not_idle')
                or 'Demonstration collection requires an idle dedicated world' in reason):
            item['batch_stop_reason'] = 'world_not_idle'
    return item


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fail2drive-root', required=True)
    parser.add_argument('--output-dir', required=True, help='Dedicated resumable batch under this branch output/')
    parser.add_argument('--worker-python')
    parser.add_argument('--collection-split', default=str(ALL_FAMILIES_SPLIT),
                        help='Collection reference partition, default 17 families with two authored clips each')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--route-ids', nargs='+', type=int, help='One or more frozen collection reference IDs')
    selection.add_argument('--family', choices=FAMILIES, help='Run only this family, Base design then Generalization design')
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
    plan = build_plan(args.fail2drive_root, args.allow_similar_road, args.collection_split)
    routes = plan['routes']
    if args.family:
        selected = sorted((tag for tag, record in routes.items() if record['family'] == args.family),
                          key=lambda tag: routes[tag]['variant'] != 'Base')
        if len(selected) != 2:
            parser.error('Selected collection partition does not contain this complete family pair')
    else:
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
            if item['status'] == 'failed':
                detail = f"{item['failure_stage']}: {item['reason']}  ({item['failure_log']})"
            else:
                detail = item['path']
            batch_ready = sum(results.get(t, {}).get('status') == 'ready_for_video_review' for t in selected)
            print(f"[{tag}] {item['status']} (batch {batch_ready}/{len(selected)}, total {len(ready)}/{len(routes)})  {detail}", flush=True)
            if item.get('batch_stop_reason'):
                print(f"本批次已停止（{item['batch_stop_reason']}，采集退出码 {item['capture_exit']}）。"
                      f"请确认 {args.host}:{args.port} 专用世界空闲后再续跑。", flush=True)
                return 2
        return 0 if all(results[t]['status'] == 'ready_for_video_review' for t in selected) else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
