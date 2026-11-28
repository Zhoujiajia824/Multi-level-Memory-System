"""Collect all authored interactions, or select one Base/Generalization pair."""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ep_driver_fail2drive.demo_catalog import FAMILIES, capture_implementation, get_demonstration, load_catalog, with_collection_route_anchor
from ep_driver_fail2drive.memory_split import ALL_FAMILIES_SPLIT, load_transfer_split, validate_transfer_source
from nine_grid.paths import filesystem_path

TRANSPORT_FAILURES = frozenset(('camera_transport_error', 'camera_timeout', 'simulator_rpc_failure',
    'simulator_unavailable', 'simulator_observation_pending', 'capture_process_terminated', 'owned_simulator_exited',
    'simulator_startup_timeout', 'managed_simulator_start_failed'))


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


def _read_simulator_health(host, port, timeout=5.):
    """Read RPC readiness without spawning actors, changing settings or ticking."""
    import carla
    result = dict(schema='fail2drive-collection-simulator-health-1', ready=False, available=False,
                  host=host, port=port, timeout_s=timeout, world_mutated=False, operations=[])
    deadline = time.monotonic()+timeout
    try:
        client = carla.Client(host, port)
        def read(name, operation):
            result['phase'] = name
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Simulator readiness deadline exceeded')
            client.set_timeout(remaining)
            started = time.monotonic()
            try:
                return operation()
            finally:
                result['operations'].append(dict(name=name, timeout_s=remaining, elapsed_s=time.monotonic()-started))
        version = read('get_server_version', client.get_server_version)
        result.update(available=True, server_version=version, client_version=client.get_client_version())
        if result['client_version'] != version:
            result['failure_code'] = 'simulator_version_mismatch'
            result['error'] = 'CARLA client/server versions differ'
            return result
        world = read('get_world', client.get_world)
        snapshot = read('get_snapshot', world.get_snapshot)
        result.update(ready=True, phase='ready', frame=snapshot.frame, time=snapshot.timestamp.elapsed_seconds)
    except Exception as error:
        pending = result['available'] and result.get('phase') in ('get_world', 'get_snapshot')
        result.update(failure_code='simulator_observation_pending' if pending else 'simulator_unavailable',
                      error=type(error).__name__+': '+str(error))
    return result


def _isolated_simulator_health(host, port, timeout):
    """Keep native SDK faults and hung read calls outside the batch process."""
    result = dict(schema='fail2drive-collection-simulator-health-1', ready=False,
                  host=host, port=port, timeout_s=timeout, world_mutated=False,
                  failure_code='simulator_unavailable')
    try:
        probe = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe-simulator',
                                host, str(port), str(timeout)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, encoding='utf-8', errors='replace', timeout=timeout+5.)
        if probe.returncode in (0, 2):
            report = json.loads(probe.stdout.strip().splitlines()[-1])
            if isinstance(report, dict) and type(report.get('ready')) is bool:
                if report['ready'] and probe.returncode != 0:
                    raise ValueError('Simulator readiness status conflicts with probe exit code')
                if not report['ready'] and not report.get('failure_code'):
                    raise ValueError('Failed simulator readiness probe has no failure classification')
                return report
        result.update(probe_exit=probe.returncode,
                      error='Read-only CARLA probe exited with code '+str(probe.returncode),
                      probe_stderr=probe.stderr[-2000:])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
        result['error'] = type(error).__name__+': '+str(error)
    return result


def simulator_health(host, port, timeout=5.):
    report = _isolated_simulator_health(host, port, timeout)
    if report.get('failure_code') != 'simulator_observation_pending':
        return report
    # A live version RPC with a delayed episode subscription is not proof of an
    # engine crash. Allow exactly one fresh, bounded subscription before recovery.
    retry = _isolated_simulator_health(host, port, min(60., max(15., timeout)))
    result = dict(retry)
    result['probes'] = [report, retry]
    return result


def clean_teardown(episode):
    """A retry must not inherit unverified settings or actors from its predecessor."""
    try:
        evaluation = read_json(episode / 'demo_evaluation.json')
        cleanup = evaluation['actor_cleanup']
        if (evaluation['world_settings_restored'] is not True
                or cleanup['post_restore_snapshot']['remaining_owned_ids'] != []):
            return False
        if any(str(error).startswith(('sensor stop:', 'restore event signals:', 'restore weather:',
                                     'restore settings:', 'Owned actors still present'))
               for error in evaluation.get('errors', [])):
            return False
        attempts = cleanup['attempts']
        if not attempts or any(row.get('errors') != [] or row.get('skipped') for row in attempts):
            return False
        owned = {actor for row in attempts for actor in row['owned_ids']}
        confirmed = {actor for row in attempts for actor in
                     row['acknowledged_ids']+row['previously_destroyed_ids']}
        return owned <= confirmed and all(set(row['requested_ids']) <= set(row['acknowledged_ids'])
                                         for row in attempts)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def physical_violations(episode):
    try:
        report = read_json(episode / 'demo_evaluation.json')
        return {key: report[key] for key in ('collision_count', 'offroad_count', 'rule_violation_count')
                if isinstance(report.get(key), (int, float)) and report[key] > 0}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def classify_capture_failure(episode, reason, args):
    try:
        failure = read_json(episode / 'capture_failure.json')
    except (OSError, ValueError):
        failure = {}
    code = failure.get('failure_code') if isinstance(failure, dict) else None
    if code == 'world_not_idle' or 'Demonstration collection requires an idle dedicated world' in reason:
        return dict(failure_code='world_not_idle', batch_stop_reason='world_not_idle')
    # Old archives may predate structured camera/RPC diagnostics.
    if code is None:
        if 'Missing exact camera frame:' in reason or 'Camera queue overflow:' in reason:
            code = 'camera_transport_error'
        elif ('while waiting for the simulator' in reason or 'rpc::timeout' in reason
              or 'simulator_unavailable' in reason):
            code = 'simulator_rpc_failure'
    health = simulator_health(args.host, args.port, getattr(args, 'simulator_timeout', 5.))
    if not health['ready']:
        return dict(failure_code=health.get('failure_code', 'simulator_unavailable'),
                    source_failure_code=code or 'capture_failed',
                    batch_stop_reason=health.get('failure_code', 'simulator_unavailable'), simulator_health=health)
    clean = clean_teardown(episode)
    result = dict(failure_code=code or 'capture_failed', source_failure_code=code or 'capture_failed', simulator_health=health,
                  teardown_verified=clean)
    if not clean:
        result['batch_stop_reason'] = 'cleanup_unverified'
    elif code in TRANSPORT_FAILURES and not physical_violations(episode):
        result['retryable_transport_failure'] = True
    return result


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
    owner = getattr(args, '_simulator_owner', None)
    if owner is not None:
        try:
            owner.prepare()
        except Exception as error:
            failure = dict(ready=False, error=type(error).__name__+': '+str(error),
                           failure_code=getattr(error, 'code', 'managed_simulator_start_failed'))
            evidence = root / 'logs' / (episode.name + '.readiness.json')
            write_json(evidence, failure)
            return dict(status='blocked', failure_stage='simulator_startup', failure_code=failure['failure_code'],
                reason=failure['error'], failure_log='logs/' + evidence.name,
                batch_stop_reason=failure['failure_code'])
    health = simulator_health(args.host, args.port, getattr(args, 'simulator_timeout', 5.))
    if not health['ready']:
        evidence = root / 'logs' / (episode.name + '.readiness.json')
        write_json(evidence, health)
        return dict(status='blocked', failure_stage='simulator_readiness',
                    failure_code=health.get('failure_code', 'simulator_unavailable'),
                    reason=health.get('error', 'Simulator RPC is not ready'),
                    failure_log='logs/' + evidence.name,
                    batch_stop_reason=health.get('failure_code', 'simulator_unavailable'),
                    simulator_health=health)
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
    item.update(status='failed', failure_stage=stage, reason=reason, failure_log='logs/' + evidence.name,
                known_physical_violations=physical_violations(episode))
    if result.returncode < 0 or (os.name == 'nt' and result.returncode >= 0xC0000000):
        item['batch_stop_reason'] = 'capture_process_terminated'
        item['source_failure_code'] = 'capture_process_terminated'
        step = last_startup_step(episode)
        if step:
            item['last_startup_step'] = step
            detail = step['operation']
            if 'timeout_s' in step:
                detail += f"（RPC 时限 {step['timeout_s']:g} 秒）"
            item['reason'] += '。最后记录的初始化步骤为 ' + detail
    elif stage == 'capture':
        item.update(classify_capture_failure(episode, reason, args))
    return item


def save_progress(root, selected, routes, results, status, current=None):
    write_json(root / 'collection.json', results)
    ready = [tag for tag in routes if results.get(tag, {}).get('status') == 'ready_for_video_review']
    write_text(root / 'review_paths.txt', ''.join(results[tag]['path'] + '\n' for tag in ready))
    write_json(root / 'batch_state.json', dict(schema='fail2drive-collection-progress-1', status=status,
        selected_route_ids=selected, ready_route_ids=ready,
        failed_route_ids=[tag for tag in selected if results.get(tag, {}).get('status') == 'failed'
                          and not results[tag].get('batch_stop_reason')],
        not_started_route_ids=[tag for tag in selected if tag not in results or results[tag]['status'] == 'blocked'],
        pending_route_ids=[tag for tag in selected if tag not in ready],
        current_route_id=current,
        batch_stop_reason=results.get(current, {}).get('batch_stop_reason') if current else None))
    return ready


def recover_owned_simulator(item, args):
    owner = getattr(args, '_simulator_owner', None)
    recoverable = item.get('retryable_transport_failure') or item.get('batch_stop_reason') in (
        'simulator_unavailable', 'simulator_observation_pending', 'cleanup_unverified', 'capture_process_terminated',
        'owned_simulator_exited', 'simulator_startup_timeout', 'managed_simulator_start_failed')
    if owner is None or not recoverable:
        return False
    try:
        owner.restart(item.get('reason', item.get('batch_stop_reason', 'capture transport failure')))
    except Exception as error:
        item['batch_stop_reason'] = getattr(error, 'code', 'managed_simulator_recovery_failed')
        item['simulator_recovery'] = dict(restarted=False, error=type(error).__name__+': '+str(error))
    else:
        item.pop('batch_stop_reason', None)
        item['simulator_recovery'] = dict(restarted=True, restarts=owner.restarts)
        source_code = item.get('source_failure_code', item.get('failure_code', 'capture_process_terminated'
                              if item.get('capture_exit', 0) < 0 else None))
        item['retryable_transport_failure'] = source_code in TRANSPORT_FAILURES and not item.get('known_physical_violations')
    return True


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
    parser.add_argument('--transport-retries', type=int, default=1,
                        help='Extra tries for transient camera/RPC failures after verified cleanup (0-3; default 1)')
    parser.add_argument('--simulator-timeout', type=float, default=5.,
                        help='Initial read-only RPC deadline (default 5s); a live RPC with a delayed episode gets one 15s recheck')
    parser.add_argument('--simulator-root', help='Opt in to owning a dedicated local CarlaUE4.sh process (Linux)')
    parser.add_argument('--simulator-gpu-uuid', help='Explicit NVIDIA GPU UUID for owned CARLA only')
    parser.add_argument('--simulator-gpu-pci', help='Matching GPU PCI address, e.g. pci-0000_c1_00_0')
    parser.add_argument('--simulator-restarts', type=int, default=8,
                        help='Whole-batch owned-simulator restart limit (0-68; default 8)')
    parser.add_argument('--simulator-startup-timeout', type=float, default=300.,
                        help='Owned-simulator RPC startup deadline in seconds (default 300)')
    parser.add_argument('--run', action='store_true', help='Execute; otherwise only verify source and print plan')
    args = parser.parse_args(argv)
    if not 0 <= args.transport_retries <= 3:
        parser.error('--transport-retries must be between 0 and 3')
    if not 0 < args.simulator_timeout <= 60:
        parser.error('--simulator-timeout must be positive and at most 60 seconds')
    if bool(args.simulator_root) != bool(args.simulator_gpu_uuid and args.simulator_gpu_pci):
        parser.error('--simulator-root requires both --simulator-gpu-uuid and --simulator-gpu-pci')
    if not args.simulator_root and (args.simulator_gpu_uuid or args.simulator_gpu_pci):
        parser.error('GPU binding options require --simulator-root')
    if not 0 <= args.simulator_restarts <= 68 or not 0 < args.simulator_startup_timeout <= 900:
        parser.error('Owned simulator restart/startup limits are out of range')
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
    with batch_lock(root), ExitStack() as lifetime:
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
        if args.simulator_root:
            from ep_driver_fail2drive.demo_simulator import DedicatedSimulator
            args._simulator_owner = lifetime.enter_context(DedicatedSimulator(
                args.simulator_root, root / 'simulator', simulator_health, host=args.host, port=args.port,
                gpu_uuid=args.simulator_gpu_uuid, gpu_pci=args.simulator_gpu_pci,
                restart_limit=args.simulator_restarts, startup_timeout=args.simulator_startup_timeout))
        save_progress(root, selected, routes, results, 'running')
        for tag in selected:
            print(f"[{tag}] {routes[tag]['family']} {routes[tag]['variant']}", flush=True)
            history = list(results.get(tag, {}).get('attempts', []))
            for retry in range(args.transport_retries+1):
                try:
                    item = collect_one(tag, routes[tag], plan, root, args)
                except KeyboardInterrupt:
                    save_progress(root, selected, routes, results, 'interrupted', tag)
                    raise
                attempt = {key: value for key, value in item.items() if key != 'attempts'}
                if not history or any(attempt.get(key) != history[-1].get(key) for key in
                                      ('path', 'log', 'failure_log', 'status')):
                    history.append(attempt)
                item['attempts'] = history
                results[tag] = item
                ready = save_progress(root, selected, routes, results, 'running', tag)
                if getattr(args, '_simulator_owner', None) is not None and item['status'] != 'ready_for_video_review':
                    save_progress(root, selected, routes, results, 'recovering_simulator', tag)
                    if recover_owned_simulator(item, args):
                        history[-1].update({key: value for key, value in item.items() if key != 'attempts'})
                    ready = save_progress(root, selected, routes, results, 'running', tag)
                if (item.get('retryable_transport_failure') and retry < args.transport_retries
                        and not item.get('batch_stop_reason')):
                    print(f"[{tag}] 服务已就绪，重试 {retry+1}/{args.transport_retries}；"
                          f"原记录保留于 {item.get('path',item.get('failure_log'))}", flush=True)
                    continue
                break
            if item['status'] != 'ready_for_video_review':
                detail = f"{item['failure_stage']}: {item['reason']}  ({item['failure_log']})"
            else:
                detail = item['path']
            batch_ready = sum(results.get(t, {}).get('status') == 'ready_for_video_review' for t in selected)
            print(f"[{tag}] {item['status']} (batch {batch_ready}/{len(selected)}, total {len(ready)}/{len(routes)})  {detail}", flush=True)
            if item.get('batch_stop_reason'):
                save_progress(root, selected, routes, results, 'interrupted', tag)
                print(f"本批次已停止（{item['batch_stop_reason']}）。已完成片段保留，其余路线仍待运行；"
                      f"请确认 {args.host}:{args.port} 专用服务可用、世界空闲后用相同命令续跑。", flush=True)
                return 2
        passed = all(results[t]['status'] == 'ready_for_video_review' for t in selected)
        save_progress(root, selected, routes, results, 'complete' if passed else 'completed_with_failures')
        return 0 if passed else 2


if __name__ == '__main__':
    if sys.argv[1:2] == ['--probe-simulator']:
        report = _read_simulator_health(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
        print(json.dumps(report, ensure_ascii=False))
        raise SystemExit(0 if report['ready'] else 2)
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
