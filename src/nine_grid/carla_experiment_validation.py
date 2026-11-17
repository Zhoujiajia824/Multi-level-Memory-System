"""Technical admission plus observed coverage, independent of planner success."""
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from .carla_experiment_spec import settings_from_manifest
from .carla_experiment_routes import RouteGeometry
from .carla_task_criteria import TaskCriteria

REQUIRED_CHECKS = ('route', 'actors', 'events', 'source_alignment', 'repeatability', 'cleanup', 'video')
CAMERAS = {'front_rgb', 'front_left_rgb', 'front_right_rgb', 'back_rgb', 'back_left_rgb', 'back_right_rgb'}
MEASUREMENTS = ('body_on_road', 'wrong_way', 'speed_limit', 'collision_events', 'lane_markings',
                'traffic_lights', 'stop_signs', 'public_priority', 'complete_physical_ticks')
INFRASTRUCTURE_KEYS = ('camera_width', 'camera_height', 'camera_fov', 'camera_height_m', 'camera_hz',
    'camera_save_hz', 'rgb_postprocess_effects', 'map_resolution_m', 'map_speed_cap_mps',
    'same_level_tolerance_m', 'arrival_margin_m')


def infrastructure_contract(config):
    missing = [key for key in INFRASTRUCTURE_KEYS if key not in config]
    if missing:
        raise ValueError('missing sensor/map contract: '+','.join(missing))
    return {key: config[key] for key in INFRASTRUCTURE_KEYS}


def sdk_supported(version):
    return isinstance(version, str) and re.match(r'^0\.9\.15(?:$|[-+ .])', version) is not None


def _read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _actors(initial):
    rows = initial.get('actors_at_source', [])
    if not rows:
        raise ValueError('initialization has no actors_at_source')
    actors = {}
    for row in rows:
        role = row.get('role')
        if not role or role in actors:
            raise ValueError('initial actor roles must be stable and unique')
        actors[role] = row
    return actors


def _xy(row):
    if 'x' in row:
        return float(row['x']), float(row['y'])
    if 'x_m' in row:
        return float(row['x_m']), float(row['y_m'])
    if 'position' in row:
        return tuple(row['position'][:2])
    raise ValueError('missing initial actor position')


def _physics(directory, settings, initial, evaluation, errors):
    task = TaskCriteria(settings)
    geometry = RouteGeometry([SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(
        x=point['x_m'], y=point['y_m'], z=point['z_m']))) for point in settings['route']['points']])
    hint = None
    source_actors = _actors(initial)
    roles = {role: str(row['carla_actor_id']) for role, row in source_actors.items()}
    if len(set(roles.values())) != len(roles):
        errors['actors'].append('initial_carla_actor_ids_are_not_unique')
    clock = {}; first_frame = None; previous_time = None
    with (directory/'physical_evaluation.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            tick, frame = row['physical_tick'], row['frame']
            if first_frame is None:
                first_frame = frame
            if type(tick) is not int or tick != len(clock) or frame != first_frame+tick:
                errors['source_alignment'].append('noncontiguous_physics')
                break
            timestamp = float(row['timestamp'])
            if (not math.isfinite(timestamp) or abs(float(row['sim_time_s'])-tick/50.) > 1e-6
                    or previous_time is not None and abs(timestamp-previous_time-.02) > 1e-5):
                errors['source_alignment'].append('physical_timestamp_mismatch')
            if row.get('instance_hash') != settings['instance_hash']:
                errors['source_alignment'].append('physical_instance_mismatch')
            previous_time = timestamp
            ego = row['ego']
            progress, _, hint, _ = geometry.project(SimpleNamespace(
                x=ego['position'][0], y=-ego['position'][1], z=ego.get('z', 0.)), hint)
            clock[tick] = {'frame': frame, 'timestamp': timestamp, 'route_s_m': row.get('route_s_m'),
                           'director_route_s_m': progress}
            if tick == 0:
                actual = {str(actor['id']): actor for actor in [ego]+row.get('actors', [])}
                for role, source_actor in source_actors.items():
                    actor = actual.get(roles[role])
                    if actor is None:
                        errors['actors'].append('initial_actor_absent_from_source_physics:'+role)
                        continue
                    if math.hypot(*[actor['position'][i]-source_actor['position'][i] for i in (0, 1)]) > .001:
                        errors['actors'].append('initial_bbox_differs_from_source_physics:'+role)
                    if math.hypot(*[actor['velocity'][i]-source_actor['velocity'][i] for i in (0, 1)]) > .001:
                        errors['actors'].append('initial_velocity_differs_from_source_physics:'+role)
                    if abs((actor['heading']-source_actor['heading']+math.pi) % (2*math.pi)-math.pi) > .0001:
                        errors['actors'].append('initial_heading_differs_from_source_physics:'+role)
                    for field in ('length', 'width', 'z'):
                        if field in source_actor and abs(float(actor[field])-float(source_actor[field])) > .001:
                            errors['actors'].append('initial_'+field+'_differs_from_source_physics:'+role)
            task.observe(tick, row['ego'], row.get('actors', []), roles)
    if (len(clock) != evaluation.get('physical_ticks', -1)+1 or len(clock) < 6
            or first_frame != initial.get('source_frame') or first_frame != evaluation.get('first_frame')):
        errors['source_alignment'].append('insufficient_or_mismatched_physics')
    recomputed = task.report()
    reported = evaluation.get('task_criteria', {})
    for key in ('route_checkpoints_completed', 'route_checkpoints_total', 'wrong_exit'):
        if reported.get(key) != recomputed[key]:
            errors['route'].append('task_report_does_not_match_physical_observations:'+key)
    return clock, recomputed


def _event_coverage(directory, settings, evaluation, clock, errors, evidence):
    sources = {actor['role']: actor for actor in settings['actors']}
    required = set(settings.get('evaluation', {}).get('required_event_ids', []))
    required.update(role for role, source in sources.items() if source.get('required'))
    rows = evaluation.get('scenario_events', [])
    observed = {str(row.get('id')): row for row in rows}
    transitions = {}; last = {}; event_path = directory/'scenario_events.jsonl'
    if sources or event_path.is_file():
        evidence.append({'path': str(event_path), 'sha256': _sha(event_path)})
        with event_path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); tick = row['physical_tick']; event = row['scenario_event']
                identity = str(event['id']); source = clock.get(tick)
                if (source is None or row.get('instance_hash') != settings['instance_hash']
                        or row['frame'] != source['frame'] or abs(row['timestamp']-source['timestamp']) > 1e-6):
                    errors['events'].append('scenario_event_physics_alignment:'+identity)
                    continue
                transitions.setdefault(identity, {})[event['state']] = tick
                last[identity] = event
        if last != observed:
            errors['events'].append('final_scenario_events_differ_from_event_log')
    covered = []; missing = []
    for identity in sorted(required):
        event = observed.get(identity, {})
        start, end = event.get('trigger_tick'), event.get('completed_tick')
        passive = sources.get(identity, {}).get('behavior') in ('follow', 'parked')
        valid = (event.get('state') == 'completed' and type(start) is int and type(end) is int
                 and 0 <= start <= end and start in clock and end in clock
                 and event.get('action_start_tick') == start+1
                 and transitions.get(identity, {}).get('active') == start+1
                 and transitions.get(identity, {}).get('completed') == end)
        if passive:
            valid = (event.get('state') == 'completed' and type(end) is int and end in clock
                     and transitions.get(identity, {}).get('completed') == end)
        if valid and identity in sources and not passive:
            trigger = sources[identity].get('trigger', {})
            bounds = trigger.get('ego_s_m', [0., float('inf')])
            # Shared contract: bbox centre on the frozen route's XY arc; height
            # only disambiguates stacked route segments.
            progress = clock[start]['director_route_s_m']
            if (progress is None or not bounds[0]-.001 <= progress <= bounds[1]+.001
                    or start < trigger.get('earliest_tick', 0)):
                errors['events'].append('event_trigger_does_not_match_physical_clock:'+identity)
                valid = False
        if valid:
            covered.append({'id': identity, 'trigger_tick': start, 'completed_tick': end})
        else:
            missing.append(identity)
    return {'required': sorted(required), 'completed': covered, 'uncovered': missing,
            'complete': not missing}


def _video(directory, video, clock, errors, evidence):
    if (video.get('status') != 'complete' or not video.get('playback_verified') or video.get('frames', 0) < 2
            or video.get('display_frames_dropped', 0) > 0 or video.get('source') != 'CARLA_same_frame_six_rgb_10hz'):
        errors['video'].append('video_recording_incomplete')
    # Absolute paths in server metadata may be stale after the evidence tree is moved.
    # Only a movie in this episode directory is admitted, and its bytes are hashed.
    name = str(video.get('path', '')).replace('\\', '/').rsplit('/', 1)[-1]
    movie = directory/name
    if not name or not movie.is_file() or movie.stat().st_size == 0:
        errors['video'].append('recorded_video_file_missing')
    else:
        evidence.append({'path': str(movie), 'sha256': _sha(movie)})
    ticks = []; count = 0
    with (directory/'video_frames.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); tick = row.get('physical_tick'); frame = row.get('carla_frame')
            if row.get('video_index') != count or type(tick) is not int or tick not in clock:
                errors['video'].append('video_index_or_physical_tick_invalid')
                break
            source = clock[tick]
            if frame != source['frame'] or abs(float(row['sim_time'])-(source['timestamp']-clock[0]['timestamp'])) > 1e-5:
                errors['video'].append('video_physics_alignment')
            cameras = row.get('source_cameras', [])
            if len(cameras) != 6 or {camera.get('view') for camera in cameras} != CAMERAS:
                errors['video'].append('six_rgb_sources_missing')
            elif any(camera.get('frame') != frame or
                     abs(float(camera.get('timestamp', float('inf')))-source['timestamp']) > 1e-6
                     for camera in cameras):
                errors['video'].append('rgb_source_frame_or_time_mismatch')
            if ticks and tick-ticks[-1] != 5:
                errors['video'].append('rgb_cadence_not_five_physical_ticks')
            ticks.append(tick); count += 1
    if (count != video.get('frames') or count != video.get('decoded_frames') or not ticks
            or ticks[0] > 4 or max(clock, default=0)-ticks[-1] > 4
            or video.get('first_carla_frame') != clock.get(ticks[0] if ticks else -1, {}).get('frame')
            or video.get('last_carla_frame') != clock.get(ticks[-1] if ticks else -1, {}).get('frame')):
        errors['video'].append('video_frame_count_or_terminal_coverage')


def _repeatability(initial, previous, errors):
    actors, prior = _actors(initial), _actors(previous)
    if set(actors) != set(prior):
        errors.append('actor_roles_changed')
    else:
        for role, actor in actors.items():
            old = prior[role]; ax, ay = _xy(actor); bx, by = _xy(old)
            if math.hypot(ax-bx, ay-by) > .25:
                errors.append('initial_position:'+role)
            if 'velocity' not in actor or 'velocity' not in old:
                errors.append('initial_velocity_missing:'+role)
            elif math.hypot(actor['velocity'][0]-old['velocity'][0], actor['velocity'][1]-old['velocity'][1]) > .2:
                errors.append('initial_velocity:'+role)
            if 'heading' not in actor or 'heading' not in old:
                errors.append('initial_heading_missing:'+role)
            elif abs((actor['heading']-old['heading']+math.pi) % (2*math.pi)-math.pi) > math.radians(3):
                errors.append('initial_heading:'+role)
    if initial.get('map_identity') != previous.get('map_identity'):
        errors.append('map_identity_changed')
    for key in ('client_version', 'server_version'):
        if initial.get(key) != previous.get(key):
            errors.append('sdk_build_changed:'+key)
    def lights(source):
        return sorted((str(row.get('opendrive_id')), tuple(round(v, 3) for v in row.get('pose', [])), row['state'])
                      for row in source.get('traffic_lights_at_source', []))
    if lights(initial) != lights(previous):
        errors.append('initial_signal_phase_changed')


def validate_runs(directories, coverage_directories=()):
    if len(directories) < 2:
        raise ValueError('technical admission requires at least two repeats of the same frozen instance')
    paths = [Path(path).resolve() for path in directories]
    witnesses = {str(Path(path).resolve()) for path in coverage_directories}-set(map(str, paths))
    paths.extend(Path(path) for path in sorted(witnesses) if Path(path) not in paths)
    if len(set(map(str, directories))) != len(directories) or len(set(paths[:len(directories)])) < 2:
        raise ValueError('two distinct episode directories are required')
    rows = []; first = None; first_initial = None; first_code = None; first_contract = None; files = []; reasons = []
    group_errors = {key: [] for key in REQUIRED_CHECKS}
    names = ('resolved_instance.json', 'configuration.json', 'summary.json', 'carla_initialization.json',
             'evaluation.json', 'carla_cleanup.json', 'video_recording.json', 'physical_evaluation.jsonl', 'video_frames.jsonl')
    for directory in paths:
        errors = {key: [] for key in REQUIRED_CHECKS}
        row = {'directory': str(directory), 'evidence_role': 'coverage_witness' if str(directory) in witnesses else 'repeatability_run',
               'coverage': {'route': False, 'events': False, 'complete': False}}
        try:
            for name in names:
                path = directory/name
                if not path.is_file():
                    raise ValueError('missing admission evidence: '+str(path))
                files.append({'path': str(path), 'sha256': _sha(path)})
            manifest = _read(directory/'resolved_instance.json'); settings = settings_from_manifest(manifest)
            if first is None:
                first = manifest
            if manifest['instance_hash'] != first['instance_hash']:
                raise ValueError('repeats and coverage witnesses must use the same frozen instance_hash')
            initial = _read(directory/'carla_initialization.json'); summary = _read(directory/'summary.json')
            cfg = _read(directory/'configuration.json')['parameters'].get('carla', {})
            ev = _read(directory/'evaluation.json'); cleanup = _read(directory/'carla_cleanup.json')
            video = _read(directory/'video_recording.json')
            code = cfg.get('implementation_sha256')
            if not isinstance(code, str) or len(code) != 64:
                errors['repeatability'].append('missing_implementation_identity')
            if first_code is None:
                first_code = code
            elif code != first_code:
                errors['repeatability'].append('implementation_changed_between_runs')
            contract = infrastructure_contract(cfg)
            if first_contract is None:
                first_contract = contract
            elif contract != first_contract:
                errors['repeatability'].append('sensor_or_map_contract_changed_between_runs')
            if cfg.get('experiment_phase') != 'development':
                errors['repeatability'].append('formal_results_cannot_be_selected_as_admission_evidence')
            if initial.get('instance_hash') != manifest['instance_hash']:
                errors['source_alignment'].append('initialization_identity')
            if initial.get('map_identity', {}).get('opendrive_sha256') != settings['route']['opendrive_sha256']:
                errors['route'].append('map_does_not_match_locked_route')
            if not sdk_supported(initial.get('client_version')) or not sdk_supported(initial.get('server_version')):
                errors['repeatability'].append('sdk_version')
            if cfg.get('runtime_mode') != 'sync':
                errors['source_alignment'].append('requires_sync_admission')
            if cfg.get('world_plan_debug'):
                errors['source_alignment'].append('world_overlay_contaminates_input')
            if not initial.get('rgb_enabled') or initial.get('rgb_initialization', {}).get('status') != 'aligned':
                errors['source_alignment'].append('camera_initial_alignment')
            if initial.get('physics_hz') != 50 or initial.get('camera_hz') != 10:
                errors['source_alignment'].append('actual_physics_or_camera_frequency')
            if ev.get('version') != 'carla-physical-evaluation-2.0' or ev.get('physical_hz') != 50:
                errors['source_alignment'].append('evaluation_version')
            if not ev.get('evaluation_valid') or not ev.get('scenario_valid'):
                errors['events'].append('invalid_evaluation_or_scenario')
            if ev.get('infrastructure_errors') or ev.get('scenario_errors') or ev.get('pending_event_count', 0):
                errors['events'].append('unresolved_events')
            if not ev.get('coverage', {}).get('complete_physical_ticks'):
                errors['source_alignment'].append('physical_tick_coverage')
            if any(ev.get('coverage', {}).get(channel) is not True for channel in MEASUREMENTS):
                errors['events'].append('incomplete_physical_measurement_channels')
            actor_roles = set(_actors(initial))
            if actor_roles != {'ego'} | {actor['role'] for actor in settings['actors']}:
                errors['actors'].append('initial_roles_differ_from_manifest')
            clock, task = _physics(directory, settings, initial, ev, errors)
            event_coverage = _event_coverage(directory, settings, ev, clock, errors, files)
            route_complete = (task['route_checkpoints_completed'] == task['route_checkpoints_total']
                              and task['route_checkpoints_total'] >= 2 and not task['wrong_exit'])
            row['coverage'] = {'route': route_complete, 'events': event_coverage['complete'],
                               'complete': route_complete and event_coverage['complete'],
                               'task': task, 'required_events': event_coverage}
            if cleanup.get('status') != 'complete' or not cleanup.get('verified'):
                errors['cleanup'].append('cleanup')
            if (summary.get('cleanup_errors') or summary.get('runtime_failure') or
                    summary.get('termination_reason') in ('runtime_error', 'initialization_failure')):
                errors['cleanup'].append('runtime_failure')
            _video(directory, video, clock, errors, files)
            if first_initial is None:
                first_initial = initial
            else:
                _repeatability(initial, first_initial, errors['repeatability'])
            row.update(physical_ticks=max(clock, default=-1), driving_outcome=ev.get('driving_outcome'),
                       strict_success=ev.get('strict_success'), client_version=initial.get('client_version'),
                       server_version=initial.get('server_version'))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            errors['source_alignment'].append(type(exc).__name__+': '+str(exc))
        row['technical_errors'] = {key: sorted(set(values)) for key, values in errors.items() if values}
        row['technical_passed'] = not row['technical_errors']
        for key, values in errors.items():
            group_errors[key].extend(values)
            reasons.extend({'directory': str(directory), 'check': key, 'reason': value} for value in sorted(set(values)))
        rows.append(row)
    covered = [row['directory'] for row in rows if row['technical_passed'] and row['coverage']['complete']]
    checks = {key: not values for key, values in group_errors.items()}
    smoke = all(checks.values())
    if not covered:
        checks['route'] = False; checks['events'] = False
        reasons.append({'check': 'scenario_coverage', 'reason': 'no_development_witness_traversed_all_route_checkpoints_and_completed_all_required_events'})
    return {'version': 'carla-instance-validation-2.0', 'instance_hash': first['instance_hash'] if first else None,
        'spec_sha256': first['spec_sha256'] if first else None, 'implementation_sha256': first_code,
        'infrastructure_contract': first_contract,
        'server_version': first_initial.get('server_version') if first_initial else None,
        'sdk_base_version': '0.9.15', 'passed': smoke and bool(covered), 'technical_smoke_passed': smoke,
        'scenario_coverage_passed': bool(covered), 'coverage_witnesses': covered,
        'checks': checks, 'runs': rows, 'evidence_files': files, 'reasons': reasons,
        'scope': 'technical_validation_and_observed_scenario_coverage; independent_of_planner_success; not_a_safety_proof',
        'limitations': ['验收只使用development证据，不能挑选A/B/C正式成功结果来替换冻结实例。',
                       '独立开发控制器可提供覆盖证据；行驶违规仍原样保留，不要求strict_success。',
                       '只验收该冻结seed/参数；不证明所有变体均有效，需审阅真实录像。']}


def verify_report(path, instance_hash):
    path = Path(path).resolve(); report = _read(path)
    if report.get('instance_hash') != instance_hash:
        raise ValueError('admission report instance mismatch')
    directories = []; witnesses = []
    for row in report.get('runs', []):
        directory = Path(row['directory'])
        if not directory.is_absolute():
            directory = (path.parent/directory).resolve()
        (witnesses if row.get('evidence_role') == 'coverage_witness' else directories).append(directory)
    recomputed = validate_runs(directories, witnesses)
    expected = {str((path.parent/Path(row['path'])).resolve()) if not Path(row['path']).is_absolute()
                else str(Path(row['path']).resolve()): row['sha256'] for row in report.get('evidence_files', [])}
    actual = {row['path']: row['sha256'] for row in recomputed['evidence_files']}
    if expected != actual:
        raise ValueError('admission evidence changed or evidence inventory incomplete')
    if not recomputed['passed']:
        raise ValueError('admission evidence lacks technical validity or observed scenario coverage')
    return recomputed
