"""Evidence gate for a separate Fail2Drive demonstration memory bank.

This module never drives CARLA, calls a teacher, approves text or publishes a
memory.  Completed collection and successful driving are different contracts.
Only actual, hashed collection evidence may reach the existing offline builder.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

from nine_grid.paths import filesystem_path
from .demo_catalog import FAMILIES
from .demo_events import requires_borrow_proof, verify_borrow_capture


DEMO_SUITE = 'fail2drive-targeted-demonstrations-1.0'
INPUT_PROTOCOL = 'observable-driving-recording-1.0'
CHECK_PROTOCOL = 'fail2drive-demo-memory-evidence-1.0'
PHASES = ('approach', 'interact', 'recovery')
TOLERANCE_S = 1e-5


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate evidence JSON field: '+key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite evidence: '+value)))


def _rows(path):
    result = []
    with path.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(
                    ValueError('Nonfinite evidence')))
            except ValueError as error:
                raise ValueError(f'Invalid JSONL at {path.name}:{number}') from error
            if not isinstance(value, dict):
                raise ValueError('Evidence JSONL rows must be objects: '+path.name)
            result.append(value)
    if not result:
        raise ValueError('Empty evidence: '+path.name)
    return result


def _contained(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '\\' in relative:
        raise ValueError('Evidence paths must be nonempty relative POSIX paths')
    target = filesystem_path(root/relative)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError('Evidence is missing or escapes the episode: '+relative)
    return target


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same(a, b, tolerance=TOLERANCE_S):
    return _number(a) and _number(b) and abs(a-b) <= tolerance


def _frame(row):
    # Both names describe an explicitly archived CARLA source frame.  Never
    # derive a frame from elapsed time or from a local image file number.
    value = row.get('carla_frame', row.get('frame'))
    if type(value) is not int or value < 0:
        raise ValueError('Missing actual CARLA frame')
    if 'carla_frame' in row and 'frame' in row and row['carla_frame'] != row['frame']:
        raise ValueError('Conflicting actual CARLA frame fields')
    return value


def _audit(row):
    audit = row.get('audit')
    if not isinstance(audit, dict):
        raise ValueError('Physical execution row lacks independent audit')
    for key in ('collision', 'offroad', 'rule_violation'):
        if audit.get(key) is not False:
            raise ValueError('Demonstration contains violation or unknown audit: '+key)
    if audit.get('projection_valid') is not True:
        raise ValueError('Demonstration contains invalid/unknown route projection')


def _actor_measurements(actor, now):
    if not isinstance(actor, dict):
        raise ValueError('Missing actual object state')
    for key in ('position', 'velocity', 'acceleration'):
        value = actor.get(key)
        if not isinstance(value, list) or len(value) != 2 or not all(map(_number, value)):
            raise ValueError('Missing actual finite XY object measurement: '+key)
    if not _number(actor.get('heading')) or any(not _number(actor.get(key)) or actor[key] <= 0
                                               for key in ('length', 'width')):
        raise ValueError('Missing actual object heading/body dimensions')
    for state in actor.get('history', []):
        if not _number(state.get('time')) or state['time'] > now+TOLERANCE_S:
            raise ValueError('Current object history contains future/unknown time evidence')


def _image(path, width=None, height=None):
    from PIL import Image
    with Image.open(path) as image:
        if image.width < 2 or image.height < 2 or (width is not None and image.width != width) or (
                height is not None and image.height != height):
            raise ValueError('Image dimensions disagree with camera evidence: '+path.name)
        image.verify()


def _camera_proof(root, row, required):
    relative = row.get('camera_manifest')
    manifest = _json(_contained(root, relative))
    required.add(relative)
    source = _frame(row)
    if (_frame(manifest) != source or manifest.get('frame_id') != row['snapshot']['frame_id']
            or not _same(manifest.get('time'), row['time'])):
        raise ValueError('Camera manifest and observed state are not the same frame/time')
    timestamp = manifest.get('carla_timestamp')
    diag = row['snapshot'].get('diagnostics', {})
    if not _number(timestamp) or not _same(diag.get('carla_timestamp'), timestamp):
        raise ValueError('Observed state lacks the matching native CARLA timestamp')
    cameras = manifest.get('cameras')
    if not isinstance(cameras, list) or len(cameras) != 6 or len({x.get('view') for x in cameras}) != 6:
        raise ValueError('Demonstration requires six distinct same-frame RGB cameras')
    for camera in cameras:
        if _frame(camera) != source or not _same(camera.get('timestamp'), timestamp):
            raise ValueError('RGB cameras are not synchronized to the observed state')
        width, height = camera.get('width'), camera.get('height')
        if (type(width) is not int or type(height) is not int or not _number(camera.get('fov_deg'))
                or not 0 < camera['fov_deg'] < 180):
            raise ValueError('Missing camera dimensions/FOV')
        transform = camera.get('world_to_camera_ue')
        if (not isinstance(transform, list) or len(transform) != 4
                or any(not isinstance(line, list) or len(line) != 4 or not all(map(_number, line))
                       for line in transform)):
            raise ValueError('Missing actual camera calibration')
        path = camera.get('rgb_path')
        _image(_contained(root, path), width, height)
        required.add(path)
    native = row.get('native_image')
    _image(_contained(root, native))
    required.add(native)


def _borrow_anchors(frames, execution_by_frame):
    """Select only actual imaged maneuver states with already verified future evidence."""
    anchors = []
    stages = (('waiting', 'interact'), ('borrowed', 'interact'), ('returned', 'recovery'))
    for stage, phase in stages:
        matches = [row for row in frames if execution_by_frame[_frame(row)].get('teacher', {})
                   .get('borrow', {}).get('stage') == stage]
        if matches:
            row = matches[0]
            anchors.append(dict(frame_id=row['snapshot']['frame_id'], time=row['time'],
                                phase=phase, maneuver_stage=stage, actual_behavior_frame=_frame(row)))
    return anchors


def _episode(root, requested_anchors=None):
    run = _json(_contained(root, 'run.json'))
    manifest = _json(_contained(root, 'capture_manifest.json'))
    if (run.get('demo_suite') != DEMO_SUITE or run.get('scenario_family') not in FAMILIES or run.get('source') != 'carla'
            or run.get('test_only') is not False or run.get('complete') is not True
            or run.get('official_evaluation') is not False or run.get('split') != 'independent_development'):
        raise ValueError('Incomplete, test-only or non-dedicated demonstration capture')
    if (manifest.get('protocol') != INPUT_PROTOCOL or manifest.get('source') != 'carla'
            or manifest.get('source_version') != run.get('source_version')
            or manifest.get('reference_driver') != run.get('reference_driver')
            or manifest.get('split') != run.get('split')
            or not isinstance(manifest.get('reference_driver'), str)
            or not manifest['reference_driver'].strip()
            or run.get('read_future_during_current_representation') is not False):
        raise ValueError('Demonstration source/driver provenance is missing or inconsistent')
    files = {'frames_path': 'frames.jsonl', 'execution_path': 'execution.jsonl',
             'event_evidence_path': 'event_evidence.jsonl', 'demo_evaluation_path': 'demo_evaluation.json'}
    if any(manifest.get(key) != value for key, value in files.items()):
        raise ValueError('Demonstration evidence file declarations are missing')
    required = {'run.json', 'raw_frames.jsonl', 'physical_events.json', *files.values()}
    evaluation = _json(_contained(root, files['demo_evaluation_path']))
    mode = run.get('capture_mode', 'full_route')
    if mode not in ('interaction_segment', 'full_route') or evaluation.get('capture_mode', 'full_route') != mode:
        raise ValueError('Capture and evaluation disagree on the demonstration mode')
    short = mode == 'interaction_segment'
    if type(evaluation.get('reached')) is not bool or (not short and not evaluation['reached']):
        raise ValueError('Actual demonstration has not passed: reached')
    for name in ('passed', 'interaction_complete', 'recovery_complete', 'projection_valid'):
        if evaluation.get(name) is not True:
            raise ValueError('Actual demonstration has not passed: '+name)
    for name in ('collision_count', 'offroad_count', 'rule_violation_count'):
        if type(evaluation.get(name)) is not int or evaluation[name] != 0:
            raise ValueError('Actual demonstration has adverse or unknown results: '+name)
    physical_events = _json(_contained(root, 'physical_events.json'))
    for name in ('collisions', 'line_events'):
        if not isinstance(physical_events.get(name), list) or physical_events[name]:
            raise ValueError('Final physical sensor evidence has adverse/unknown events: '+name)
    execution = _rows(_contained(root, files['execution_path']))
    by_frame = {}
    previous = None
    for item in execution:
        frame, now = _frame(item), item.get('time')
        if not _number(now) or (previous is not None and (frame != previous[0]+1
                or not _same(now-previous[1], .05))):
            raise ValueError('Physical execution evidence is not continuous 20Hz')
        control = item.get('applied_control')
        if (not isinstance(control, dict)
                or any(not _number(control.get(key)) for key in ('throttle', 'brake', 'steer'))
                or not 0 <= control['throttle'] <= 1 or not 0 <= control['brake'] <= 1
                or not -1 <= control['steer'] <= 1):
            raise ValueError('Missing/invalid actual applied vehicle control')
        _audit(item)
        by_frame[frame] = item
        previous = frame, now
    frames = _rows(_contained(root, files['frames_path']))
    raw_frames = _rows(_contained(root, 'raw_frames.jsonl'))
    if len(raw_frames) != len(frames):
        raise ValueError('Normalized captures do not match the complete raw observation sequence')
    previous = None
    frame_ids = set()
    for index, row in enumerate(frames):
        source, now = _frame(row), row.get('time')
        scene = row.get('snapshot', {})
        original = raw_frames[index]
        raw_scene = original.get('scene', {})
        if (_frame(original) != source or not _same(original.get('time'), now)
                or original.get('native_image') != row.get('native_image')
                or original.get('camera_manifest') != row.get('camera_manifest')
                or any(raw_scene.get('ego', {}).get(key) != scene.get('ego', {}).get(key)
                       for key in ('position', 'velocity', 'heading'))):
            raise ValueError('Normalized capture changed its raw source frame/image/ego state')
        if (row.get('index') != index or source not in by_frame or not _same(now, by_frame[source]['time'])
                or not _same(scene.get('time'), now) or not isinstance(scene.get('frame_id'), str)
                or not scene['frame_id'] or scene['frame_id'] in frame_ids):
            raise ValueError('Capture state is not bound to a unique physical execution frame')
        if previous is not None and (source-previous[0] != 20 or not _same(now-previous[1], 1.)):
            raise ValueError('Capture evidence is not actual 1Hz sampling')
        if scene.get('coordinates') != 'world_xy_m' or scene.get('diagnostics', {}).get('carla_frame') != source:
            raise ValueError('Capture lacks matching CARLA frame/right-handed coordinates')
        if (not scene.get('route') or len(scene['route']) < 2 or not scene.get('lanes')
                or not scene.get('diagnostics', {}).get('visibility_wkt')):
            raise ValueError('Capture lacks observed road/navigation/support evidence')
        _actor_measurements(scene.get('ego'), now)
        for actor in scene.get('actors', []):
            if actor.get('visible') is not True:
                raise ValueError('Current snapshot contains an object outside its observed set')
            _actor_measurements(actor, now)
        _camera_proof(root, row, required)
        _audit(row)
        frame_ids.add(scene['frame_id'])
        previous = source, now
    if len(frames) < 4:
        raise ValueError('No complete actual current/+1/+2/+3 second evidence')
    events = _rows(_contained(root, files['event_evidence_path']))
    parameters=run.get('parameters') or {}
    if requires_borrow_proof(parameters):
        for name in ('demonstration_actors.json','development_site_geometry.json','demonstration_design.json'):
            _contained(root,name)
            required.add(name)
        proof=verify_borrow_capture(root,parameters,execution,events)
        if evaluation.get('opposing_borrow_proof')!=proof:
            raise ValueError('Opposing-borrow capture proof differs from the replay of actual frames')
    phases = {}
    for event in events:
        source = _frame(event)
        if source not in by_frame or not _same(event.get('time'), by_frame[source]['time']):
            raise ValueError('Interaction event is not bound to actual execution frame/time')
        phase = event.get('phase')
        if phase not in PHASES or type(event.get('completed')) is not bool:
            raise ValueError('Interaction event has unknown phase/completion evidence')
        # Activation is a genuine incomplete event, not an unsuccessful phase
        # completion.  It remains audit evidence and cannot certify completion.
        if event['completed']:
            phases.setdefault(phase, []).append(event)
    if set(phases) != set(PHASES):
        raise ValueError('Missing actual approach/interaction/recovery phase evidence')
    if not all(min(row['time'] for row in phases[a]) < min(row['time'] for row in phases[b])
               for a, b in zip(PHASES, PHASES[1:])):
        raise ValueError('Actual approach/interaction/recovery evidence is out of order')
    if short:
        clip = evaluation.get('interaction_segment') or {}
        anchor = clip.get('recovery_anchor_frame')
        if type(anchor) is not int or clip.get('complete') is not True:
            raise ValueError('Missing completed recovery window')
        tail = [anchor + offset * 20 for offset in range(4)]
        observed = {_frame(row):row['time'] for row in frames}
        if (clip.get('future_source_frames') != tail or any(frame not in observed for frame in tail)
                or clip.get('future_source_times') != [observed[frame] for frame in tail]
                or clip.get('end_frame') != tail[-1] or _frame(execution[-1]) != tail[-1]):
            raise ValueError('Recovery window does not match recorded 1Hz images and execution')
        if not all(by_frame[frame].get('interaction_recovery', {}).get('stable_recovery') is True
                   for frame in range(anchor, tail[-1]+1)):
            raise ValueError('Recovery was interrupted during the recorded clip')
    eligible = []
    for row in frames:
        if all(any(_same(other['time'], row['time']+offset) for other in frames) for offset in (0., 1., 2., 3.)):
            eligible.append(row)
    if requested_anchors:
        anchors = [row for row in eligible if row['snapshot']['frame_id'] in requested_anchors]
        if len(anchors) != len(requested_anchors):
            raise ValueError('Explicit anchor is absent or lacks three seconds of actual future evidence')
        if not 1 <= len(anchors) <= 3:
            raise ValueError('Use one to three explicit anchors per demonstration')
        selected = [{'frame_id': row['snapshot']['frame_id'], 'time': row['time'], 'phase': 'explicit'} for row in anchors]
    elif short:
        activated = next((row for row in events if row.get('event') == 'activated'), None)
        if activated is None:
            raise ValueError('Clip lacks an observed interaction trigger')
        # Save cues before intervention and the actual response. E.g. the end
        # of a pedestrian crossing alone does not show why the driver yielded.
        onset = activated['time']
        handled = phases['interact'][0]['time']
        during = [row for row in execution if onset <= row['time'] <= handled]
        if not during:
            raise ValueError('No executed controls within the interaction')
        moving = [row for row in during if math.hypot(*row.get('observed_ego', {}).get('velocity', [])[:2]) > .2]
        response = max(moving or during, key=lambda row: max(row['applied_control']['brake'],
                                                             abs(row['applied_control']['steer'])))
        selected, used = [], set()
        for phase, target in [('approach', onset-1.), ('interact', response['time']), ('recovery', observed[anchor])]:
            row = min(eligible, key=lambda item: abs(item['time']-target))
            identity = row['snapshot']['frame_id']
            if identity not in used:
                selected.append(dict(frame_id=identity, time=row['time'], phase=phase))
                used.add(identity)
        borrow_anchors = _borrow_anchors(eligible, by_frame)
        if borrow_anchors:
            selected = borrow_anchors
    else:
        selected, used = [], set()
        for phase in PHASES:
            candidates = [(abs(row['time']-event['time']), row) for row in eligible for event in phases[phase]
                          if abs(row['time']-event['time']) <= .5+TOLERANCE_S]
            if not candidates:
                continue
            row = min(candidates, key=lambda item: (item[0], item[1]['time']))[1]
            identity = row['snapshot']['frame_id']
            if identity not in used:
                selected.append({'frame_id': identity, 'time': row['time'], 'phase': phase})
                used.add(identity)
        if not selected:
            raise ValueError('No real event anchor has complete three-second evidence; choose an explicit frame')
    sums = manifest.get('source_hashes')
    if not isinstance(sums, dict) or not required.issubset(sums):
        raise ValueError('Source manifest does not hash all execution/state/RGB/result evidence')
    for name, expected in sums.items():
        if (not isinstance(expected, str) or len(expected) != 64
                or _hash(_contained(root, name)) != expected):
            raise ValueError('Source evidence hash mismatch: '+name)
    return {'episode': str(root), 'scenario_family': run['scenario_family'], 'seed': run.get('seed'),
            'capture_mode': mode, 'reached': evaluation['reached'],
            'driver': run['reference_driver'], 'capture_count': len(frames), 'execution_count': len(execution),
            'anchors': selected, 'source_manifest_sha256': _hash(root/'capture_manifest.json'),
            'source_hashes': sums, 'test_only': False, 'actual_success_evidence': True}


def check_demonstrations(captures, *, anchor_frames=None, require_all_families=False):
    """Read-only checks.  A failed/unfinished episode is never silently skipped."""
    paths = [filesystem_path(path) for path in captures]
    roots = []
    for path in paths:
        if not path.is_dir():
            raise ValueError('Demonstration directory does not exist: '+str(path))
        found = [path] if (path/'run.json').is_file() else sorted(p.parent for p in path.rglob('run.json'))
        if not found:
            raise ValueError('No demonstration run.json found: '+str(path))
        roots.extend(found)
    if len(set(roots)) != len(roots):
        raise ValueError('Demonstration directories overlap or repeat')
    if anchor_frames and (len(roots) != 1 or len(set(anchor_frames)) != len(anchor_frames)):
        raise ValueError('Explicit anchors require one episode and unique frame IDs')
    episodes, rejected = [], []
    for root in roots:
        try:
            episodes.append(_episode(root, anchor_frames))
        except (ValueError, KeyError, TypeError, OSError) as error:
            rejected.append({'episode': str(root), 'reason': str(error)})
    families = sorted({item['scenario_family'] for item in episodes})
    # Completeness is descriptive unless explicitly requested.  Never invent
    # seventeen successful demonstrations because a catalog contains 17 names.
    if require_all_families and set(families) != set(FAMILIES):
        rejected.append({'episode': None, 'reason': 'Expected actual successful demonstrations for 17 distinct families'})
    return {'protocol': CHECK_PROTOCOL, 'demo_suite': DEMO_SUITE, 'valid': bool(episodes) and not rejected,
            'successful_episodes': len(episodes), 'scenario_families': families,
            'all_17_families_present': set(families) == set(FAMILIES), 'episodes': episodes, 'rejected': rejected,
            'teacher_called': False, 'human_review_fabricated': False, 'memory_published': False}


def prepare_demonstrations(captures, output, workbench_root, *, pipeline_config=None, dino_model=None,
                           device='cpu', anchor_frames=None, require_all_families=False):
    """Prepare real evidence with an explicitly selected existing builder.

    The builder is imported read-only; no package installation, source copying,
    teacher request, review or publication is performed.
    """
    checked = check_demonstrations(captures, anchor_frames=anchor_frames,
                                  require_all_families=require_all_families)
    if not checked['valid']:
        raise ValueError('Demonstrations are not eligible: '+json.dumps(checked['rejected'], ensure_ascii=False))
    output, workbench = filesystem_path(output), filesystem_path(workbench_root)
    if output.exists():
        raise ValueError('Preparation output must be a new directory')
    for source in [filesystem_path(item['episode']) for item in checked['episodes']]+[workbench]:
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Preparation output must be separate from source episodes and the workbench')
    module_path = workbench/'src'/'offline_memory_builder'
    if not (module_path/'pipeline.py').is_file() or not (module_path/'storage.py').is_file():
        raise ValueError('Explicit workbench root lacks the existing offline builder')
    if 'offline_memory_builder' in sys.modules:
        loaded = filesystem_path(sys.modules['offline_memory_builder'].__file__).parent
        if loaded != module_path:
            raise ValueError('A different offline workbench is already imported')
    original_path, original_bytecode = list(sys.path), sys.dont_write_bytecode
    old_matplotlib = os.environ.get('MPLCONFIGDIR')
    # All newly generated caches belong to the new build, never the other
    # working tree or an existing published memory library.
    output.mkdir(parents=True)
    os.environ['MPLCONFIGDIR'] = str(output/'.matplotlib')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(workbench/'src'))
    try:
        pipeline = importlib.import_module('offline_memory_builder.pipeline')
        storage = importlib.import_module('offline_memory_builder.storage')
        if 'selected_anchor_frame_ids' not in inspect.signature(pipeline.prepare).parameters:
            raise ValueError('Workbench prepare API does not support explicit actual anchor IDs')
        cfg = deepcopy(storage.configuration(pipeline_config or workbench/'config'/'pipeline.yaml'))
        cfg['sampling'].update(interval_s=1., start_s=0., future_offsets_s=[0., 1., 2., 3.],
                               time_tolerance_s=.001, max_per_clip=3)
        cfg['features']['dino_model'] = str(filesystem_path(dino_model)) if dino_model else None
        cfg['features']['device'] = device
        preparations = []
        for episode in checked['episodes']:
            # Re-read all source hashes immediately before the external builder
            # reads the evidence.  Editing a collection cannot reuse approval.
            current = _episode(filesystem_path(episode['episode']),
                               [item['frame_id'] for item in episode['anchors']])
            if current['source_manifest_sha256'] != episode['source_manifest_sha256']:
                raise ValueError('Demonstration changed after evidence check')
            result = pipeline.prepare(cfg, filesystem_path(episode['episode']), output,
                model_path=cfg['features']['dino_model'], device=device, anchor_alignment='strict',
                selected_anchor_frame_ids=[item['frame_id'] for item in episode['anchors']])
            preparations.append(result)
            if _hash(filesystem_path(episode['episode'])/'capture_manifest.json') != episode['source_manifest_sha256']:
                raise ValueError('Demonstration manifest changed during preparation')
            for name, expected in episode['source_hashes'].items():
                if _hash(_contained(filesystem_path(episode['episode']), name)) != expected:
                    raise ValueError('Demonstration evidence changed during preparation: '+name)
        provenance = dict(checked, status='prepared_pending_teacher_and_human_review',
                          workbench_root=str(workbench), preparations=preparations,
                          published=False, reviewed=False)
        storage.atomic_json(output/'demo_preparation.json', provenance)
        return provenance
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode
        if old_matplotlib is None:
            os.environ.pop('MPLCONFIGDIR', None)
        else:
            os.environ['MPLCONFIGDIR'] = old_matplotlib
