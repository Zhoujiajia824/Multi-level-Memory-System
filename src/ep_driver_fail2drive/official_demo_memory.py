"""Offline evidence gate for demonstrations in unchanged official scenarios.

This module never imports CARLA, drives actors, calls an annotation model,
approves text or publishes a library. Official episode outcomes certify driving;
raw observations remain current-only and are never rewritten to hide failures.
"""
from copy import deepcopy
import importlib
import inspect
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

from nine_grid.paths import filesystem_path
from . import OFFICIAL_COMMIT
from .demo_catalog import FAMILIES
from .demo_memory import (_json, _rows, _contained, _hash, _number, _same,
                          _frame, _actor_measurements, _camera_proof)

SUITE = 'fail2drive-official-demonstrations-1.0'
SPLIT = 'official_scenario_demonstration'
PROTOCOL = 'fail2drive-official-demo-evidence-1.0'
INPUT_PROTOCOL = 'observable-driving-recording-1.0'
CAMERAS = frozenset(('front_rgb','front_left_rgb','front_right_rgb','back_rgb','back_left_rgb','back_right_rgb'))
INFRACTIONS = frozenset(('collisions_layout','collisions_pedestrian','collisions_vehicle','red_light',
    'stop_infraction','outside_route_lanes','min_speed_infractions','yield_emergency_vehicle_infractions',
    'scenario_timeouts','route_dev','vehicle_blocked','route_timeout'))


def _write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def _copy_once(source, destination):
    if destination.exists():
        if _hash(source) != _hash(destination):
            raise ValueError('Previously captured official evidence changed: '+destination.name)
    else:
        shutil.copyfile(source,destination)


def _root(directory):
    root = filesystem_path(directory)
    if (root/'run.json').is_file():
        return root
    candidates = [p.parent for p in root.rglob('run.json')]
    if len(candidates) != 1:
        raise ValueError('Select the exact single agent capture directory; found '+str(len(candidates)))
    return candidates[0]


def official_outcome(data, route_id):
    """Validate the complete upstream single-route checkpoint, including every infraction."""
    reasons=[]
    checkpoint=data.get('_checkpoint',{})
    records=checkpoint.get('records')
    record=records[0] if isinstance(records,list) and len(records)==1 and isinstance(records[0],dict) else {}
    global_record=checkpoint.get('global_record',{})
    if data.get('entry_status')!='Finished' or data.get('eligible') is not True or checkpoint.get('progress')!=[1,1]:
        reasons.append('official_evaluator_not_finished_or_ineligible')
    if type(route_id) is not int or record.get('route_id')!='RouteScenario_%s_rep0'%route_id:
        reasons.append('official_route_identity_mismatch')
    if record.get('status')!='Perfect' or global_record.get('status')!='Perfect':
        reasons.append('official_route_not_perfect')
    scores=record.get('scores',{})
    if (not _same(scores.get('score_route'),100.) or not _same(scores.get('score_penalty'),1.)
            or not _same(scores.get('score_composed'),100.)):
        reasons.append('official_route_incomplete_or_penalized')
    infractions=record.get('infractions')
    if (not isinstance(infractions,dict) or not INFRACTIONS.issubset(infractions)
            or any(not isinstance(value,list) or value for value in infractions.values())
            or record.get('num_infractions')!=0):
        reasons.append('official_infractions_nonempty_or_unknown')
    totals=global_record.get('infractions')
    if (not isinstance(totals,dict) or not INFRACTIONS.issubset(totals)
            or any(not _number(value) or value!=0 for value in totals.values())):
        reasons.append('official_global_infractions_nonzero_or_unknown')
    return dict(valid=not reasons,passed=not reasons,reasons=reasons,
        entry_status=data.get('entry_status'),status=record.get('status'),scores=scores,
        infractions=infractions,global_infractions=totals,official_route_id=route_id)


def _identity(root, run, launch):
    if (run.get('demo_suite')!=SUITE or run.get('source')!='carla' or run.get('split')!=SPLIT
            or run.get('complete') is not True or run.get('test_only') is not False
            or run.get('official_evaluation') is not True or run.get('errors')!=[]
            or run.get('read_future_during_current_representation') is not False
            or run.get('normalization_protocol')!=INPUT_PROTOCOL):
        raise ValueError('Incomplete, test-only or non-official-scenario demonstration')
    if (launch.get('official_commit')!=OFFICIAL_COMMIT or launch.get('scenario_mutations') is not False
            or launch.get('driver_kind')!='official_autopilot' or launch.get('purpose')!='demonstration_only'):
        raise ValueError('Official launch identity or ego-only demonstration purpose is missing')
    route_id=launch.get('official_route_id')
    seed=launch.get('repetition_seed')
    if type(route_id) is not int or route_id<0 or type(seed) is not int or seed<1:
        raise ValueError('Official route/seed identity is invalid')
    if launch.get('traffic_manager_seed')!=route_id%1000+10000*seed:
        raise ValueError('Official traffic-manager seed changed')
    if (run.get('official_route_id')!=route_id or run.get('seed',run.get('repetition_seed'))!=seed
            or run.get('scenario_family')!=launch.get('scenario_family') or run.get('scenario_family') not in FAMILIES
            or run.get('driver_kind')!='official_autopilot' or run.get('purpose')!='demonstration_only'
            or run.get('scenario_mutations') is not False or run.get('official_xml_unchanged') is not True):
        raise ValueError('Captured run differs from its official launch')
    route_path=_contained(root,'official_route.xml')
    if _hash(route_path)!=launch.get('routes_sha256'):
        raise ValueError('Official route XML bytes changed')
    origin=Path(launch.get('official_routes_path','')).name
    match=re.fullmatch(r'(Base|Generalization)_(.+)_(\d{4})\.xml',origin)
    if not match or int(match.group(3))!=route_id or match.group(2)!=run['scenario_family']:
        raise ValueError('Scenario family was not derived from the original official route filename')
    routes=ET.parse(route_path).getroot().findall('route')
    if len(routes)!=1 or routes[0].get('id')!=str(route_id):
        raise ValueError('Official XML contains another or multiple routes')
    if run.get('world_map','').split('/')[-1]!=routes[0].get('town'):
        raise ValueError('Captured CARLA map differs from the official XML')
    if run.get('ego_vehicle_type')!='vehicle.lincoln.mkz_2020':
        raise ValueError('Captured ego vehicle differs from the pinned official RouteScenario')
    if not isinstance(run.get('source_version'),str) or not run['source_version']:
        raise ValueError('Missing actual simulator version')
    return route_id


def _guard(root):
    guard=_json(_contained(root,'expert_world_guard.json'))
    if (guard.get('complete') is not True or guard.get('scope')!='ego_control_only'
            or guard.get('world_mutations_allowed') is not False
            or guard.get('non_ego_mutation_attempts')!=[] or guard.get('failures',guard.get('errors'))!=[]):
        raise ValueError('Expert world guard is incomplete or observed non-ego mutations')
    return guard


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        allow_nan=False,separators=(',',':')).encode()).hexdigest()


def _binding(root,run,launch,guard):
    """Bind the source recorder, expert and evaluator launch, without remote paths."""
    from .official_expert import SOURCES,SPEED_LIMIT_DATA,LOCAL_BASE,LOCAL_BASE_SHA256
    configuration=_json(_contained(root,'configuration.json'))
    inputs=launch.get('fingerprint_inputs',{})
    association=configuration.get('archive_association',{})
    if (not isinstance(inputs,dict) or launch.get('fingerprint')!=_digest(inputs)
            or inputs.get('config')!=configuration or launch.get('config')!=configuration
            or run.get('archive_association')!=association
            or inputs.get('official_commit')!=OFFICIAL_COMMIT or run.get('official_commit')!=OFFICIAL_COMMIT
            or inputs.get('official_route_id')!=launch['official_route_id']
            or inputs.get('repetition_seed')!=launch['repetition_seed']
            or inputs.get('traffic_manager_seed')!=launch['traffic_manager_seed']
            or inputs.get('routes_sha256')!=launch['routes_sha256']):
        raise ValueError('Capture configuration/run differs from its fingerprinted official launch')
    if (association.get('official_route_id')!=launch['official_route_id']
            or association.get('repetition_seed')!=launch['repetition_seed']
            or association.get('traffic_manager_seed')!=launch['traffic_manager_seed']
            or association.get('official_routes_path')!=launch['official_routes_path']
            or association.get('repetitions')!=1 or association.get('routes_subset')!=''
            or configuration.get('official_commit')!=OFFICIAL_COMMIT):
        raise ValueError('Official checkpoint association changed route, repetitions or seed')
    spec=configuration.get('official_demo',{})
    if (spec.get('scenario_family')!=run['scenario_family'] or spec.get('official_route_id')!=run['official_route_id']
            or spec.get('driver')!='official_autopilot' or spec.get('purpose')!='demonstration_only'
            or spec.get('scenario_mutations') is not False):
        raise ValueError('Recorded expert configuration differs from its official scenario')
    expert=launch.get('official_expert_files_sha256',{})
    if (guard.get('official_commit')!=OFFICIAL_COMMIT or guard.get('sources_sha256')!=SOURCES
            or guard.get('data_sha256')!=SPEED_LIMIT_DATA
            or guard.get('base_sources_sha256')!={LOCAL_BASE:LOCAL_BASE_SHA256}
            or guard.get('private_seed')!=launch['repetition_seed']
            or guard.get('official_source_files_modified') is not False
            or not isinstance(expert,dict)
            or any(expert.get('team_code/'+name)!=digest for name,digest in SOURCES.items())
            or any(expert.get(name)!=digest for name,digest in SPEED_LIMIT_DATA.items())
            or inputs.get('official_expert_files_sha256')!=expert
            or inputs.get('official_expert_source_sha256')!=_digest(expert)
            or launch.get('official_expert_source_sha256')!=_digest(expert)
            or spec.get('official_expert_source_sha256')!=_digest(expert)
            or inputs.get('demonstration_implementation_sha256')!=launch.get('implementation_sha256')
            or not isinstance(launch.get('implementation_sha256'),dict)
            or not launch['implementation_sha256']):
        raise ValueError('Expert guard is not bound to the pinned launch source/data identity')
    identity=_json(_contained(root,'carla_identity.json'))
    if (identity.get('client_version')!=run['source_version'] or identity.get('server_version')!=run['source_version']
            or identity.get('map')!=run['world_map'] or identity.get('physics_hz')!=20
            or identity.get('synchronous_mode') is not True):
        raise ValueError('Recorded simulator identity differs from the captured official run')
    return configuration


def _matrix(value):
    return (isinstance(value,list) and len(value)==4 and all(isinstance(row,list) and len(row)==4
            and all(map(_number,row)) for row in value))


def _calibration(root,row):
    manifest=_json(_contained(root,row['camera_manifest']))
    cameras=manifest.get('cameras',[])
    if {c.get('view') for c in cameras}!=CAMERAS:
        raise ValueError('Official source requires the exact six camera views')
    corrected=[]
    for camera in cameras:
        world_to_camera=camera.get('world_to_camera_ue')
        offset=camera.get('api_global_minus_ue_origin_m')
        if (not _matrix(world_to_camera) or not isinstance(offset,list) or len(offset)!=3
                or not all(map(_number,offset))):
            raise ValueError('Missing same-frame UE/global camera calibration evidence')
        global_matrix=deepcopy(world_to_camera)
        # p_UE = p_API - offset; camera = W2C_UE @ p_UE.
        for i in range(3):
            global_matrix[i][3]-=sum(world_to_camera[i][j]*offset[j] for j in range(3))
        corrected.append(dict(view=camera['view'],frame=_frame(camera),timestamp=camera['timestamp'],
            world_to_camera_global=global_matrix,world_to_camera_ue=world_to_camera,
            api_global_minus_ue_origin_m=offset,source_manifest=row['camera_manifest']))
    return corrected


def _execution(root):
    rows=_rows(_contained(root,'execution.jsonl'))
    by_frame={};prior=None;previous_control=None
    for row in rows:
        frame,now=_frame(row),row.get('time')
        if not _number(now) or prior is not None and (frame!=prior[0]+1 or not _same(now-prior[1],.05)):
            raise ValueError('Execution is not a continuous actual 20Hz source sequence')
        if (row.get('source_frame')!=frame or not _same(row.get('source_time'),now)
                or row.get('driver')!='official_autopilot' or row.get('control_applied_by')!='official_evaluator'
                or row.get('observed_applied_control')!=row.get('applied_control')):
            raise ValueError('Expert control is not bound to the actual official source frame')
        for field in ('applied_control','returned_control'):
            control=row.get(field)
            if (not isinstance(control,dict) or any(not _number(control.get(k)) for k in ('throttle','brake','steer'))
                    or not 0<=control['throttle']<=1 or not 0<=control['brake']<=1 or not -1<=control['steer']<=1):
                raise ValueError('Missing actual applied/returned control: '+field)
        actual=row['applied_control']
        if previous_control is not None and (any(not _same(actual[k],previous_control[k],1e-4)
                for k in ('throttle','brake','steer')) or any(actual.get(k)!=previous_control.get(k)
                for k in ('hand_brake','reverse'))):
            raise ValueError('Observed actual pedals/steering differ from the preceding expert control')
        _actor_measurements(row.get('observed_ego'),now)
        by_frame[frame]=row;prior=frame,now;previous_control=row['returned_control']
    return rows,by_frame


def _video(root,execution):
    execution_count=len(execution)
    status=_json(_contained(root,'planning/video_recording.json'))
    if (status.get('requested') is not True or status.get('status')!='complete'
            or status.get('playback_verified') is not True or status.get('frames')!=execution_count
            or status.get('decoded_frames')!=execution_count or status.get('interpolated_frames')!=0
            or not _same(status.get('fps'),20.)
            or status.get('first_source_frame')!=execution[0]['frame']
            or status.get('last_source_frame')!=execution[-1]['frame']
            or not _same(status.get('first_source_time'),execution[0]['time'])
            or not _same(status.get('last_source_time'),execution[-1]['time'])):
        raise ValueError('Actual demonstration video did not pass full source-frame decode verification')
    _contained(root,'planning/'+status['path'])


def _counts(run,guard,execution,raw):
    if (run.get('raw_frame_count')!=len(raw) or guard.get('control_calls')!=len(execution)
            or guard.get('source_frame')!=execution[-1]['frame']
            or not _same(guard.get('source_time'),execution[-1]['time'])
            or raw[0]['frame']!=execution[0]['frame']
            or len(raw)!=(len(execution)-1)//20+1):
        raise ValueError('Guard/run counters do not bind the complete source and execution streams')


def _observations(root, execution_by_frame, *, normalized):
    raw=_rows(_contained(root,'raw_frames.jsonl'))
    frames=_rows(_contained(root,'frames.jsonl')) if normalized else None
    if frames is not None and len(frames)!=len(raw):
        raise ValueError('Normalized and raw observation counts differ')
    required={'run.json','configuration.json','carla_identity.json','raw_frames.jsonl','execution.jsonl','expert_world_guard.json','official_route.xml',
              'official_result.json','official_launch.json','official_demo_evaluation.json','planning/video_recording.json'}
    prior=None
    for index,row in enumerate(raw):
        frame,now=_frame(row),row.get('time');scene=row.get('scene',{})
        if (frame not in execution_by_frame or not _same(now,execution_by_frame[frame]['time'])
                or not _same(scene.get('time'),now) or scene.get('diagnostics',{}).get('carla_frame')!=frame):
            raise ValueError('Raw observation is not bound to its actual executed source frame')
        if prior is not None and (frame-prior[0]!=20 or not _same(now-prior[1],1.)):
            raise ValueError('Raw captures are not actual 1Hz samples')
        _actor_measurements(scene.get('ego'),now)
        if scene['ego']!=execution_by_frame[frame]['observed_ego']:
            raise ValueError('Raw capture and execution have different measured ego states')
        for actor in scene.get('actors',[]):
            _actor_measurements(actor,now)
            if actor.get('visible') is not True:
                raise ValueError('Current observation contains a non-observed object')
        proof=dict(row,snapshot=scene)
        _camera_proof(root,proof,required)
        _calibration(root,row)
        images=row.get('images',[])
        if len(images)!=6 or {image.get('id') for image in images}!=CAMERAS:
            raise ValueError('Frozen raw packet lacks six source images')
        calibration=_json(_contained(root,row['camera_manifest']))
        camera_paths={c['view']:c['rgb_path'] for c in calibration['cameras']}
        for image in images:
            if image.get('frame')!=frame or not _same(image.get('timestamp'),now):
                raise ValueError('Raw source RGB identity differs from state')
            # The immutable packet keeps its original absolute server path.
            # The manifest owns the relocatable asset path used after transfer.
            if not str(image.get('path','')).replace('\\','/').endswith('/'+camera_paths[image['id']]):
                raise ValueError('Raw source RGB path differs from its camera manifest')
        if frames is not None:
            exported=frames[index];snapshot=exported.get('snapshot',{})
            if (exported.get('index')!=index or _frame(exported)!=frame or not _same(exported.get('time'),now)
                    or exported.get('native_image')!=row['native_image']
                    or exported.get('camera_manifest')!=row['camera_manifest']
                    or any(snapshot.get('ego',{}).get(k)!=scene['ego'].get(k)
                           for k in ('position','velocity','acceleration','heading','length','width'))
                    or snapshot.get('coordinates')!='world_xy_m'
                    or snapshot.get('frame_id')!=scene.get('frame_id')
                    or not snapshot.get('diagnostics',{}).get('visibility_wkt')):
                raise ValueError('Normalization changed actual source kinematics or scene identity')
            expected=_calibration(root,row)
            if snapshot.get('diagnostics',{}).get('camera_global_calibration')!=expected:
                raise ValueError('Normalized global camera calibration is not bound to raw UE evidence')
            expected_images=[dict(view=c['view'],path=c['rgb_path'],frame_id=scene['frame_id'],time=now)
                             for c in calibration['cameras']]
            if snapshot.get('assets',{}).get('camera_images')!=expected_images:
                raise ValueError('Normalized camera assets differ from their relocatable raw evidence')
            audit=exported.get('audit',{})
            if (audit.get('source')!='official_episode_outcome' or audit.get('scope')!='whole_episode'
                    or any(audit.get(k) is not False for k in ('collision','offroad','rule_violation'))):
                raise ValueError('Normalized driving outcome is not attributed to the official whole episode')
        prior=frame,now
    if len(raw)<4:
        raise ValueError('Missing actual current/+1/+2/+3-second source evidence')
    return raw,frames,required


def select_anchors(frames,execution,requested=None):
    """Use actual controls offline, not scenario labels or hidden event plans."""
    eligible=[row for row in frames if all(any(_same(other['time'],row['time']+offset)
              for other in frames) for offset in (0.,1.,2.,3.))]
    if requested:
        if len(set(requested))!=len(requested) or not 1<=len(requested)<=3:
            raise ValueError('Select one to three unique actual anchor frame IDs')
        selected=[dict(frame_id=row['snapshot']['frame_id'],time=row['time'],anchor_reason='human_selected')
                  for row in eligible if row['snapshot']['frame_id'] in requested]
        if len(selected)!=len(requested):
            raise ValueError('Explicit anchor is absent or lacks actual three-second evidence')
        return selected
    events={};stopped_since=None;previous=None
    for row in execution:
        speed=math.hypot(*row['observed_ego']['velocity']);control=row['returned_control'];stamp=row['time']
        old=previous['returned_control'] if previous else {'brake':0.,'steer':0.}
        if control['brake']>.2 and old['brake']<=.05 and speed>1.:
            events.setdefault('observed_braking_onset',stamp)
        if abs(control['steer'])>.12 and abs(old['steer'])<=.12 and speed>.5:
            events.setdefault('observed_turning_or_bypass_onset',stamp)
        if speed<.15:
            stopped_since=stamp if stopped_since is None else stopped_since
        elif speed>.5:
            if stopped_since is not None and stamp-stopped_since>=.5 and control['throttle']>.05:
                events.setdefault('observed_restart_after_stop',stamp)
            stopped_since=None
        previous=row
    selected=[];used=set()
    for reason,stamp in events.items():
        candidates=[row for row in eligible if abs(row['time']-stamp)<=.5+1e-5]
        if not candidates:continue
        row=min(candidates,key=lambda r:(abs(r['time']-stamp),r['time']))
        identity=row['snapshot']['frame_id']
        if identity not in used:
            selected.append(dict(frame_id=identity,time=row['time'],anchor_reason=reason,observed_event_time=stamp));used.add(identity)
        if len(selected)==3:break
    if not selected and eligible:
        for index in sorted({0,len(eligible)//2,len(eligible)-1}):
            row=eligible[index]
            selected.append(dict(frame_id=row['snapshot']['frame_id'],time=row['time'],anchor_reason='uniform_no_observed_control_event'))
    if not selected:raise ValueError('No source anchor with complete actual future evidence')
    return selected


def _export(root,run,raw):
    from shapely.geometry import Polygon
    from nine_grid.contracts import plain
    from .protocol import SCHEMA
    from .worker import decode_scene
    destination=root/'frames.jsonl'
    if destination.exists():raise ValueError('Existing normalized capture cannot be overwritten')
    temporary=root/'frames.jsonl.tmp'
    with temporary.open('x',encoding='utf-8') as stream:
        for index,row in enumerate(raw):
            manifest=_json(_contained(root,row['camera_manifest']))
            images=[dict(id=c['view'],frame=row['frame'],timestamp=row['time'],path=str(_contained(root,c['rgb_path'])))
                    for c in manifest['cameras']]
            packet=dict(schema=SCHEMA,route_id=row['route_id'],frame=row['frame'],time=row['time'],scene=row['scene'],images=images)
            scene=decode_scene(packet)
            scene.assets['camera_images']=[dict(view=c['view'],path=c['rgb_path'],
                frame_id=scene.frame_id,time=row['time']) for c in manifest['cameras']]
            scene.assets['native_image']=row['native_image']
            scene.assets['camera_manifest_json']=row['camera_manifest']
            visibility=Polygon(row['scene']['diagnostics']['visibility_polygon'])
            if not visibility.is_valid or visibility.is_empty:raise ValueError('Invalid source observation support')
            scene.diagnostics.update(visibility_wkt=visibility.wkt,driver_source=run['reference_driver'],
                camera_global_calibration=_calibration(root,row),
                demonstration_source=dict(official_scenario_used=True,original_xml_unchanged=True,
                    official_route_id=run['official_route_id'],future_used_for_current_features=False,
                    exact_npc_frame_replay_claimed=False))
            snapshot=plain(scene)
            snapshot['diagnostics']['captured_motion_models']={actor['id']:actor.pop('motion_model',None)
                for actor in [snapshot['ego'],*snapshot['actors']]}
            audit=dict(source='official_episode_outcome',scope='whole_episode',collision=False,offroad=False,rule_violation=False,
                per_frame_independent_safety_claimed=False)
            stream.write(json.dumps(dict(index=index,frame=row['frame'],time=row['time'],snapshot=snapshot,
                native_image=row['native_image'],camera_manifest=row['camera_manifest'],audit=audit),ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(destination)


def finalize_official_capture(run_dir, official_checkpoint, launch_metadata):
    """Freeze new derivative evidence after evaluator/Agent teardown; never edit raw data."""
    root=_root(run_dir)
    checkpoint=filesystem_path(official_checkpoint)
    launch=_json(filesystem_path(launch_metadata)) if isinstance(launch_metadata,(str,Path)) else deepcopy(launch_metadata)
    if not isinstance(launch,dict):raise ValueError('Launch metadata must be an object')
    existing=root/'capture_manifest.json'
    if existing.exists():
        if (_hash(checkpoint)!=_hash(_contained(root,'official_result.json'))
                or launch!=_json(_contained(root,'official_launch.json'))):
            raise ValueError('Frozen capture cannot be rebound to another official result or launch')
        return _check_episode(root)
    run=_json(_contained(root,'run.json'))
    if not checkpoint.is_file():raise ValueError('Official checkpoint is missing')
    _copy_once(checkpoint,root/'official_result.json')
    route_source=filesystem_path(launch.get('official_routes_path',''))
    if not route_source.is_file():raise ValueError('Original pinned official XML is missing')
    _copy_once(route_source,root/'official_route.xml')
    if (root/'official_launch.json').exists() and _json(root/'official_launch.json')!=launch:
        raise ValueError('Original launch evidence changed')
    _write(root/'official_launch.json',launch)
    outcome=official_outcome(_json(root/'official_result.json'),launch.get('official_route_id',-1))
    report=dict(protocol=PROTOCOL,run_dir=str(root),memory_published=False,official_outcome=outcome,
        valid=False,passed=False,reasons=list(outcome['reasons']),evaluation_path=str(root/'official_demo_evaluation.json'))
    try:
        _identity(root,run,launch);guard=_guard(root);_binding(root,run,launch,guard)
        execution,by_frame=_execution(root);_video(root,execution)
        raw,_,_=_observations(root,by_frame,normalized=False)
        _counts(run,guard,execution,raw)
        if outcome['valid']:
            if not (root/'frames.jsonl').exists():_export(root,run,raw)
            _,frames,_=_observations(root,by_frame,normalized=True)
            report.update(valid=True,passed=True,capture_count=len(raw),execution_count=len(execution),
                anchors=select_anchors(frames,execution),scenario_family=run['scenario_family'],
                official_route_id=launch['official_route_id'],seed=launch['repetition_seed'])
    except (ValueError,KeyError,TypeError,OSError,ImportError) as exc:
        report['reasons'].append(str(exc))
    _write(root/'official_demo_evaluation.json',report)
    if report['valid']:
        sums={p.relative_to(root).as_posix():_hash(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='capture_manifest.json'}
        _write(root/'capture_manifest.json',dict(protocol=INPUT_PROTOCOL,demo_suite=SUITE,source='carla',
            source_version=run['source_version'],split=SPLIT,reference_driver=run['reference_driver'],frames_path='frames.jsonl',
            execution_path='execution.jsonl',demo_evaluation_path='official_demo_evaluation.json',source_hashes=sums))
        # Re-read frozen bytes exactly as the later local workstation will.
        return _check_episode(root)
    return report


def _check_episode(root,requested=None):
    run=_json(_contained(root,'run.json'));manifest=_json(_contained(root,'capture_manifest.json'))
    launch=_json(_contained(root,'official_launch.json'))
    route_id=_identity(root,run,launch);guard=_guard(root);_binding(root,run,launch,guard)
    outcome=official_outcome(_json(_contained(root,'official_result.json')),route_id)
    if not outcome['valid']:raise ValueError('Official demonstration failed: '+str(outcome['reasons']))
    if (manifest.get('protocol')!=INPUT_PROTOCOL or manifest.get('demo_suite')!=SUITE or manifest.get('source')!='carla'
            or manifest.get('split')!=SPLIT or manifest.get('source_version')!=run['source_version']
            or manifest.get('reference_driver')!=run['reference_driver'] or manifest.get('frames_path')!='frames.jsonl'):
        raise ValueError('Frozen official demonstration provenance mismatch')
    evaluation=_json(_contained(root,'official_demo_evaluation.json'))
    if evaluation.get('valid') is not True or evaluation.get('passed') is not True or evaluation.get('reasons')!=[]:
        raise ValueError('Demonstration was not successfully finalized')
    execution,by_frame=_execution(root);_video(root,execution)
    raw,frames,required=_observations(root,by_frame,normalized=True)
    _counts(run,guard,execution,raw)
    required.add('frames.jsonl')
    video=_json(root/'planning/video_recording.json');required.add('planning/'+video['path'])
    sums=manifest.get('source_hashes')
    if not isinstance(sums,dict) or not required.issubset(sums):raise ValueError('Incomplete frozen evidence checksums')
    for name,expected in sums.items():
        if not isinstance(expected,str) or len(expected)!=64 or _hash(_contained(root,name))!=expected:
            raise ValueError('Frozen official evidence changed: '+name)
    return dict(protocol=PROTOCOL,valid=True,passed=True,run_dir=str(root),episode=str(root),
        scenario_family=run['scenario_family'],official_route_id=route_id,seed=launch['repetition_seed'],
        driver=run['reference_driver'],capture_count=len(frames),execution_count=len(execution),
        anchors=select_anchors(frames,execution,requested),source_manifest_sha256=_hash(root/'capture_manifest.json'),
        source_hashes=sums,memory_published=False,official_outcome=outcome)


def check_official_demonstrations(captures,*,anchor_frames=None,require_all_families=False):
    roots=[]
    for directory in captures:
        path=filesystem_path(directory)
        if not path.is_dir():raise ValueError('Capture directory missing: '+str(path))
        roots.extend([path] if (path/'run.json').is_file() else sorted(p.parent for p in path.rglob('run.json')))
    if not roots or len(set(roots))!=len(roots):raise ValueError('No captures or overlapping capture selections')
    if anchor_frames and len(roots)!=1:raise ValueError('Explicit anchors require one capture')
    episodes=[];rejected=[]
    for root in roots:
        try:episodes.append(_check_episode(root,anchor_frames))
        except (ValueError,KeyError,TypeError,OSError) as exc:rejected.append(dict(episode=str(root),reason=str(exc)))
    families=sorted({episode['scenario_family'] for episode in episodes})
    if require_all_families and set(families)!=set(FAMILIES):
        rejected.append(dict(episode=None,reason='Expected actual official demonstrations for all 17 families'))
    return dict(protocol=PROTOCOL,demo_suite=SUITE,valid=bool(episodes) and not rejected,episodes=episodes,rejected=rejected,
        successful_episodes=len(episodes),scenario_families=families,all_17_families_present=set(families)==set(FAMILIES),
        memory_published=False,teacher_called=False,human_review_fabricated=False,
        data_use='targeted_seen_scenario_demonstration;not_unseen_route_generalization')


def prepare_official_demonstrations(captures,output,workbench_root,*,pipeline_config=None,dino_model=None,
                                    device='cpu',anchor_frames=None,require_all_families=False):
    checked=check_official_demonstrations(captures,anchor_frames=anchor_frames,require_all_families=require_all_families)
    if not checked['valid']:raise ValueError('Official demonstrations not eligible: '+str(checked['rejected']))
    output=filesystem_path(output);workbench=filesystem_path(workbench_root)
    if output.exists():raise ValueError('Use a new preparation output directory')
    for source in [workbench,*[filesystem_path(x['episode']) for x in checked['episodes']]]:
        if output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Output overlaps source/workbench')
    package=workbench/'src'/'offline_memory_builder'
    if not (package/'pipeline.py').is_file() or not (package/'storage.py').is_file():raise ValueError('Missing explicit existing workbench')
    if 'offline_memory_builder' in sys.modules and filesystem_path(sys.modules['offline_memory_builder'].__file__).parent!=package:
        raise ValueError('A different offline workbench is already imported')
    original_path=list(sys.path);old_bytecode=sys.dont_write_bytecode;old_mpl=os.environ.get('MPLCONFIGDIR')
    output.mkdir(parents=True);os.environ['MPLCONFIGDIR']=str(output/'.matplotlib');sys.dont_write_bytecode=True
    sys.path.insert(0,str(workbench/'src'))
    try:
        pipeline=importlib.import_module('offline_memory_builder.pipeline');storage=importlib.import_module('offline_memory_builder.storage')
        if 'selected_anchor_frame_ids' not in inspect.signature(pipeline.prepare).parameters:raise ValueError('Workbench lacks exact anchor API')
        cfg=deepcopy(storage.configuration(pipeline_config or workbench/'config'/'pipeline.yaml'))
        cfg['sampling'].update(interval_s=1.,start_s=0.,future_offsets_s=[0.,1.,2.,3.],time_tolerance_s=.001,max_per_clip=3)
        cfg['features'].update(dino_model=str(filesystem_path(dino_model)) if dino_model else None,device=device)
        results=[]
        for episode in checked['episodes']:
            source=filesystem_path(episode['episode']);ids=[a['frame_id'] for a in episode['anchors']]
            current=_check_episode(source,ids)
            if current['source_manifest_sha256']!=episode['source_manifest_sha256']:raise ValueError('Source changed before preparation')
            results.append(pipeline.prepare(cfg,source,output,model_path=cfg['features']['dino_model'],device=device,
                anchor_alignment='strict',selected_anchor_frame_ids=ids))
            if _hash(source/'capture_manifest.json')!=episode['source_manifest_sha256']:raise ValueError('Source changed during preparation')
            for name,digest in episode['source_hashes'].items():
                if _hash(_contained(source,name))!=digest:raise ValueError('Source asset changed during preparation: '+name)
        report=dict(checked,status='prepared_pending_real_teacher_and_human_review',preparations=results,published=False,reviewed=False)
        storage.atomic_json(output/'official_demo_preparation.json',report)
        return report
    finally:
        sys.path[:]=original_path;sys.dont_write_bytecode=old_bytecode
        if old_mpl is None:os.environ.pop('MPLCONFIGDIR',None)
        else:os.environ['MPLCONFIGDIR']=old_mpl
