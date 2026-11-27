"""Offline evidence gate for demonstrations in unchanged official scenarios.

This module never imports CARLA, drives actors, calls an annotation model,
approves text or publishes a library. Official outcomes validate the benchmark
result, not human driving quality; raw observations are never rewritten to hide
failures. Demonstrations still require review of the full observed interaction.
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


def _admission_policy(allow_min_speed_only=False):
    if type(allow_min_speed_only) is not bool:raise ValueError('Minimum-speed admission must be explicitly boolean')
    return dict(schema='official-demo-admission-policy-1.0',
        mode='allow_min_speed_only' if allow_min_speed_only else 'strict_perfect',
        official_scores_preserved=True,non_min_speed_infraction_free_required=True,human_review_required=True)


def _segment_admission_policy(allow_min_speed_only=False):
    _admission_policy(allow_min_speed_only)
    return dict(schema='official-demo-segment-admission-policy-1.0',
        mode='interaction_segment_allow_min_speed_only' if allow_min_speed_only else 'interaction_segment_strict',
        official_scores_preserved=True,whole_route_completion_required=False,
        non_min_speed_infraction_free_required=True,human_review_required=True,
        segment_completion_evidence_required=True,official_route_success_claimed=False)


def _capture_mode(run):
    mode=run.get('capture_mode','full_route')
    if mode not in ('full_route','interaction_segment'):
        raise ValueError('Unknown official demonstration capture mode')
    return mode


def segment_official_outcome(data, route_id, *, allow_min_speed_only=False):
    """Check unchanged official partial-route results, without claiming route success.

    Only a separate, source-bound segment-completion audit can admit a capture.
    A bare failed/partial official result is never sufficient.
    """
    profile=_segment_admission_policy(allow_min_speed_only)
    reasons=[]
    checkpoint=data.get('_checkpoint',{}) if isinstance(data,dict) else {}
    if not isinstance(checkpoint,dict):checkpoint={}
    records=checkpoint.get('records')
    record=records[0] if isinstance(records,list) and len(records)==1 and isinstance(records[0],dict) else {}
    global_record=checkpoint.get('global_record',{})
    if not isinstance(global_record,dict):global_record={}
    if (not isinstance(data,dict) or data.get('entry_status')!='Finished' or data.get('eligible') is not True
            or checkpoint.get('progress')!=[1,1]):
        reasons.append('official_evaluator_not_finished_or_ineligible')
    if type(route_id) is not int or record.get('route_id')!='RouteScenario_%s_rep0'%route_id:
        reasons.append('official_route_identity_mismatch')
    infractions=record.get('infractions');totals=global_record.get('infractions')
    minor=infractions.get('min_speed_infractions') if isinstance(infractions,dict) else None
    if (not isinstance(infractions,dict) or set(infractions)!=INFRACTIONS
            or any(not isinstance(v,list) for v in infractions.values())
            or any(v for k,v in infractions.items() if k!='min_speed_infractions')
            or not isinstance(minor,list) or minor and not allow_min_speed_only
            or type(record.get('num_infractions')) is not int
            or record['num_infractions']!=len(minor)):
        reasons.append('segment_non_min_speed_or_unknown_infraction_present')
    percentages=[]
    for message in minor or []:
        match=re.fullmatch(r"Average speed is ([0-9]+(?:\.[0-9]{1,2})?)% of the surrounding traffic's one",message) if isinstance(message,str) else None
        value=float(match.group(1)) if match else None
        if value is None or not 0<=value<100:
            reasons.append('unrecognized_min_speed_penalty_evidence');break
        percentages.append(value)
    penalty=math.prod(1.-.3*(1.-v/100.) for v in percentages)
    scores=record.get('scores',{});means=global_record.get('scores_mean')
    if not isinstance(scores,dict):scores={}
    route=scores.get('score_route')
    if (not _number(route) or not 0<route<=100
            or any(not _number(scores.get(k)) for k in ('score_penalty','score_composed'))
            or not _same(scores.get('score_penalty'),round(penalty,6),1e-6)
            or not _same(scores.get('score_composed'),round((route or 0)*penalty,6),1e-6)
            or not isinstance(means,dict) or any(not _same(means.get(k),scores.get(k),1e-6)
                for k in ('score_route','score_penalty','score_composed'))):
        reasons.append('segment_original_scores_inconsistent')
    expected_status=('Completed' if minor else 'Perfect') if _number(route) and route>=100 else 'Failed'
    if record.get('status')!=expected_status or global_record.get('status')!=expected_status:
        reasons.append('segment_official_status_has_unknown_failure_or_error')
    length=record.get('meta',{}).get('route_length') if isinstance(record.get('meta'),dict) else None
    if (not isinstance(totals,dict) or set(totals)!=INFRACTIONS
            or any(not _number(v) or v<0 for v in totals.values())
            or any(v!=0 for k,v in totals.items() if k!='min_speed_infractions')
            or not _number(length) or length<=0 or not _number(route)):
        reasons.append('segment_global_infractions_or_route_metadata_invalid')
    elif not _same(totals['min_speed_infractions'],round(len(minor or []) / max(length/1000.*route/100.,.001),3),1e-6):
        reasons.append('segment_original_min_speed_rate_inconsistent')
    return dict(valid=not reasons,passed=not reasons,reasons=reasons,entry_status=data.get('entry_status') if isinstance(data,dict) else None,
        status=record.get('status'),scores=scores,infractions=infractions,global_infractions=totals,official_route_id=route_id,
        admission_policy=profile,accepted_with_min_speed_penalty=bool(minor and not reasons),
        whole_route_completed=bool(_number(route) and route>=100),official_route_success_claimed=False,
        scope='captured_interaction_segment',human_driving_quality_review_required=True)


def official_outcome(data, route_id, *, allow_min_speed_only=False):
    """Validate the complete upstream single-route checkpoint, including every infraction."""
    reasons=[]
    if not isinstance(data,dict):data={}
    checkpoint=data.get('_checkpoint',{})
    if not isinstance(checkpoint,dict):checkpoint={}
    records=checkpoint.get('records')
    record=records[0] if isinstance(records,list) and len(records)==1 and isinstance(records[0],dict) else {}
    global_record=checkpoint.get('global_record',{})
    if not isinstance(global_record,dict):global_record={}
    if data.get('entry_status')!='Finished' or data.get('eligible') is not True or checkpoint.get('progress')!=[1,1]:
        reasons.append('official_evaluator_not_finished_or_ineligible')
    if type(route_id) is not int or record.get('route_id')!='RouteScenario_%s_rep0'%route_id:
        reasons.append('official_route_identity_mismatch')
    profile=_admission_policy(allow_min_speed_only)
    infractions=record.get('infractions')
    minor=(infractions.get('min_speed_infractions') if isinstance(infractions,dict) else None)
    relaxed=allow_min_speed_only and isinstance(minor,list) and bool(minor)
    if not relaxed and (record.get('status')!='Perfect' or global_record.get('status')!='Perfect'):
        reasons.append('official_route_not_perfect')
    scores=record.get('scores',{})
    if not isinstance(scores,dict):scores={}
    if not relaxed and (not _same(scores.get('score_route'),100.) or not _same(scores.get('score_penalty'),1.)
            or not _same(scores.get('score_composed'),100.)):
        reasons.append('official_route_incomplete_or_penalized')
    if not relaxed and (not isinstance(infractions,dict) or not INFRACTIONS.issubset(infractions)
            or any(not isinstance(value,list) or value for value in infractions.values())
            or record.get('num_infractions')!=0):
        reasons.append('official_infractions_nonempty_or_unknown')
    totals=global_record.get('infractions')
    if not relaxed and (not isinstance(totals,dict) or not INFRACTIONS.issubset(totals)
            or any(not _number(value) or value!=0 for value in totals.values())):
        reasons.append('official_global_infractions_nonzero_or_unknown')
    if relaxed:
        if (record.get('status')!='Completed'
                or global_record.get('status')!=record.get('status')):
            reasons.append('min_speed_only_requires_completed_consistent_status')
        if (set(infractions)!=INFRACTIONS or any(not isinstance(v,list) for v in infractions.values())
                or any(v for k,v in infractions.items() if k!='min_speed_infractions')
                or type(record.get('num_infractions')) is not int or record['num_infractions']!=len(minor)):
            reasons.append('non_min_speed_or_unknown_infraction_present')
        percentages=[]
        for message in minor:
            match=re.fullmatch(r"Average speed is ([0-9]+(?:\.[0-9]{1,2})?)% of the surrounding traffic's one",message) if isinstance(message,str) else None
            value=float(match.group(1)) if match else None
            if value is None or not 0<=value<100:
                reasons.append('unrecognized_min_speed_penalty_evidence');break
            percentages.append(value)
        penalty=math.prod(1.-(1.-.7)*(1.-v/100.) for v in percentages)
        if (len(percentages)!=len(minor) or any(not _number(scores.get(k)) for k in ('score_route','score_penalty','score_composed'))
                or abs(scores.get('score_route',0)-100.)>1e-6
                or abs(scores.get('score_penalty',0)-round(penalty,6))>1e-6
                or abs(scores.get('score_composed',0)-round(100.*penalty,6))>1e-6):
            reasons.append('scores_do_not_match_only_official_min_speed_penalties')
        if (not isinstance(totals,dict) or set(totals)!=INFRACTIONS
                or any(not _number(v) or v<0 for v in totals.values())
                or any(v!=0 for k,v in totals.items() if k!='min_speed_infractions')):
            reasons.append('official_global_non_min_speed_or_unknown_infraction')
        else:
            meta=record.get('meta',{});length=meta.get('route_length') if isinstance(meta,dict) else None
            if (not _number(length) or length<=0
                    or abs(totals['min_speed_infractions']-round(len(minor)/max(length/1000.,.001),3))>1e-6):
                reasons.append('global_min_speed_rate_does_not_match_actual_route')
        means=global_record.get('scores_mean')
        if (not isinstance(means,dict) or any(not _number(means.get(k))
                or not _number(scores.get(k)) or abs(means[k]-scores[k])>1e-6
                for k in ('score_route','score_penalty','score_composed'))):
            reasons.append('official_global_scores_disagree_with_single_route')
    return dict(valid=not reasons,passed=not reasons,reasons=reasons,
        entry_status=data.get('entry_status'),status=record.get('status'),scores=scores,
        infractions=infractions,global_infractions=totals,official_route_id=route_id,
        admission_policy=profile,accepted_with_min_speed_penalty=bool(relaxed and not reasons),
        human_driving_quality_review_required=True)


def _require_complete_capture(run):
    if run.get('complete') is not True or run.get('errors')!=[]:
        errors=run.get('errors')
        first=errors[0] if isinstance(errors,list) and errors and isinstance(errors[0],dict) else {}
        detail=('%s: %s'%(first.get('type','capture_error'),first.get('error','closing metadata is incomplete')))[:1200]
        raise ValueError('Demonstration capture did not close successfully; '+detail)


def _identity(root, run, launch):
    _require_complete_capture(run)
    _capture_mode(run)
    if (run.get('demo_suite')!=SUITE or run.get('source')!='carla' or run.get('split')!=SPLIT
            or run.get('test_only') is not False or run.get('official_evaluation') is not True
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
    if (spec.get('collection_split_sha256')!=launch.get('collection_split_sha256')
            or spec.get('collection_fallback')!=launch.get('collection_fallback')):
        raise ValueError('Collection partition or reserve authorization differs from the fingerprinted launch')
    if (spec.get('scenario_family')!=run['scenario_family'] or spec.get('official_route_id')!=run['official_route_id']
            or spec.get('driver')!='official_autopilot' or spec.get('purpose')!='demonstration_only'
            or spec.get('scenario_mutations') is not False):
        raise ValueError('Recorded expert configuration differs from its official scenario')
    if spec.get('capture_mode','full_route')!=_capture_mode(run):
        raise ValueError('Recorded capture mode differs from its fingerprinted official launch')
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
    from .camera_calibration import global_camera_calibration
    return global_camera_calibration(_json(_contained(root,row['camera_manifest'])),row['camera_manifest'])


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


def _physical_world(root,run,execution,raw):
    """Verify an explicitly declared grounding audit; never invent legacy data."""
    declaration=run.get('physical_world_audit')
    if declaration is None:
        return dict(declared=False,verified=False,scope='legacy_capture_without_grounding_audit')
    from .source_world import SCHEMA
    if declaration!=SCHEMA:
        raise ValueError('Unsupported declared physical-world audit')
    rows=_rows(_contained(root,'physical_world_audit.jsonl'))
    if len(rows)!=len(execution):
        raise ValueError('Physical-world audit does not cover every executed source frame')
    observed={_frame(row):row for row in raw}
    expected_limits=dict(below_route_m=3.,descending_below_route_m=1.,downward_speed_mps=5.)
    for row,step in zip(rows,execution):
        if (row.get('schema')!=SCHEMA or _frame(row)!=step['frame'] or not _same(row.get('time'),step['time'])
                or row.get('valid') is not True or row.get('classification')!='physical_source_valid'
                or row.get('reason')!='within_gross_grounding_limits'
                or row.get('limits')!=expected_limits
                or any(row.get(k) is not False for k in ('source_height_corrected','world_modified','driving_control_generated'))):
            raise ValueError('Physical-world audit is invalid or mismatched to the actual execution source')
        for key in ('position_xyz_carla_m','velocity_xyz_carla_mps','acceleration_xyz_carla_mps2'):
            vector=row.get(key)
            if not isinstance(vector,list) or len(vector)!=3 or not all(map(_number,vector)):
                raise ValueError('Physical-world audit lacks actual finite XYZ measurements')
        xyz=row['position_xyz_carla_m'];velocity=row['velocity_xyz_carla_mps']
        road=row.get('public_route_height_m');residual=row.get('actor_origin_minus_route_height_m')
        if (not _number(road) or not _same(residual,xyz[2]-road)
                or residual < -3. or residual < -1. and velocity[2] < -5.
                or not _same(row.get('speed_xy_mps'),math.hypot(*velocity[:2]))
                or not _same(row.get('speed_xyz_mps'),math.sqrt(sum(v*v for v in velocity)))):
            raise ValueError('Physical-world audit contradicts its recorded grounding measurements')
        if row['frame'] in observed:
            pose=observed[row['frame']]['scene']['diagnostics'].get('frozen_actor_origin_pose',{})
            origin=pose.get('position_xyz_carla')
            if (not isinstance(origin,list) or len(origin)!=3
                    or any(not _same(a,b) for a,b in zip(xyz,origin))):
                raise ValueError('Physical-world audit and raw image source have different frozen actor origins')
    return dict(declared=True,verified=True,schema=SCHEMA,frames=len(rows),
        scope='gross_grounding_source_validity_only;not_a_driving_safety_certificate')


def _behavior(root,run,execution):
    """Independent sampled process evidence, not an official-score substitute."""
    from .demo_behavior import SCHEMA,POLICY,validate_rows,interaction_windows
    declaration=run.get('demonstration_behavior_audit')
    if declaration is None:
        return dict(declared=False,verified=False,human_review_required=True,
            automatic_human_quality_approval=False,interaction_windows=[],
            scope='legacy_capture_without_new_behavior_review_evidence'),None
    if (declaration!=SCHEMA or run.get('demonstration_quality_policy')!=POLICY
            or run.get('human_driving_quality_review_required') is not True):
        raise ValueError('Declared demonstration behavior policy is missing or unsupported')
    rows=_rows(_contained(root,'demonstration_behavior_audit.jsonl'))
    physical=_rows(_contained(root,'physical_world_audit.jsonl'))
    if len(physical)!=len(execution):raise ValueError('Behavior audit lacks the complete physical source binding')
    report=validate_rows(rows,execution,physical)
    windows=interaction_windows(rows)
    for window in windows:
        review_start=max(execution[0]['frame'],window['start_frame']-40)
        review_end=min(execution[-1]['frame'],window['end_frame']+60)
        window.update(review_start_frame=review_start,review_end_frame=review_end,
            video_first_frame_index=review_start-execution[0]['frame'],
            video_last_frame_index=review_end-execution[0]['frame'],
            approach_context_available_s=(window['start_frame']-review_start)*.05,
            recovery_context_available_s=(review_end-window['end_frame'])*.05)
    video=_json(_contained(root,'planning/video_recording.json'))
    report.update(interaction_windows=windows,
        full_video=dict(path='planning/'+video['path'],fps=20.,
            first_source_frame=execution[0]['frame'],last_source_frame=execution[-1]['frame'],
            first_source_time=execution[0]['time'],last_source_time=execution[-1]['time']),
        review_required_topics=['actual_target_interaction_present_and_completed',
            'pedestrian_and_vehicle_clearance','appropriate_local_approach_speed',
            'smooth_bypass_wait_and_recovery',('all_flagged_intervals_and_full_captured_segment_video'
                if _capture_mode(run)=='interaction_segment' else 'all_flagged_intervals_and_full_route_video')],
        observed_proximity_does_not_identify_official_target=True)
    return report,rows


def _observations(root, execution_by_frame, *, normalized, admission_outcome=None):
    run=_json(_contained(root,'run.json'))
    requested_segment=_capture_mode(run)=='interaction_segment'
    if admission_outcome is None and normalized and (root/'official_demo_evaluation.json').is_file():
        admission_outcome=_json(root/'official_demo_evaluation.json').get('admission_outcome')
    segment=requested_segment and (admission_outcome is None or admission_outcome.get('scope')=='captured_interaction_segment')
    raw=_rows(_contained(root,'raw_frames.jsonl'))
    frames=_rows(_contained(root,'frames.jsonl')) if normalized else None
    if frames is not None and len(frames)!=len(raw):
        raise ValueError('Normalized and raw observation counts differ')
    required={'run.json','configuration.json','carla_identity.json','raw_frames.jsonl','execution.jsonl','expert_world_guard.json','official_route.xml',
              'official_result.json','official_launch.json','official_demo_evaluation.json','planning/video_recording.json'}
    if _json(_contained(root,'run.json')).get('physical_world_audit') is not None:
        required.add('physical_world_audit.jsonl')
    if _json(_contained(root,'run.json')).get('demonstration_behavior_audit') is not None:
        required.add('demonstration_behavior_audit.jsonl')
    if requested_segment:
        required.update(('segment_capture.json','segment_ticks.jsonl'))
        if _json(_contained(root,'configuration.json')).get('official_demo',{}).get('segment',{}).get('variant')=='Base':
            required.add('paired_generalization_route.xml')
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
            if (audit.get('source')!=('official_segment_outcome' if segment else 'official_episode_outcome')
                    or audit.get('scope')!=('captured_interaction_segment' if segment else 'whole_episode')
                    or any(audit.get(k) is not False for k in ('collision','offroad','rule_violation'))):
                raise ValueError('Normalized driving outcome is not attributed to its actual audited scope')
            if segment and audit.get('official_route_success_claimed') is not False:
                raise ValueError('Partial capture cannot claim whole-route official success')
        prior=frame,now
    if len(raw)<4:
        raise ValueError('Missing actual current/+1/+2/+3-second source evidence')
    return raw,frames,required


def _borrow_anchors(eligible,execution,*,earliest_entry=-math.inf):
    """Map actual opposing-lane decisions to their latest available 1Hz observation."""
    maneuvers={}
    for step in execution:
        decision=step.get('expert_decision') or {}
        policy=decision.get('maneuver_policy') or {}
        update=policy.get('return_update')
        if not policy.get('maneuvers') and update is None:continue
        frame,time=step['frame'],step['time']
        if decision.get('source_frame')!=frame or not _same(decision.get('source_time'),time):
            raise ValueError('Borrow decision does not match its executed source frame')
        for row in policy.get('maneuvers',[]):
            key=(row['kind'],row['from_index'])
            events=maneuvers.setdefault(key,{})
            if (row.get('current_preentry_vehicle_gap_clear') is False
                    and row.get('maneuver_committed') is False
                    and math.hypot(*step['observed_ego']['velocity'])<.15):
                reason='stopped_without_borrow_clearance'
                stopped=events.setdefault(reason,dict(frame=frame,time=time))
                blocker=policy.get('entry_gap_blocker')
                if blocker and 'blocker_actor_id' not in stopped:
                    if (type(blocker.get('actor_id')) is not int or blocker.get('reason')
                            not in ('vehicle_between_ego_and_return','arrival_gap_too_small','passing_body_not_clear')):
                        raise ValueError('Invalid current vehicle evidence for denied borrowing clearance')
                    # Prefer the first actual stopped frame with an identified
                    # vehicle over an earlier denial without an attributed cause.
                    events[reason]=dict(frame=frame,time=time,blocker_actor_id=blocker['actor_id'],
                                        blocker_reason=blocker['reason'])
            entry=row.get('physical_entry_evidence')
            if entry and entry.get('source_frame')==frame and row.get('maneuver_committed') is True:
                if not _same(entry.get('source_time'),time):
                    raise ValueError('Physical opposing-lane entry has inconsistent source time')
                events.setdefault('physical_borrow_entry',dict(frame=frame,time=time,
                    original_lane_id=entry['original_lane_id'],borrowed_side=entry['borrowed_side']))
        if update is not None:
            if (update.get('source_frame')!=frame or not _same(update.get('source_time'),time)
                    or type(update.get('oncoming_actor_id')) is not int
                    or not all(type(update.get(k)) is int for k in ('from_index','updated_return_index','previous_return_index'))
                    or not update['from_index']<update['updated_return_index']<update['previous_return_index']
                    or not _number(update.get('advancement_m')) or update['advancement_m']<=0):
                raise ValueError('Earlier return lacks a source-bound oncoming actor and shortened reference')
            key=(update['kind'],update['from_index'])
            if 'physical_borrow_entry' not in maneuvers.get(key,{}):
                raise ValueError('Earlier return decision has no observed entry into the borrowed lane')
            maneuvers[key].setdefault('earlier_return_decision',dict(frame=frame,time=time,
                oncoming_actor_id=update['oncoming_actor_id'],advancement_m=update['advancement_m']))
    # Keep one coherent maneuver per capture even if multiple objects are passed.
    choices=[(key,events) for key,events in maneuvers.items()
             if 'physical_borrow_entry' in events and events['physical_borrow_entry']['time']>=earliest_entry]
    if not choices:return []
    (kind,start),events=max(choices,key=lambda item:(len(item[1]),'earlier_return_decision' in item[1]))
    anchors={}
    for reason,observed in sorted(events.items(),key=lambda item:item[1]['time']):
        candidates=[r for r in eligible if -1e-5<=observed['time']-r['time']<1.+1e-5]
        if not candidates:continue
        source=max(candidates,key=lambda r:r['time']);identity=source['snapshot']['frame_id']
        anchor=anchors.setdefault(identity,dict(frame_id=identity,time=source['time'],anchor_reason=reason,
            maneuver_kind=kind,maneuver_from_index=start,observed_decisions=[],
            target_interaction_requires_human_confirmation=True))
        anchor['observed_decisions'].append(dict(phase=reason,**observed))
    return list(anchors.values())


def select_anchors(frames,execution,requested=None,behavior_rows=None):
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
    borrowing=_borrow_anchors(eligible,execution)
    if borrowing:return borrowing
    if behavior_rows:
        from .demo_behavior import interaction_windows
        windows=interaction_windows(behavior_rows)
        if windows:
            # Real observed neighboring bodies give interaction context. A first
            # unrelated road bend must not consume the sole turning anchor.
            primary=min(windows,key=lambda w:(-len(w['review_flags']),w['minimum_clearance_m'],w['start_time']))
            low,high=primary['start_time'],primary['end_time']
            during=[r for r in execution if low<=r['time']<=high]
            strongest=max(during,key=lambda r:(r['returned_control']['brake']+abs(r['returned_control']['steer'])))
            selected=[];used=set()
            for reason,target,start,end in (('observed_interaction_approach',low-1.,low-2.,low),
                    ('observed_interaction_brake_or_bypass',strongest['time'],low,high),
                    ('observed_interaction_recovery',high+1.,high+.001,high+3.)):
                candidates=[r for r in eligible if start<=r['time']<=end
                    and r['snapshot']['frame_id'] not in used]
                if not candidates:continue
                row=min(candidates,key=lambda r:(abs(r['time']-target),r['time']))
                used.add(row['snapshot']['frame_id'])
                selected.append(dict(frame_id=row['snapshot']['frame_id'],time=row['time'],anchor_reason=reason,
                    observed_actor_ids=primary['actor_ids'],interaction_start_time=low,interaction_end_time=high,
                    target_interaction_requires_human_confirmation=True))
            if selected:return selected
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


def _export(root,run,raw,official=None):
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
            segment=_capture_mode(run)=='interaction_segment' and (official or {}).get('scope')=='captured_interaction_segment'
            audit=dict(source='official_segment_outcome' if segment else 'official_episode_outcome',
                scope='captured_interaction_segment' if segment else 'whole_episode',collision=False,offroad=False,rule_violation=False,
                per_frame_independent_safety_claimed=False,
                rule_violation_scope='non_min_speed_traffic_safety_items_only',
                official_infraction_free=not bool((official or {}).get('accepted_with_min_speed_penalty')),
                benchmark_min_speed_penalty=bool((official or {}).get('accepted_with_min_speed_penalty')),
                benchmark_min_speed_infraction_count=len((official or {}).get('infractions',{}).get('min_speed_infractions',[])))
            if segment:audit['official_route_success_claimed']=False
            stream.write(json.dumps(dict(index=index,frame=row['frame'],time=row['time'],snapshot=snapshot,
                native_image=row['native_image'],camera_manifest=row['camera_manifest'],audit=audit),ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(destination)


def _segment_source_binding(root,run,launch,*,freeze=False):
    """Rebuild target geometry from frozen original bytes after server transfer."""
    from .demo_segment import segment_targets
    spec=launch.get('config',{}).get('official_demo',{}).get('segment')
    if not isinstance(spec,dict):raise ValueError('Missing fingerprinted interaction segment specification')
    original=_contained(root,'official_route.xml')
    variant=spec.get('variant');rid=run['official_route_id'];family=run['scenario_family']
    if (variant not in ('Base','Generalization') or spec.get('route_id')!=rid or spec.get('family')!=family
            or spec.get('map')!=run['world_map'].split('/')[-1]
            or spec.get('original_xml_sha256')!=_hash(original)):
        raise ValueError('Segment target does not match original route/map/family bytes')
    if variant=='Base':
        expected='Generalization_%s_%04d.xml'%(family,rid+1000)
        paired=root/'paired_generalization_route.xml'
        if freeze:
            source=filesystem_path(launch['official_routes_path']).with_name(expected)
            if not source.is_file() or _hash(source)!=spec.get('paired_generalization_xml_sha256'):
                raise ValueError('Paired Generalization source is missing or changed')
            _copy_once(source,paired)
        paired=_contained(root,'paired_generalization_route.xml');paired_id=rid+1000
    else:
        paired=original;paired_id=rid
    if spec.get('paired_generalization_route_id')!=paired_id or _hash(paired)!=spec.get('paired_generalization_xml_sha256'):
        raise ValueError('Frozen paired Generalization route bytes changed')
    def scenarios(path,identity):
        routes=ET.parse(path).getroot().findall('route')
        if len(routes)!=1 or routes[0].get('id')!=str(identity) or routes[0].get('town')!=spec['map']:
            raise ValueError('Frozen segment XML route/map identity changed')
        return segment_targets(routes[0])
    if (spec.get('expected_scenarios')!=scenarios(original,rid)
            or spec.get('paired_gen_scenarios')!=scenarios(paired,paired_id)):
        raise ValueError('Segment target geometry does not reproduce the frozen original XML')
    return spec


def _segment_evidence(root,run,launch,execution,raw,*,natural_full_route=False):
    from .demo_segment import validate_segment_evidence
    if (run.get('segment_capture_path')!='segment_capture.json'
            or run.get('physical_world_audit') is None or run.get('demonstration_behavior_audit') is None):
        raise ValueError('Interaction segment requires declared full physical and behavior evidence')
    summary=_json(_contained(root,'segment_capture.json'))
    spec=_segment_source_binding(root,run,launch)
    if summary.get('spec')!=spec:
        raise ValueError('Segment specification differs from its frozen launch')
    ticks=_rows(_contained(root,'segment_ticks.jsonl'))
    result=validate_segment_evidence(summary,ticks,execution=execution,source_times=[row['time'] for row in raw],
        **({'allow_natural_full_route':True} if natural_full_route else {}))
    if not isinstance(result,dict) or result.get('valid') is not True or result.get('passed') is False:
        raise ValueError('Interaction segment did not pass source-bound completion audit: '+str(result))
    return summary


def _capture_outcomes(root,run,launch,*,allow_min_speed_only=False):
    data=_json(_contained(root,'official_result.json'))
    official=official_outcome(data,launch.get('official_route_id',-1),allow_min_speed_only=allow_min_speed_only)
    # A short route can naturally end before the five-second recovery tail.
    # Its genuine complete official result remains valid full-route evidence;
    # do not fabricate a segment-completed state or alter the original run.
    admitted=(segment_official_outcome(data,launch.get('official_route_id',-1),allow_min_speed_only=allow_min_speed_only)
              if _capture_mode(run)=='interaction_segment' and not official['valid'] else official)
    return official,admitted


def _anchors_for_capture(frames,execution,requested=None,behavior_rows=None,segment=None):
    if segment is None:return select_anchors(frames,execution,requested,behavior_rows)
    spec=segment.get('spec',{})
    ordinary=(spec.get('variant')=='Base' and spec.get('expected_scenarios')==[])
    if requested:
        selected=select_anchors(frames,execution,requested)
        if ordinary:
            for item in selected:item['capture_context']='paired_region_ordinary_driving'
        return selected
    eligible=[row for row in frames if all(any(_same(other['time'],row['time']+offset)
        for other in frames) for offset in (0.,1.,2.,3.))]
    events=segment.get('events',[])
    onset=next((e.get('time') for e in events if e.get('phase')=='interaction'),None)
    finish=segment.get('interaction_completed_time')
    if onset is None:onset=next((e.get('time') for e in events if e.get('phase')=='recovery'),None)
    if not _number(onset) or not _number(finish) or finish<onset:
        raise ValueError('Segment anchors lack source-bound interaction/recovery times')
    if not ordinary:
        borrowing=_borrow_anchors(eligible,[r for r in execution if r['time']<=finish],earliest_entry=onset)
        if borrowing:return borrowing
        for maneuver in (segment.get('physical_bypass') or {}).values():
            if maneuver.get('status')!='returned_to_route_lane':continue
            alongside=next((r for r in execution if r['frame']==maneuver['alongside_frame']),None)
            if alongside is not None:
                prior=[r for r in eligible if -1e-5<=alongside['time']-r['time']<1.+1e-5]
                if prior:
                    source=max(prior,key=lambda r:r['time'])
                    return [dict(frame_id=source['snapshot']['frame_id'],time=source['time'],
                        anchor_reason='observed_borrowed_lane_passage',target_actor_id=maneuver['passed_actor_id'],
                        actual_alongside_frame=maneuver['alongside_frame'],actual_return_frame=maneuver['return_frame'],
                        target_interaction_requires_human_confirmation=True)]
    during=[r for r in execution if onset<=r['time']<=finish]
    if not during:raise ValueError('No actual controls in captured interaction')
    strongest=max(during,key=lambda r:r['returned_control']['brake']+abs(r['returned_control']['steer']))
    # One difficult decision state is the default memory key. Extra anchors
    # remain an explicit human selection, rather than filling the library with
    # easy post-interaction recovery states.
    targets=[('interaction_decision_context',onset-1.),
             ('interaction_brake_or_bypass_context',max(onset,strongest['time']-1.))]
    selected=[];used=set()
    for reason,target in targets:
        candidates=[r for r in eligible if abs(r['time']-target)<=1.+1e-5]
        if not candidates:continue
        row=min(candidates,key=lambda r:(abs(r['time']-target),r['time']))
        identity=row['snapshot']['frame_id']
        if identity not in used:
            selected.append(dict(frame_id=identity,time=row['time'],anchor_reason=reason))
            if ordinary:
                selected[-1].update(anchor_reason='paired_region_ordinary_driving',capture_context='paired_region_ordinary_driving')
            break
    if not selected:raise ValueError('No interaction anchor with complete actual three-second future evidence')
    return selected


def finalize_official_capture(run_dir, official_checkpoint, launch_metadata,*,allow_min_speed_only=False):
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
        return _check_episode(root,allow_min_speed_only=allow_min_speed_only)
    run=_json(_contained(root,'run.json'))
    if not checkpoint.is_file():raise ValueError('Official checkpoint is missing')
    _copy_once(checkpoint,root/'official_result.json')
    route_source=filesystem_path(launch.get('official_routes_path',''))
    if not route_source.is_file():raise ValueError('Original pinned official XML is missing')
    _copy_once(route_source,root/'official_route.xml')
    if (root/'official_launch.json').exists() and _json(root/'official_launch.json')!=launch:
        raise ValueError('Original launch evidence changed')
    _write(root/'official_launch.json',launch)
    official,outcome=_capture_outcomes(root,run,launch,allow_min_speed_only=allow_min_speed_only)
    report=dict(protocol=PROTOCOL,run_dir=str(root),memory_published=False,official_outcome=official,
        valid=False,passed=False,reasons=list(outcome['reasons']),evaluation_path=str(root/'official_demo_evaluation.json'),
        capture_mode=_capture_mode(run),actual_admission_scope=outcome.get('scope','whole_episode'),admission_outcome=outcome,
        admission_policy=deepcopy(outcome['admission_policy']),
        accepted_with_min_speed_penalty=outcome['accepted_with_min_speed_penalty'])
    try:
        _identity(root,run,launch);guard=_guard(root);_binding(root,run,launch,guard)
        if _capture_mode(run)=='interaction_segment':_segment_source_binding(root,run,launch,freeze=True)
        execution,by_frame=_execution(root);_video(root,execution)
        raw,_,_=_observations(root,by_frame,normalized=False)
        _counts(run,guard,execution,raw)
        physical=_physical_world(root,run,execution,raw)
        behavior,behavior_rows=_behavior(root,run,execution)
        segment=(_segment_evidence(root,run,launch,execution,raw,natural_full_route=official['valid'])
                 if _capture_mode(run)=='interaction_segment' else None)
        if outcome['valid']:
            if not (root/'frames.jsonl').exists():_export(root,run,raw,outcome)
            _,frames,_=_observations(root,by_frame,normalized=True,admission_outcome=outcome)
            report.update(valid=True,passed=True,capture_count=len(raw),execution_count=len(execution),
                physical_world_audit=physical,
                demonstration_behavior_audit=behavior,
                anchors=_anchors_for_capture(frames,execution,behavior_rows=behavior_rows,segment=segment),scenario_family=run['scenario_family'],
                official_route_id=launch['official_route_id'],seed=launch['repetition_seed'])
            if segment is not None:report.update(segment_capture=segment,official_route_success_claimed=False)
    except (ValueError,KeyError,TypeError,OSError,ImportError) as exc:
        report['reasons'].append(str(exc))
    _write(root/'official_demo_evaluation.json',report)
    if report['valid']:
        sums={p.relative_to(root).as_posix():_hash(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='capture_manifest.json'}
        _write(root/'capture_manifest.json',dict(protocol=INPUT_PROTOCOL,demo_suite=SUITE,source='carla',
            source_version=run['source_version'],split=SPLIT,reference_driver=run['reference_driver'],frames_path='frames.jsonl',
            execution_path='execution.jsonl',demo_evaluation_path='official_demo_evaluation.json',source_hashes=sums,
            admission_policy=deepcopy(outcome['admission_policy']),
            accepted_with_min_speed_penalty=outcome['accepted_with_min_speed_penalty']))
        # Re-read frozen bytes exactly as the later local workstation will.
        return _check_episode(root,allow_min_speed_only=allow_min_speed_only)
    return report


def _check_episode(root,requested=None,*,allow_min_speed_only=False):
    _admission_policy(allow_min_speed_only)
    run=_json(_contained(root,'run.json'))
    _require_complete_capture(run)
    if not (root/'capture_manifest.json').is_file():
        raise ValueError('Successful Agent capture is not finalized; run finalize to create capture_manifest.json')
    manifest=_json(_contained(root,'capture_manifest.json'))
    launch=_json(_contained(root,'official_launch.json'))
    route_id=_identity(root,run,launch);guard=_guard(root);_binding(root,run,launch,guard)
    frozen_policy=manifest.get('admission_policy',_admission_policy(False))
    if frozen_policy not in (_admission_policy(False),_admission_policy(True),
                             _segment_admission_policy(False),_segment_admission_policy(True)):
        raise ValueError('Frozen official admission policy is missing or unsupported')
    official,outcome=_capture_outcomes(root,run,launch,
        allow_min_speed_only=frozen_policy['mode'] in ('allow_min_speed_only','interaction_segment_allow_min_speed_only'))
    if outcome['admission_policy']!=frozen_policy:
        raise ValueError('Frozen admission policy differs from its source capture mode')
    if outcome['accepted_with_min_speed_penalty'] and not allow_min_speed_only:
        raise ValueError('Frozen source has an original minimum-speed penalty; explicitly pass --allow-min-speed-only')
    if not outcome['valid']:raise ValueError('Official demonstration failed: '+str(outcome['reasons']))
    if (manifest.get('protocol')!=INPUT_PROTOCOL or manifest.get('demo_suite')!=SUITE or manifest.get('source')!='carla'
            or manifest.get('split')!=SPLIT or manifest.get('source_version')!=run['source_version']
            or manifest.get('reference_driver')!=run['reference_driver'] or manifest.get('frames_path')!='frames.jsonl'):
        raise ValueError('Frozen official demonstration provenance mismatch')
    evaluation=_json(_contained(root,'official_demo_evaluation.json'))
    if 'admission_policy' in manifest and (evaluation.get('admission_policy')!=frozen_policy
            or evaluation.get('accepted_with_min_speed_penalty')!=outcome['accepted_with_min_speed_penalty']
            or manifest.get('accepted_with_min_speed_penalty')!=outcome['accepted_with_min_speed_penalty']):
        raise ValueError('Frozen admission policy or original minimum-speed outcome was changed')
    if evaluation.get('valid') is not True or evaluation.get('passed') is not True or evaluation.get('reasons')!=[]:
        raise ValueError('Demonstration was not successfully finalized')
    execution,by_frame=_execution(root);_video(root,execution)
    raw,frames,required=_observations(root,by_frame,normalized=True,admission_outcome=outcome)
    _counts(run,guard,execution,raw)
    physical=_physical_world(root,run,execution,raw)
    behavior,behavior_rows=_behavior(root,run,execution)
    segment=(_segment_evidence(root,run,launch,execution,raw,natural_full_route=official['valid'])
             if _capture_mode(run)=='interaction_segment' else None)
    if segment is not None and (evaluation.get('segment_capture')!=segment
            or evaluation.get('official_route_success_claimed') is not False
            or evaluation.get('official_outcome')!=official or evaluation.get('admission_outcome')!=outcome):
        raise ValueError('Frozen segment completion or original official outcome changed')
    required.add('frames.jsonl')
    video=_json(root/'planning/video_recording.json');required.add('planning/'+video['path'])
    sums=manifest.get('source_hashes')
    if not isinstance(sums,dict) or not required.issubset(sums):raise ValueError('Incomplete frozen evidence checksums')
    for name,expected in sums.items():
        if not isinstance(expected,str) or len(expected)!=64 or _hash(_contained(root,name))!=expected:
            raise ValueError('Frozen official evidence changed: '+name)
    return dict(protocol=PROTOCOL,valid=True,passed=True,run_dir=str(root),episode=str(root),
        scenario_family=run['scenario_family'],official_route_id=route_id,seed=launch['repetition_seed'],
        capture_collection_split_sha256=launch.get('collection_split_sha256'),
        collection_fallback=deepcopy(launch.get('collection_fallback')),
        driver=run['reference_driver'],capture_count=len(frames),execution_count=len(execution),
        physical_world_audit=physical,
        demonstration_behavior_audit=behavior,
        anchors=_anchors_for_capture(frames,execution,requested,behavior_rows,segment),source_manifest_sha256=_hash(root/'capture_manifest.json'),
        source_hashes=sums,memory_published=False,official_outcome=official,admission_outcome=outcome,
        capture_mode=_capture_mode(run),actual_admission_scope=outcome.get('scope','whole_episode'),segment_capture=segment,
        human_driving_quality_approved=False,review_status='pending_real_human_driving_quality_review',
        admission_policy=deepcopy(frozen_policy),accepted_with_min_speed_penalty=outcome['accepted_with_min_speed_penalty'])


def _generalization_interaction_coverage(split, partition, episodes):
    """Summarize already audited target encounters separately from family labels and route scores."""
    from .demo_segment import verified_bypass_evidence
    selected = {(r['official_route_id'], r['source_manifest_sha256'])
                for r in partition['selection_snapshot']['captures']}
    admitted = set()
    completed = set()
    for episode in episodes:
        rid = episode['official_route_id']
        if (rid, episode.get('source_manifest_sha256')) not in selected:
            continue
        admitted.add(rid)
        segment = episode.get('segment_capture') or {}
        spec = segment.get('spec', {})
        if (spec.get('variant') == 'Generalization' and spec.get('expected_scenarios')
                and segment.get('target_observed') is True and segment.get('interaction_completed_frame') is not None
                and verified_bypass_evidence(segment)):
            # _check_episode has already replayed this evidence. This includes
            # an official route ending naturally after the actual interaction.
            completed.add(rid)
    verified, unverified, missing = [], [], []
    for row in split['rows']:
        rid = next(r['route_id'] for r in row['collection'] if r['variant'] == 'Generalization')
        if rid in completed:
            verified.append(row['family'])
        elif rid in admitted:
            unverified.append(row['family'])
        else:
            missing.append(row['family'])
    return dict(verified_target_interaction_families=verified,
                admitted_without_target_completion_evidence=unverified,
                missing_admitted_generalization_families=missing)


def check_official_demonstrations(captures,*,anchor_frames=None,require_all_families=False,collection_split=None,
                                 allow_partial_collection=False,allow_min_speed_only=False):
    _admission_policy(allow_min_speed_only)
    if type(allow_partial_collection) is not bool or allow_partial_collection and collection_split is None:
        raise ValueError('Explicit partial collection requires --collection-split')
    roots=[]
    for directory in captures:
        path=filesystem_path(directory)
        if not path.is_dir():raise ValueError('Capture directory missing: '+str(path))
        roots.extend([path] if (path/'run.json').is_file() else sorted(p.parent for p in path.rglob('run.json')))
    if not roots or len(set(roots))!=len(roots):raise ValueError('No captures or overlapping capture selections')
    if anchor_frames and len(roots)!=1:raise ValueError('Explicit anchors require one capture')
    episodes=[];rejected=[]
    for root in roots:
        try:episodes.append(_check_episode(root,anchor_frames,**({'allow_min_speed_only':True} if allow_min_speed_only else {})))
        except (ValueError,KeyError,TypeError,OSError) as exc:rejected.append(dict(episode=str(root),reason=str(exc)))
    families=sorted({episode['scenario_family'] for episode in episodes})
    if require_all_families and set(families)!=set(FAMILIES):
        rejected.append(dict(episode=None,reason='Expected actual official demonstrations for all 17 families'))
    report=dict(protocol=PROTOCOL,demo_suite=SUITE,valid=bool(episodes) and not rejected,episodes=episodes,rejected=rejected,
        successful_episodes=len(episodes),scenario_families=families,all_17_families_present=set(families)==set(FAMILIES),
        memory_published=False,teacher_called=False,human_review_fabricated=False,
        data_use='targeted_seen_scenario_demonstration;not_unseen_route_generalization')
    if collection_split is not None:
        from .memory_split import load_transfer_split,validate_collection_partition
        split=load_transfer_split(collection_split)
        partition=validate_collection_partition(split,episodes,allow_partial=allow_partial_collection)
        partition['generalization_interaction_coverage']=_generalization_interaction_coverage(split,partition,episodes)
        report.update(collection_partition=partition,valid=report['valid'] and partition['valid'],
            data_use='same_family_heldout_route_pair;not_unseen_map_or_family_generalization')
    return report


def prepare_official_demonstrations(captures,output,workbench_root,*,pipeline_config=None,dino_model=None,
                                    device='cpu',anchor_frames=None,require_all_families=False,collection_split=None,
                                    allow_partial_collection=False,allow_min_speed_only=False):
    checked=check_official_demonstrations(captures,anchor_frames=anchor_frames,
        require_all_families=require_all_families,collection_split=collection_split,
        allow_partial_collection=allow_partial_collection,allow_min_speed_only=allow_min_speed_only)
    if not checked['valid']:
        raise ValueError('Official demonstrations not eligible: '+str(dict(
            rejected=checked['rejected'],partition=checked.get('collection_partition'))))
    split_snapshot=None
    if collection_split is not None:
        from .memory_split import load_transfer_split
        split_snapshot=load_transfer_split(collection_split)
        if _digest(split_snapshot)!=checked['collection_partition']['split_sha256']:
            raise ValueError('Collection split changed between admission and preparation')
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
            current=_check_episode(source,ids,**({'allow_min_speed_only':True} if allow_min_speed_only else {}))
            if current['source_manifest_sha256']!=episode['source_manifest_sha256']:raise ValueError('Source changed before preparation')
            results.append(pipeline.prepare(cfg,source,output,model_path=cfg['features']['dino_model'],device=device,
                anchor_alignment='strict',selected_anchor_frame_ids=ids))
            if _hash(source/'capture_manifest.json')!=episode['source_manifest_sha256']:raise ValueError('Source changed during preparation')
            for name,digest in episode['source_hashes'].items():
                if _hash(_contained(source,name))!=digest:raise ValueError('Source asset changed during preparation: '+name)
        behavior_tasks=_bind_behavior_review(output,checked,storage.atomic_json)
        partition_tasks=None
        if split_snapshot is not None:
            partition_tasks=_bind_prepared_partition(output,checked,split_snapshot,storage.atomic_json)
        report=dict(checked,status='prepared_pending_real_teacher_and_human_review',preparations=results,published=False,reviewed=False,
            demonstration_review_task_binding=behavior_tasks)
        if split_snapshot is not None:
            report.update(collection_split_snapshot=split_snapshot,partition_task_binding=partition_tasks)
        storage.atomic_json(output/'official_demo_preparation.json',report)
        return report
    finally:
        sys.path[:]=original_path;sys.dont_write_bytecode=old_bytecode
        if old_mpl is None:os.environ.pop('MPLCONFIGDIR',None)
        else:os.environ['MPLCONFIGDIR']=old_mpl


def _bind_behavior_review(output,checked,write_json):
    """Retain full-video/interaction review requirements in task audit metadata.

    This records pending human review; it never fabricates a review decision or
    adds audit measurements to a teacher/model's ordinary current-frame input.
    """
    approved={_digest(_json(filesystem_path(ep['episode'])/'run.json')):ep for ep in checked['episodes']}
    tasks=sorted((output/'tasks').glob('*/task.json'))
    if not tasks:raise ValueError('No workbench tasks available to bind required demonstration review')
    for path in tasks:
        task=_json(path);episode=approved.get(_digest(task.get('capture',{})))
        if episode is None:raise ValueError('Review task is not bound to an admitted demonstration')
        if any((path.parent/name).exists() for name in ('annotations','experience.json','review.json')):
            raise ValueError('Demonstration review requirements must be attached before annotation or human review')
        review=dict(schema='official-demonstration-human-review-requirement-1.0',required=True,
            status='pending_real_human_review',approved=False,model_input=False,
            source_manifest_sha256=episode['source_manifest_sha256'],source_capture=episode['episode'],
            evidence=episode.get('demonstration_behavior_audit',dict(declared=False,verified=False)),
            official_result_admission_policy=episode.get('admission_policy',_admission_policy(False)),
            original_min_speed_penalty=bool(episode.get('accepted_with_min_speed_penalty',False)),
            review_must_include_full_interaction_video=True)
        spec=(episode.get('segment_capture') or {}).get('spec',{})
        if spec.get('variant')=='Base' and spec.get('expected_scenarios')==[]:
            review.update(capture_context='paired_region_ordinary_driving',official_scripted_target_present=False)
        decision_anchor=next((a for a in episode.get('anchors',[]) if a['frame_id']==task.get('anchor_frame_id')
                              and a.get('observed_decisions')),None)
        if decision_anchor:
            review.update(borrow_decision_evidence=decision_anchor,
                review_borrowing=['confirm_oncoming_vehicles_and_wait_in_video','check_clearance_before_borrow',
                                  'confirm_actual_return_to_original_lane_clear_of_obstacle_and_traffic'])
        if episode.get('capture_mode')=='interaction_segment':
            review.update(capture_mode='interaction_segment',segment_completion_evidence=episode['segment_capture'],
                original_official_outcome=episode['official_outcome'],official_route_success_claimed=False)
        if task.get('demonstration_review') not in (None,review):
            raise ValueError('Workbench task carries different demonstration review evidence')
        task['demonstration_review']=review;write_json(path,task)
    return dict(task_count=len(tasks),human_review_required=True,review_approval_fabricated=False,
        preserved_by_upstream='audit/task.json included in review evidence and published checksums',
        limitation='pending_requirement_metadata;human_must_actually_review_full_video_and_interaction')


def _bind_prepared_partition(output,checked,split_snapshot,write_json):
    """Add audit-only provenance before annotation/review; upstream publish retains task.json."""
    expected={r['route_id']:(row['family'],r) for row in split_snapshot['rows'] for r in row['collection']}
    approved={_digest(_json(filesystem_path(ep['episode'])/'run.json')):ep for ep in checked['episodes']}
    signature=_digest(split_snapshot)
    if signature!=checked['collection_partition']['split_sha256']:
        raise ValueError('Prepared partition differs from admitted split identity')
    tasks=sorted((output/'tasks').glob('*/task.json'))
    if not tasks:
        raise ValueError('No actual workbench tasks were prepared for the frozen collection partition')
    bound=[]
    for path in tasks:
        task=_json(path);capture=task.get('capture',{})
        episode=approved.get(_digest(capture))
        if episode is None or episode['official_route_id'] not in expected:
            raise ValueError('Workbench task source is not one of the admitted collection captures')
        rid=episode['official_route_id'];family,record=expected[rid]
        if (capture.get('official_route_id')!=rid or capture.get('scenario_family')!=family
                or capture.get('seed')!=split_snapshot['collection_seed']):
            raise ValueError('Workbench task capture changed its official collection identity')
        if any((path.parent/name).exists() for name in ('annotations','experience.json','review.json')):
            raise ValueError('Partition provenance may only be attached before teacher annotation or human review')
        provenance=dict(schema='fail2drive-task-partition-audit-1.0',role='collection',
            split_sha256=signature,split_snapshot=deepcopy(split_snapshot),
            official_route_id=rid,official_route_variant=record['variant'],official_route_sha256=record['sha256'],
            scenario_family=family,collection_seed=episode['seed'],
            source_manifest_sha256=episode['source_manifest_sha256'],
            source_admission_policy=episode.get('admission_policy',_admission_policy(False)),
            accepted_with_min_speed_penalty=bool(episode.get('accepted_with_min_speed_penalty',False)),
            model_input=False,retrieval_key=False,
            publication_path='record/audit/task.json:experiment_partition')
        partition=checked['collection_partition']
        if partition.get('selection_snapshot') is not None:
            selection=deepcopy(partition['selection_snapshot'])
            if (_digest(selection)!=partition.get('selection_sha256')
                    or selection.get('split_sha256')!=signature
                    or rid not in {r['official_route_id'] for r in selection.get('captures',[])}):
                raise ValueError('Prepared task selection differs from the admitted successful captures')
            provenance.update(selection_snapshot=selection,selection_sha256=partition['selection_sha256'],
                allow_partial_collection=partition['allow_partial_collection'])
        if task.get('experiment_partition') not in (None,provenance):
            raise ValueError('Workbench task already carries another experiment partition')
        task['experiment_partition']=provenance
        write_json(path,task);bound.append(task['id'])
    return dict(schema='fail2drive-task-partition-audit-1.0',split_sha256=signature,
        task_count=len(bound),task_ids=bound,preserved_by_upstream='audit/task.json included in review digest and published checksums',
        teacher_called=False,human_review_fabricated=False)
