"""Collection-only interaction segments on unchanged official routes.

No paper-model inputs or controls are generated here. The official manager
continues owning each tick; a scoped after-tick hook only ends collection.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import weakref
import xml.etree.ElementTree as ET

SPEC_SCHEMA = 'official-demo-segment-spec-1.0'
SCHEMA = 'official-demo-segment-capture-1.0'
STOP_METHOD = 'capture_only_manager_after_official_tick'
FAMILIES = ('BadParking','ConstructionPedestrian','ConstructionPermutations','CustomObstacles',
            'HardBrake','RightOfWay','PassableObstacles','PedestrianOtherBlocker','Animals','PedestriansOnRoad')
_ALLOWED_FAILURES = frozenset(('MIN_SPEED_INFRACTION',))


def _number(value):
    return type(value) in (int,float) and math.isfinite(value)


def _read_xml(path):
    from nine_grid.paths import filesystem_path
    path=filesystem_path(path)
    data=path.read_bytes(); root=ET.fromstring(data);route=root.find('route')
    match=re.fullmatch(r'(Base|Generalization)_(.+)_(\d{4})\.xml',path.name)
    if route is None or match is None:raise ValueError('Segment requires an original official split XML')
    result=[]
    for item in route.findall('.//scenario'):
        trigger=item.find('trigger_point')
        if trigger is None:raise ValueError('Segment target lacks its original trigger')
        kind=item.attrib['type']
        distance=item.find('distance')
        offset=float(distance.attrib['value']) if distance is not None else (12. if kind=='DynamicObjectCrossing' else 0.)
        result.append(dict(type=kind,name=item.attrib.get('name',''),
            trigger_xyz_carla=[float(trigger.attrib[k]) for k in ('x','y','z')],distance_m=offset))
    waypoints=route.find('waypoints')
    if waypoints is None:raise ValueError('Original official split XML lacks sparse route geometry')
    geometry=[dict(item.attrib) for item in waypoints]
    return dict(route_id=int(match.group(3)),variant=match.group(1),family=match.group(2),geometry=geometry,
                map=route.attrib['town'],sha256=hashlib.sha256(data).hexdigest(),scenarios=result)


def build_segment_spec(original_xml_path, paired_generalization_xml_path=None, *, max_seconds=90.):
    original=_read_xml(original_xml_path)
    paired=_read_xml(paired_generalization_xml_path or original_xml_path)
    if (original['family'] not in FAMILIES or paired['family']!=original['family']
            or paired['variant']!='Generalization' or original['map']!=paired['map']):
        raise ValueError('Interaction segments require one selected family and its paired Generalization XML')
    paired_id=original['route_id']+(1000 if original['variant']=='Base' else 0)
    if paired['route_id']!=paired_id or paired['geometry']!=original['geometry']:
        raise ValueError('Segment reference is not the original matching route pair')
    if not paired['scenarios']:raise ValueError('Generalization segment target must contain an original scenario')
    if not _number(max_seconds) or not 10.<=max_seconds<=600.:raise ValueError('Segment budget must be 10..600 simulated seconds')
    return dict(schema=SPEC_SCHEMA,capture_mode='interaction_segment',family=original['family'],
        variant=original['variant'],route_id=original['route_id'],map=original['map'],
        original_xml_sha256=original['sha256'],paired_generalization_xml_sha256=paired['sha256'],
        paired_generalization_route_id=paired['route_id'],
        expected_scenarios=original['scenarios'],paired_gen_scenarios=paired['scenarios'],
        approach_context_s=3.,recovery_s=5.,max_sim_seconds=float(max_seconds),
        minimum_future_evidence_s=3.,target_near_distance_m=30.,pass_margin_m=5.,
        source_policy='original_start_route_map_seed_NPC_and_event_behavior_unchanged',
        admission_scope='interaction_segment_only;not_official_route_success')


def validate_spec(spec):
    if (not isinstance(spec,dict) or spec.get('schema')!=SPEC_SCHEMA
            or spec.get('capture_mode')!='interaction_segment' or spec.get('family') not in FAMILIES
            or spec.get('variant') not in ('Base','Generalization')
            or not isinstance(spec.get('expected_scenarios'),list) or not spec.get('paired_gen_scenarios')
            or spec.get('recovery_s')!=5. or spec.get('minimum_future_evidence_s')!=3.
            or not _number(spec.get('max_sim_seconds')) or not 10.<=spec['max_sim_seconds']<=600.):
        raise ValueError('Invalid immutable interaction segment specification')
    return deepcopy(spec)


class SegmentTracker:
    """Pure state machine: immutable observed ticks are replayed at admission."""
    def __init__(self,spec):
        self.spec=validate_spec(spec);self.status='incomplete';self.events=[];self.previous=None
        self.start=None;self.interaction=None;self.completed=None;self.recovery_start=None
        self.last=None;self.reason=None;self.target_seen=False;self.active_seen=set();self.end_seen=set()
        self.brake_seen=False;self.actor_seen=set();self.vulnerable_seen=False
        self.expected_keys={str(i)+':'+item['type'] for i,item in enumerate(self.spec['expected_scenarios'])}

    def event(self,phase,row,reason):
        value=dict(phase=phase,frame=row['frame'],time=row['time'],reason=reason)
        self.events.append(value);return value

    def observe(self,row):
        if self.status!='incomplete':raise ValueError('Segment contains physical ticks after terminal state')
        frame,stamp=row.get('frame'),row.get('time')
        if type(frame) is not int or not _number(stamp):raise ValueError('Invalid segment source identity')
        if self.previous is not None and (frame!=self.previous['frame']+1 or abs(stamp-self.previous['time']-.05)>1e-5):
            raise ValueError('Segment evidence is not consecutive official 20Hz')
        self.previous=row;self.last=row
        if self.start is None:self.start=self.event('approach',row,'original_official_start')
        failure=row.get('failure_reason')
        if failure:
            self.status='failed';self.reason=failure;self.event('failed',row,failure);return
        if row.get('projection_valid') is not True:
            self.status='failed';self.reason='invalid_route_projection';self.event('failed',row,self.reason);return
        targets=row.get('target_scenarios',[])
        expected=self.spec['expected_scenarios']
        for target in targets:
            identity=target['target_key']
            if identity not in self.expected_keys:raise ValueError('Unknown segment target identity')
            if target.get('vulnerable_motion_observed') is True:self.vulnerable_seen=True
            if target.get('active') is True:self.active_seen.add(identity)
            if target.get('observed_near') is True and identity in self.active_seen:
                self.actor_seen.add(identity)
            if target.get('ended') is True and identity in self.active_seen:self.end_seen.add(identity)
        if row.get('lead_braking_observed') is True and any(t.get('active') is True for t in targets):self.brake_seen=True
        if expected:
            self.target_seen=bool(self.actor_seen)
            entered=self.target_seen
            count=len(expected)
            done=(len(self.active_seen)>=count and len(self.end_seen)>=count and len(self.actor_seen)>=count
                  and row.get('past_target_gate') is True)
            if self.spec['family']=='HardBrake':done=done and self.brake_seen
            if any(item['type'] in ('ConstructionObstaclePedestrian','DynamicObjectCrossing','PedestriansOnRoad') for item in expected):
                done=done and self.vulnerable_seen
        else:
            # Empty Base has no invented actor event: cross the paired Gen's
            # exact along-route interaction region and retain ordinary driving.
            entered=row.get('in_target_approach') is True
            self.target_seen=self.target_seen or entered
            done=self.target_seen and row.get('past_target_gate') is True
        if entered and self.interaction is None:
            self.interaction=self.event('interaction',row,'observed_target_encounter' if expected else 'paired_region_normal_driving')
        if (done and self.interaction is not None and self.completed is None
                and stamp-self.start['time']>=self.spec['approach_context_s']):
            self.completed=self.event('recovery',row,'target_behavior_ended_and_route_gate_passed' if expected else 'paired_region_passed')
        stable=(row.get('route_lane_match') is True and row.get('speed_mps',0.)>=.5
                and row.get('projection_distance_m',math.inf)<=1.
                and row.get('heading_error_rad',math.inf)<=.35)
        if self.completed is not None:
            if stable:
                if self.recovery_start is None:self.recovery_start=dict(frame=frame,time=stamp)
                if stamp-self.recovery_start['time']>=self.spec['recovery_s']-1e-5:
                    self.status='completed';self.reason='interaction_and_stable_recovery_completed'
                    self.event('completed',row,self.reason)
            else:self.recovery_start=None
        if self.status=='incomplete' and stamp-self.start['time']>=self.spec['max_sim_seconds']-1e-5:
            self.status='failed';self.reason='collection_budget_exhausted';self.event('failed',row,self.reason)

    def report(self):
        def field(event,key):return None if event is None else event[key]
        return dict(schema=SCHEMA,capture_mode='interaction_segment',spec=deepcopy(self.spec),status=self.status,
            events=deepcopy(self.events),stop_reason=self.reason,start_frame=field(self.start,'frame'),start_time=field(self.start,'time'),
            end_frame=field(self.last,'frame'),end_time=field(self.last,'time'),
            interaction_start_frame=field(self.interaction,'frame'),interaction_start_time=field(self.interaction,'time'),
            interaction_completed_frame=field(self.completed,'frame'),interaction_completed_time=field(self.completed,'time'),
            approach_context_available_s=(None if self.interaction is None else self.interaction['time']-self.start['time']),
            stable_recovery_start=deepcopy(self.recovery_start),recovery_s=self.spec['recovery_s'],
            target_observed=self.target_seen,official_route_success_claimed=False,
            control_stop=dict(requested=self.status!='incomplete',confirmed=False,method=STOP_METHOD,
                              frame=field(self.last,'frame'),time=field(self.last,'time')))


def validate_segment_report(report,run=None,spec=None):
    if (not isinstance(report,dict) or report.get('schema')!=SCHEMA
            or report.get('capture_mode')!='interaction_segment' or report.get('status') not in ('completed','failed','incomplete')
            or report.get('official_route_success_claimed') is not False):raise ValueError('Invalid segment report identity')
    validate_spec(report.get('spec'))
    if spec is not None and report['spec']!=spec:raise ValueError('Segment specification changed after launch')
    if run is not None and (run.get('capture_mode')!='interaction_segment' or run.get('segment_capture_path')!='segment_capture.json'):
        raise ValueError('Agent run did not declare an isolated segment capture')
    control=report.get('control_stop',{})
    controlled=(control.get('requested') is True and control.get('confirmed') is True
        and control.get('method')==STOP_METHOD and control.get('frame')==report.get('end_frame')
        and control.get('time')==report.get('end_time'))
    return dict(valid=True,completed=report['status']=='completed' and controlled,controlled_stop=controlled,
                status=report['status'],stop_reason=report.get('stop_reason'))


def validate_segment_evidence(summary,ticks,execution=None,source_times=None,*,allow_natural_full_route=False):
    quick=validate_segment_report(summary)
    natural=(allow_natural_full_route and summary.get('termination')=='official_route_ended_naturally'
             and summary.get('status') in ('incomplete','completed') and summary.get('interaction_completed_time') is not None)
    if not (quick['completed'] or natural) or not ticks:raise ValueError('Interaction segment did not complete a controlled after-tick stop')
    replay=SegmentTracker(summary['spec'])
    for row in ticks:replay.observe(row)
    actual=replay.report()
    for key in actual:
        if key!='control_stop' and actual[key]!=summary.get(key):raise ValueError('Segment state/evidence replay mismatch: '+key)
    if actual['status']!='completed' and not natural:raise ValueError('No completed target interaction and stable recovery')
    if execution is not None:
        if len(execution)!=len(ticks):raise ValueError('Segment/actual execution physical tick count mismatch')
        for row,control in zip(ticks,execution):
            if row['frame']!=control['frame'] or abs(row['time']-control['time'])>1e-7:
                raise ValueError('Segment tick is not bound to actual physical execution')
            ego=control.get('observed_ego',{})
            velocity=ego.get('velocity')
            if isinstance(velocity,list) and len(velocity)>=2:
                if abs(math.hypot(*velocity[:2])-row.get('speed_mps',math.inf))>1e-5:
                    raise ValueError('Segment recovery speed differs from actual execution source')
            position=ego.get('position')
            if isinstance(position,list) and len(position)>=2:
                recorded=row.get('ego_position_xyz_carla_m',[])
                if len(recorded)!=3 or math.dist(position[:2],[recorded[0],-recorded[1]])>1e-5:
                    raise ValueError('Segment position differs from actual execution source')
            if ego.get('lane_id') is not None and ego['lane_id']!=row.get('ego_lane_id'):
                raise ValueError('Segment lane differs from actual execution source')
    if source_times is not None:
        times=list(source_times)
        if any(not _number(t) for t in times):raise ValueError('Nonfinite actual source timestamp')
        finish=summary['interaction_completed_time']
        eligible=[t for t in times if t>=summary['interaction_start_time']-1e-5 and t<=finish+1.+1e-5
                  and all(any(abs(other-t-offset)<1e-5 for other in times) for offset in (0.,1.,2.,3.))]
        if not eligible:raise ValueError('Segment lacks a critical anchor with actual current/+1/+2/+3s evidence')
    return dict(valid=True,scope='interaction_segment_only',frames=len(ticks),status=actual['status'],
                target_observed=summary['target_observed'],recovery_s=summary['recovery_s'],
                official_route_success_claimed=False,human_review_required=True)
