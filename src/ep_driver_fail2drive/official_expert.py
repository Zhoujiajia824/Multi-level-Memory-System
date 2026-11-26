"""Offline-only adapter for the pinned Fail2Drive data-collection expert.

Official scenario creation, NPC behavior, world advancement and scoring stay
upstream-owned. The expert gets private scenario bookkeeping and private NumPy
randomness, and may return only the ego's VehicleControl. It is a teacher
candidate, never a guarantee of successful or safe driving.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

from nine_grid.paths import filesystem_path
from . import OFFICIAL_COMMIT


SCHEMA = 'ep-driver-f2d-official-expert-adapter-1.0'
DECISION_SCHEMA = 'ep-driver-f2d-offline-expert-decision-1.0'
DRIVING_POLICY_SCHEMA = 'ep-driver-f2d-cautious-demonstration-1.0'
SOURCES = {
    'autopilot.py': '8e05a4e489530c9a40a7a458ce0b1661180bf2f9c50bb0f5e3c7500b1008e92d',
    'transfuser_utils.py': '6e47c04d8279cd968698cab0f638b3192f49cd088c60dec7809d63ff30c34449',
    'nav_planner.py': '18d5af2b34a9f3bd27f2e0b1330898630decf62d068a133513f236b718a07768',
    'lateral_controller.py': '5da07b60e1d960e34074968689a254967c1c83a702b8015659433be1a0ed84e5',
    'privileged_route_planner.py': '09c0e93fcf80f44abf635abfe51ca256611acc8050cdac468eee88f4b1dd3518',
    'config.py': '46e0c91128a5e26184fea52a012681b084a7e35dc62638eab5709b9a48733a51',
    'longitudinal_controller.py': '550a884e4e331a513417be7b06a0aa929c2fb4a00a5eddb56e2857937f826509',
    'kinematic_bicycle_model.py': '9a2016fe41dc0ef4b2ad2da8d89652ea2bc4993937db5a3313a02c23042d1449',
}
UTILITIES = ('get_relative_transform', 'get_traffic_light_waypoints', 'inverse_conversion_2d',
             'preprocess_compass', 'rotate_point', 'normalize_angle')
SPEED_LIMIT_DATA = {'scenario_runner/speed_limits/Town13_speed_limits.npy':
                    'b1453f1b09b4e1cc0ac43d93440487fd94ed924e31e1514ed413cdee281e02e5'}
LOCAL_BASE = 'leaderboard/leaderboard/autoagents/autonomous_agent_local.py'
LOCAL_BASE_SHA256 = '4221810e0a223c8d074973607f524ea1a726b9120bc1a90ccf1016a255b5d607'
_SIDE_EFFECT_METHODS = frozenset({
    'destroy', 'set_transform', 'set_location', 'set_target_velocity', 'set_velocity',
    'set_angular_velocity', 'set_target_angular_velocity', 'apply_control',
    'set_simulate_physics', 'set_autopilot', 'set_light_state', 'set_state',
    'set_green_time', 'set_red_time', 'set_yellow_time', 'freeze',
    'spawn_actor', 'try_spawn_actor', 'request_new_actor', 'request_new_actors',
    'tick', 'wait_for_tick', 'apply_settings', 'load_world', 'reload_world',
    'apply_batch', 'apply_batch_sync', 'start_recorder', 'stop_recorder',
})


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _decision_record(ego_speed_xyz, target_speed, speed_limit, keep_driving,
                     brake_hazard, control_brake, carla_vehicle_limit_kmh=None):
    """Scalar evidence from the existing decision, never a new driving action."""
    numbers = dict(ego_speed_xyz=float(ego_speed_xyz), target_speed=float(target_speed),
                   speed_limit=float(speed_limit))
    if any(not _number(value) for value in numbers.values()):
        raise ValueError('Official expert decision contains a nonfinite speed')
    carla_limit=(float(carla_vehicle_limit_kmh)/3.6 if carla_vehicle_limit_kmh is not None else None)
    if carla_limit is not None and (not _number(carla_limit) or carla_limit<=0):
        carla_limit=None
    return dict(numbers, keep_driving=bool(keep_driving),brake_hazard=bool(brake_hazard),
        control_brake=bool(control_brake),carla_vehicle_limit_mps=carla_limit,
        npy_route_limit_mps=numbers['speed_limit'],
        speed_limit_conflict=(abs(carla_limit-numbers['speed_limit'])>1e-3 if carla_limit is not None else None),
        speed_limit_source='pinned_preprocessed_NPY_route_nearest_point',
        carla_limit_source='Vehicle.get_speed_limit_current',
        limit_conflict_is_speeding_proof=False,
        limit_preprocessing='upstream_5m_neighborhood_max_and_round_up_to_50_80_100_120_kmh')


def _container_copy(value):
    """Copy only Python containers; preserve live CARLA actor references."""
    if isinstance(value, list):
        return [_container_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_container_copy(item) for item in value)
    if isinstance(value, dict):
        return {key: _container_copy(item) for key, item in value.items()}
    return value


def _json_decision_copy(value, path='expert_decision'):
    """Copy evidence into strict JSON types without changing live decisions.

    NumPy route-index comparisons return numpy.bool_, and other expert evidence
    may retain NumPy integer/float scalars. Only those real scalar types are
    normalized; arrays, SDK objects and nonfinite values remain invalid.
    """
    kind=type(value)
    if value is None or kind in (bool,int,str):
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError('Nonfinite official expert decision at '+path)
        return value
    if kind is dict:
        if any(type(key) is not str for key in value):
            raise TypeError('Official expert decision keys must be strings at '+path)
        return {key:_json_decision_copy(item,path+'.'+key) for key,item in value.items()}
    if kind in (list,tuple):
        return [_json_decision_copy(item,'%s[%d]'%(path,index)) for index,item in enumerate(value)]
    # Keep importing/preflighting this module stdlib-only. Do not accept an
    # arbitrary object's .item()/__float__ method as a serialization fallback.
    if kind.__module__.split('.')[0]=='numpy':
        import numpy as np
        if isinstance(value,np.bool_):
            return bool(value)
        if isinstance(value,np.integer):
            return int(value)
        if isinstance(value,np.floating):
            return _json_decision_copy(float(value),path)
    raise TypeError('Unsupported official expert decision at %s: %s'%(path,kind.__name__))


def _ep_cautious_target(pilot, route_points, target_speed, ego_speed):
    """Apply only the explicit offline teacher policy, before hazard prediction."""
    policy = getattr(pilot, '_ep_caution_policy', None)
    if policy is None:
        raise RuntimeError('Cautious demonstration policy is not installed')
    limit, evidence = policy.limit(pilot, route_points, target_speed, ego_speed)
    limit = float(limit)
    if not _number(limit) or limit < 0 or limit > float(target_speed)+1e-9:
        raise ValueError('Cautious demonstration policy must return a finite nonincreasing target')
    if not isinstance(evidence, dict):
        raise ValueError('Cautious demonstration policy omitted its decision evidence')
    pilot._ep_caution_decision = _container_copy(evidence)
    return limit


def _maneuver_key(kind,start,end):
    return (str(kind),int(start),int(end))


def _ep_before_opposing_entry(pilot,kind,direction,start,end):
    """Keep checking the gap until the frozen ego body enters the borrowed lane.

    Reference-shift progress is deliberately insufficient: a long smooth
    transition can begin while the actual car is still wholly in its lane.
    Once observed, entry is latched for this maneuver so a new cached-gap test
    cannot command a stop merely because the return transition has started.
    """
    key=_maneuver_key(kind,start,end)
    if key in pilot._ep_committed_maneuvers:
        return False
    try:
        planner=pilot._waypoint_planner
        index=int(planner.route_index)
        wp=planner.route_waypoints[index]
        origin=wp.transform.location
        normal=wp.transform.get_right_vector()
        width=float(wp.lane_width)
        nx,ny=float(normal.x),float(normal.y)
        scale=math.hypot(nx,ny)
        if direction not in ('left','right') or width<=0 or scale<=0:
            raise ValueError('invalid physical lane cross-section')
        nx,ny=nx/scale,ny/scale
        snapshot=pilot._world.get_snapshot()
        frozen=snapshot.find(pilot._vehicle.id)
        if frozen is None:
            raise ValueError('ego is absent from frozen source')
        pose=frozen.get_transform()
        vertices=list(pilot._vehicle.bounding_box.get_world_vertices(pose))
        lateral=[(float(v.x)-float(origin.x))*nx+(float(v.y)-float(origin.y))*ny for v in vertices]
        numbers=[width,nx,ny,float(origin.x),float(origin.y),*lateral]
        if len(vertices)!=8 or not all(math.isfinite(value) for value in numbers):
            raise ValueError('missing finite physical body/lane geometry')
    except (AttributeError,IndexError,TypeError,ValueError) as error:
        raise ValueError('Cannot certify demonstration borrowing entry from frozen body/lane geometry') from error
    # CARLA's right vector is positive to the right. A right-side obstacle is
    # passed on the left; the trigger is the real body, not a waypoint center.
    extreme=-min(lateral) if direction=='right' else max(lateral)
    entered=extreme>=width/2.
    if entered:
        pilot._ep_committed_maneuvers.add(key)
        pilot._ep_entry_commit_evidence[key]=dict(source_frame=int(snapshot.frame),
            source_time=float(snapshot.timestamp.elapsed_seconds),
            original_route_index=index,original_lane_id='%s|%s|%s'%(
                wp.road_id,wp.section_id,wp.lane_id),original_lane_width_m=width,
            physical_body_lateral_bounds_m=[min(lateral),max(lateral)],
            borrowed_side='left' if direction=='right' else 'right',
            borrowed_boundary_m=-width/2. if direction=='right' else width/2.,
            source='frozen_ego_transform;actual_body_world_vertices;unshifted_route_waypoint')
    return not entered


def _ep_cautious_gap_time(pilot,distance,target_speed,ego_speed,min_speed,to_index=None):
    """Estimate the complete borrowed-lane occupancy at the teacher's speed.

    The upstream minimum-time acceleration model is inappropriate after slowing
    the teacher to walking speed. This estimate includes the existing route-end,
    body and safety-distance terms supplied by the caller, uses a modest 1m/s2
    startup model, and never assumes a speed above the actual cautious limit.
    An unexpected later yield can still extend occupancy; this is not a proof.
    """
    from .demonstration_policy import SETTINGS
    map_limit=float(pilot._vehicle.get_speed_limit())/3.6
    values=[float(distance),float(target_speed),float(ego_speed),float(min_speed),map_limit]
    if not all(math.isfinite(value) for value in values) or distance<0 or map_limit<=0:
        raise ValueError('Cannot evaluate demonstration opposing gap with invalid current speed/distance')
    cap=max(0.,min(float(target_speed),float(min_speed),map_limit,float(SETTINGS['maneuver_speed_mps'])))
    route=pilot._waypoint_planner.route_points[pilot._waypoint_planner.route_index:]
    cap,policy_evidence=pilot._ep_caution_policy.limit(pilot,route,cap,ego_speed)
    cap=float(cap)
    if not math.isfinite(cap) or cap<0 or cap>min(float(target_speed),float(min_speed),map_limit,
                                               float(SETTINGS['maneuver_speed_mps']))+1e-9:
        raise ValueError('Opposing gap policy must preserve the cautious speed upper bound')
    snapshot=pilot._world.get_snapshot();frozen=snapshot.find(pilot._vehicle.id)
    if frozen is None:raise ValueError('Cannot evaluate opposing gap without frozen ego velocity')
    velocity=frozen.get_velocity();speed_xy=math.hypot(velocity.x,velocity.y)
    if not math.isfinite(speed_xy):raise ValueError('Invalid frozen horizontal speed for opposing gap')
    supplied_distance=float(distance);arc_distance=None;margin=None
    if to_index is not None:
        planner=pilot._waypoint_planner;start=int(planner.route_index);end=int(to_index)
        if not 0<=start<end<len(planner.route_points):
            raise ValueError('Opposing gap has no forward reference through the return endpoint')
        points=[[float(value) for value in point[:3]] for point in planner.route_points[start:end+1]]
        pose=frozen.get_transform();here=[float(getattr(pose.location,k)) for k in ('x','y','z')]
        if (not all(len(point)==3 and all(math.isfinite(value) for value in point) for point in points)
                or not all(math.isfinite(value) for value in here)):
            raise ValueError('Opposing gap reference lacks finite XYZ arc geometry')
        arc_distance=math.dist(here,points[0])+sum(math.dist(a,b) for a,b in zip(points,points[1:]))
        margin=2.*float(pilot._vehicle.bounding_box.extent.x)+float(pilot.config.check_path_free_safety_distance)
        if not math.isfinite(margin) or margin<0:
            raise ValueError('Invalid original opposing-gap body/safety margin')
        # Preserve every original margin while replacing a curved-road chord
        # with the longer actual borrowed-and-return reference.
        distance=max(supplied_distance,arc_distance+margin)
    # Clamping a faster initial measurement down to the maneuver speed avoids
    # treating a high approach speed as permission to race through the gap.
    initial=min(speed_xy,cap);acceleration=1.;estimate=None
    if cap>0:
        accelerate_distance=max(0.,(cap*cap-initial*initial)/(2.*acceleration))
        if distance<=accelerate_distance:
            estimate=(math.sqrt(initial*initial+2.*acceleration*distance)-initial)/acceleration
        else:
            estimate=(cap-initial)/acceleration+(distance-accelerate_distance)/cap
        estimate=max(float(distance)/cap,estimate)
    evidence=dict(schema=DRIVING_POLICY_SCHEMA,source_frame=int(snapshot.frame),
        source_time=float(snapshot.timestamp.elapsed_seconds),distance_including_original_margins_m=float(distance),
        supplied_chord_distance_with_margins_m=supplied_distance,reference_arc_to_return_end_m=arc_distance,
        original_body_and_safety_margin_m=margin,
        distance_basis='max_original_and_actual_reference_arc_plus_original_margins' if to_index is not None else
                       'explicit_distance_without_reference_endpoint',
        cautious_gap_speed_mps=cap,map_vehicle_limit_mps=map_limit,observed_speed_xy_mps=speed_xy,
        startup_speed_mps=initial,startup_acceleration_mps2=acceleration,occupancy_estimate_s=estimate,
        current_policy_binding_reasons=_container_copy(policy_evidence.get('binding_reasons',[])),
        method='cautious_startup_then_speed_limited_full_borrow_and_return',
        upstream_minimum_time_model_used=False,original_gap_safety_margins_unchanged=True,
        future_waits_bounded=False,collision_free_proof=False)
    pilot._ep_gap_time_estimates.append(evidence)
    return estimate


def _maneuver_record(pilot, provider):
    """Label cached pre-entry clearance honestly; it is not current safety."""
    planner = pilot._waypoint_planner
    route_index = int(planner.route_index)
    rows = []
    for kind, data in provider.active_scenarios:
        if kind in {'AccidentTwoWays', 'ConstructionObstacleTwoWays', 'ParkedObstacleTwoWays',
                    'VehicleOpensDoorTwoWays'} and len(data) == 7:
            changed, start, end, gap = data[3:7]
        elif kind == 'HazardAtSideLaneTwoWays' and len(data) == 6:
            changed, start, end, gap = data[2:6]
        else:
            continue
        if not changed:
            continue
        key=_maneuver_key(kind,start,end)
        committed=key in getattr(pilot,'_ep_committed_maneuvers',set())
        evaluated = (bool(getattr(pilot,'_ep_entry_gap_evaluated',False)) and not committed
                     and key==getattr(pilot,'_ep_entry_gap_key',None))
        rows.append(dict(kind=kind,route_index=route_index,from_index=int(start),to_index=int(end),
            reference_shift_started=bool(route_index>start),maneuver_committed=committed,
            physical_entry_evidence=_container_copy(getattr(pilot,'_ep_entry_commit_evidence',{}).get(key)),
            last_preentry_vehicle_gap_clear=bool(gap),
            current_preentry_vehicle_gap_clear=bool(gap) if evaluated else None,
            clearance_semantics=('cached_entry_decision_not_current_safety' if committed else
                                 'current_vehicle_gap_only' if evaluated else 'not_evaluated_this_frame'),
            actor_hazard_and_traffic_checks='executed_every_frame',
            gap_rule_forces_stop_in_committed_opposite_lane=False))
    return dict(schema=DRIVING_POLICY_SCHEMA,maneuvers=rows,
        preentry_gap_rechecks=getattr(pilot,'_ep_preentry_gap_rechecks',0),
        gap_time_estimates=_container_copy(getattr(pilot,'_ep_gap_time_estimates',[])),
        keep_driving_is_safety_override=False)


class _ProviderView:
    """Private expert notes; the scenario runner's registry is never changed."""
    def __init__(self, provider):
        self._provider = provider
        self.active_scenarios = []
        self._working = {}
        self._consumed = set()
        self._last_keys = {}

    def refresh(self):
        original = self._provider.active_scenarios
        if not isinstance(original, list):
            raise ValueError('Official scenario registry must be a list')
        values, keys = [], {}
        for entry in original:
            if (not isinstance(entry, (tuple, list)) or len(entry) != 2
                    or not isinstance(entry[0], str) or not isinstance(entry[1], list)):
                raise ValueError('Unsupported pinned expert scenario entry')
            # Keep the source list itself alive in _working: identity cannot be
            # recycled and mistaken for a different scenario during the route.
            key = (entry[0], id(entry[1]))
            if key in self._consumed:
                continue
            if key not in self._working:
                self._working[key] = (entry[1], _container_copy(entry))
            private = self._working[key][1]
            values.append(private)
            keys[id(private[1])] = key
        self.active_scenarios, self._last_keys = values, keys

    def finish(self):
        remaining = {id(entry[1]) for entry in self.active_scenarios}
        self._consumed.update(key for identity, key in self._last_keys.items() if identity not in remaining)

    def __getattr__(self, name):
        if name not in {'get_map', 'get_world', 'get_hero_actor', 'get_velocity', 'get_client'}:
            raise RuntimeError('Expert provider method is not a permitted read: '+name)
        return getattr(self._provider, name)


class _NumpyView:
    def __init__(self, numpy, seed):
        self._numpy = numpy
        self.random = numpy.random.RandomState(seed)

    def __getattr__(self, name):
        return getattr(self._numpy, name)


def source_identity(official_root):
    """Validate every reused source before any expert code is loaded."""
    root = filesystem_path(official_root)
    result = {}
    for name, expected in SOURCES.items():
        target = root/'team_code'/name
        if not target.is_file():
            raise ValueError('Pinned expert source missing: '+name)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Pinned expert source SHA256 mismatch: '+name)
        result[name] = actual
    data = {}
    for name, expected in SPEED_LIMIT_DATA.items():
        target = root/name
        if not target.is_file():
            raise ValueError('Pinned expert speed-limit data missing: '+name)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Pinned expert speed-limit data differs or is an LFS pointer: '+name)
        data[name] = actual
    base = root/LOCAL_BASE
    if not base.is_file() or hashlib.sha256(base.read_bytes()).hexdigest() != LOCAL_BASE_SHA256:
        raise ValueError('Pinned expert local agent base SHA256 mismatch or missing')
    return dict(schema=SCHEMA, official_commit=OFFICIAL_COMMIT, sources_sha256=result,data_sha256=data,
                base_sources_sha256={LOCAL_BASE:LOCAL_BASE_SHA256})


def _no_npc_deletion(actor):
    return RuntimeError('Expert encountered a degenerate NPC body; preserving actor '+str(actor.id)+
                        ' and rejecting the demonstration instead of deleting it')


def _disabled_recorder(*_args, **_kwargs):
    raise RuntimeError('Expert-owned recorder/logger is disabled; official launcher owns recording')


def _side_effects(tree):
    return sorted({node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr in _SIDE_EFFECT_METHODS})


def _expert_tree(text, filename):
    """Small auditable adapter edits in memory, never writes upstream files."""
    tree = ast.parse(text, filename=filename)
    removed = {'os', 'ujson', 'numpy', 'transfuser_utils'}
    removed_from = {'nav_planner', 'lateral_controller', 'privileged_route_planner', 'config',
                    'scenario_logger', 'longitudinal_controller', 'kinematic_bicycle_model',
                    'srunner.scenariomanager.carla_data_provider'}
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(alias.name in removed for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module in removed_from)]
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AutoPilot']
    if len(classes) != 1:
        raise ValueError('Pinned source no longer has exactly one AutoPilot class')
    patched = 0
    decision_patches = 0
    hazard_patches = 0
    gap_patches = 0
    gap_result_patches = 0
    gap_time_patches = 0
    for method in classes[0].body:
        if not isinstance(method, ast.FunctionDef):
            continue
        if method.name == '_init':
            class RejectDeletion(ast.NodeTransformer):
                def visit_Expr(self, node):
                    nonlocal patched
                    if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                            and isinstance(node.value.func.value, ast.Name)
                            and node.value.func.value.id == 'actor' and node.value.func.attr == 'destroy'):
                        patched += 1
                        return ast.copy_location(ast.Raise(exc=ast.Call(func=ast.Name(id='_no_npc_deletion',ctx=ast.Load()),
                            args=[ast.Name(id='actor',ctx=ast.Load())], keywords=[]),cause=None),node)
                    return self.generic_visit(node)
            RejectDeletion().visit(method)
        elif method.name == 'toggle_recording':
            method.body = [ast.Expr(value=ast.Call(func=ast.Name(id='_disabled_recorder',ctx=ast.Load()),args=[],keywords=[]))]
        elif method.name == '_get_control':
            # The upstream keep_driving branch suppresses all pedestrian,
            # vehicle, traffic-light and stop-sign checks during overtaking.
            # A teacher must retain those checks throughout the maneuver.
            class AlwaysCheckHazards(ast.NodeTransformer):
                def visit_If(self, node):
                    nonlocal hazard_patches
                    node = self.generic_visit(node)
                    if isinstance(node.test,ast.Name) and node.test.id == 'keep_driving':
                        if (len(node.orelse) != 1 or not isinstance(node.orelse[0],ast.Assign)
                                or not isinstance(node.orelse[0].value,ast.Call)
                                or not isinstance(node.orelse[0].value.func,ast.Attribute)
                                or node.orelse[0].value.func.attr != 'get_brake_and_target_speed'):
                            raise ValueError('Unexpected overtaking hazard bypass in pinned expert')
                        hazard_patches += 1
                        before = ast.parse(
                            'target_speed = min(target_speed, target_speed_route_obstacle)\n'
                            'self._ep_maneuver_decision = _maneuver_record(self, CarlaDataProvider)\n'
                            'target_speed = _ep_cautious_target(self, route_np, target_speed, ego_speed)\n').body
                        return [*[ast.copy_location(item,node) for item in before],*node.orelse]
                    return node
            AlwaysCheckHazards().visit(method)
            # Read final scalars immediately before the sole control return.
            class RecordDecision(ast.NodeTransformer):
                def visit_Return(self, node):
                    nonlocal decision_patches
                    if (not isinstance(node.value, ast.Tuple) or len(node.value.elts) != 2
                            or any(not isinstance(value,ast.Name) or value.id != expected
                                   for value,expected in zip(node.value.elts,('control','driving_data')))):
                        raise ValueError('Unexpected control return shape in pinned expert')
                    decision_patches += 1
                    target=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),
                                         attr='_ep_last_decision',ctx=ast.Store())
                    call=ast.Call(func=ast.Name(id='_decision_record',ctx=ast.Load()),
                        args=[ast.Name(id=name,ctx=ast.Load()) for name in
                              ('ego_speed','target_speed','speed_limit','keep_driving','brake','control_brake')]+[
                            ast.Call(func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),
                                attr='_vehicle',ctx=ast.Load()),attr='get_speed_limit',ctx=ast.Load()),args=[],keywords=[])],
                        keywords=[])
                    return [ast.copy_location(ast.Assign(targets=[target],value=call),node),node]
            RecordDecision().visit(method)
        elif method.name == '_manage_route_obstacle_scenarios':
            class RecheckEntryGap(ast.NodeTransformer):
                def visit_Assign(self,node):
                    nonlocal gap_result_patches,gap_time_patches
                    node=self.generic_visit(node)
                    if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                            and node.targets[0].id=='ego_time' and isinstance(node.value,ast.Call)
                            and isinstance(node.value.func,ast.Name)
                            and node.value.func.id=='compute_min_time_for_distance'):
                        gap_time_patches+=1
                        replacement=ast.parse(
                            'ego_time = _ep_cautious_gap_time(self, ego_distance, target_speed, ego_speed, min_speed, to_index)\n'
                            'if ego_time is None:\n    return False\n').body
                        return [ast.copy_location(item,node) for item in replacement]
                    if (len(node.targets)==1 and isinstance(node.targets[0],ast.Subscript)
                            and isinstance(node.targets[0].value,ast.Name)
                            and node.targets[0].value.id=='scenario_data'
                            and isinstance(node.targets[0].slice,ast.Constant)
                            and node.targets[0].slice.value==6
                            and isinstance(node.value,ast.Name) and node.value.id=='path_clear'):
                        gap_result_patches+=1
                        after=ast.parse('self._ep_preentry_gap_rechecks += 1\nself._ep_entry_gap_evaluated = True\n'
                            'self._ep_entry_gap_key = (scenario_type, int(from_index), int(to_index))').body
                        return [node,*[ast.copy_location(item,node) for item in after]]
                    return node
                def visit_If(self,node):
                    nonlocal gap_patches
                    node = self.generic_visit(node)
                    if not isinstance(node.test,ast.BoolOp):
                        return node
                    matches=[item for item in node.test.values if isinstance(item,ast.UnaryOp)
                        and isinstance(item.op,ast.Not) and isinstance(item.operand,ast.Name)
                        and item.operand.id == 'path_clear']
                    if not matches:
                        return node
                    if len(matches) != 1 or not any(isinstance(item,ast.Name) and item.id=='changed_route'
                                                  for item in node.test.values):
                        raise ValueError('Unexpected pre-entry gap condition in pinned expert')
                    gap_patches += 1
                    # Reassess even a previously clear gap until crossing the
                    # maneuver start. Once committed, do not turn a cached gap
                    # test into a new stop command in the opposing lane; the
                    # full current actor/traffic forecast still applies.
                    replacement=ast.parse(
                        '_ep_before_opposing_entry(self, scenario_type, direction, from_index, to_index)',
                        mode='eval').body
                    node.test.values=[replacement if item is matches[0] else item for item in node.test.values]
                    return node
            RecheckEntryGap().visit(method)
            reset=ast.parse('self._ep_entry_gap_evaluated = False\nself._ep_entry_gap_key = None\n'
                            'self._ep_gap_time_estimates = []').body
            # Preserve the method's public docstring.
            method.body[1:1]=[ast.copy_location(item,method.body[0]) for item in reset]
    if patched != 1:
        raise ValueError('Unexpected NPC cleanup shape in pinned expert')
    if decision_patches != 1:
        raise ValueError('Unexpected decision recording shape in pinned expert')
    if hazard_patches != 1 or gap_patches != 1 or gap_result_patches != 1 or gap_time_patches != 1:
        raise ValueError('Unexpected cautious demonstration control/gap shape in pinned expert')
    effects = _side_effects(tree)
    if effects:
        raise ValueError('Expert attempted a forbidden world/actor mutation: '+str(effects))
    return ast.fix_missing_locations(tree)


def _load_module(name, path, private_os):
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text,filename=str(path))
    if _side_effects(tree):
        raise ValueError('Expert dependency contains world/actor mutation: '+name)
    if any(isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name)
           and node.value.id == 'np' and node.attr == 'random' for node in ast.walk(tree)):
        raise ValueError('Expert dependency would consume scenario NumPy randomness: '+name)
    tree.body = [node for node in tree.body if not (
        isinstance(node,ast.Import) and any(alias.name == 'os' for alias in node.names))]
    module = ModuleType('_ep_f2d_offline_expert_'+name)
    module.__file__ = str(path)
    module.__dict__['os'] = private_os
    exec(compile(tree,str(path),'exec'),module.__dict__)
    return module


def _load_utilities(path, carla, numpy):
    original = ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    nodes = [node for node in original.body if isinstance(node,ast.FunctionDef) and node.name in UTILITIES]
    if {node.name for node in nodes} != set(UTILITIES):
        raise ValueError('Pinned expert pure-utility dependency set changed')
    tree = ast.Module(body=nodes,type_ignores=[])
    if _side_effects(tree):
        raise ValueError('Expert utility mutates world or actors')
    module = ModuleType('_ep_f2d_offline_expert_utils')
    module.__dict__.update(carla=carla,np=numpy,math=math)
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
    return module


class OfficialExpert:
    """Official candidate teacher, confined to ego control for data collection."""
    def __init__(self, official_root, *, seed=1, cruise_ratio=None):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Expert private seed must be an unsigned 32-bit integer')
        if cruise_ratio is not None and (not _number(cruise_ratio) or not 0<cruise_ratio<=1.):
            raise ValueError('Explicit offline expert cruise ratio must be finite in (0,1]')
        self.root = filesystem_path(official_root)
        self.seed = seed
        self._requested_cruise_ratio = cruise_ratio
        self._identity = source_identity(self.root)
        self._pilot = self._provider = self._carla = None
        self._last_frame = self._last_time = None
        self._last_game_time = None
        self._calls, self._closed = 0, False
        self._guard_complete = False
        self._failures, self._mutation_attempts = [], []
        self._cruise_policy = None
        self._last_decision = None

    def _reject_npc_deletion(self, actor):
        self._mutation_attempts.append(dict(method='actor.destroy',actor_id=actor.id,
                                           action='rejected_without_deleting_actor'))
        return _no_npc_deletion(actor)

    def setup(self, global_plan_gps, dense_plan):
        if self._pilot is not None or self._closed:
            raise RuntimeError('Expert setup may run exactly once')
        if len(dense_plan) < 2 or len(global_plan_gps) != len(dense_plan):
            raise ValueError('Expert requires the complete official GPS and dense world plan')
        # These are existing SDK-environment dependencies, not installations.
        import carla
        import numpy as np
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        from leaderboard.autoagents import autonomous_agent_local
        loaded_base = filesystem_path(autonomous_agent_local.__file__)
        if hashlib.sha256(loaded_base.read_bytes()).hexdigest() != LOCAL_BASE_SHA256:
            raise ValueError('Imported local agent base differs from the pinned expert base')
        code = self.root/'team_code'
        private_environment = dict(os.environ)
        private_environment.pop('SAVE_PATH',None)
        private_environment.update(DATAGEN='0',HISTOGRAM='0',TP_STATS='0',DEBUG_CHALLENGE='0')
        private_environment['SCENARIO_RUNNER_ROOT'] = str(self.root/'scenario_runner')
        private_os = SimpleNamespace(environ=private_environment,path=os.path,listdir=os.listdir)
        private_np = _NumpyView(np,self.seed)
        modules = {name:_load_module(name,code/(name+'.py'),private_os) for name in (
            'config','nav_planner','lateral_controller','privileged_route_planner',
            'longitudinal_controller','kinematic_bicycle_model')}
        self._provider = _ProviderView(CarlaDataProvider)
        module = ModuleType('_ep_f2d_offline_pinned_autopilot')
        path = code/'autopilot.py'
        module.__file__ = str(path)
        module.__dict__.update(os=private_os,ujson=json,np=private_np,
            CarlaDataProvider=self._provider,t_u=_load_utilities(code/'transfuser_utils.py',carla,np),
            RoutePlanner=modules['nav_planner'].RoutePlanner,
            LateralPIDController=modules['lateral_controller'].LateralPIDController,
            PrivilegedRoutePlanner=modules['privileged_route_planner'].PrivilegedRoutePlanner,
            GlobalConfig=modules['config'].GlobalConfig,
            ScenarioLogger=_disabled_recorder,
            LongitudinalLinearRegressionController=modules['longitudinal_controller'].LongitudinalLinearRegressionController,
            KinematicBicycleModel=modules['kinematic_bicycle_model'].KinematicBicycleModel,
            _no_npc_deletion=self._reject_npc_deletion,_disabled_recorder=_disabled_recorder,
            _decision_record=_decision_record,_ep_cautious_target=_ep_cautious_target,
            _maneuver_record=_maneuver_record,_ep_before_opposing_entry=_ep_before_opposing_entry,
            _ep_cautious_gap_time=_ep_cautious_gap_time)
        exec(compile(_expert_tree(path.read_text(encoding='utf-8'),str(path)),str(path),'exec'),module.__dict__)
        pilot = module.AutoPilot(None)
        pilot.set_global_plan(global_plan_gps,dense_plan)
        pilot.setup('offline_official_demonstration')
        # The upstream anti-blocking full-throttle action is inappropriate for
        # a demonstration waiting behind an active 60s blockage. It is disabled
        # only for this teacher, never added to the paper planner.
        pilot.config.max_blocked_ticks = 2**63-1
        pilot.config.steer_noise = 0.
        pilot._ep_preentry_gap_rechecks = 0
        pilot._ep_committed_maneuvers = set()
        pilot._ep_entry_commit_evidence = {}
        pilot._ep_gap_time_estimates = []
        from .demonstration_policy import CautiousDemonstrationPolicy
        pilot._ep_caution_policy = CautiousDemonstrationPolicy()
        transition_keys=('transition_length_construction_obstacle_two_ways',
                         'transition_length_accident_two_ways',
                         'transition_length_parked_obstacle_two_ways',
                         'transition_length_vehicle_opens_door_two_ways',
                         'transition_smoothness_factor_construction_obstacle')
        transitions={}
        for name in transition_keys:
            previous=float(getattr(pilot.config,name))
            applied=max(previous,10.*pilot.config.points_per_meter)
            setattr(pilot.config,name,int(applied))
            transitions[name]=dict(original_points=previous,applied_points=int(applied),
                original_m=previous/pilot.config.points_per_meter,
                applied_m=applied/pilot.config.points_per_meter)
        self._route_policy = dict(schema=DRIVING_POLICY_SCHEMA,
            points_per_meter=pilot.config.points_per_meter,construction_transitions=transitions,
            hazard_bypass_removed=True,preentry_opposing_gap='recheck_every_frame_in_activation_window',
            committed_maneuver='do_not_stop_from_cached_gap_alone;current_actor_and_traffic_checks_remain',
            scope='offline_demonstration_only;private_ego_reference_not_official_route_or_NPC_mutation')
        # The NPY map limit has upstream preprocessing and can differ from the
        # vehicle's live limit. Preserve the original 72% default; changing the
        # offline cruise fraction must be explicit, never score-driven.
        upstream_ratio=float(pilot.config.ratio_target_speed_limit)
        selected_ratio=(upstream_ratio if self._requested_cruise_ratio is None else float(self._requested_cruise_ratio))
        self._cruise_policy = dict(upstream_ratio=upstream_ratio,adapted_ratio=selected_ratio,
            selection='pinned_default' if self._requested_cruise_ratio is None else 'explicit_offline_argument',
            maximum_target_speed_mps=20.,maximum_junction_speed_mps=float(pilot.config.max_speed_in_junction),
            hazard_and_idm_reductions='always_executed_after_nonincreasing_cautious_target',score_feedback_used=False,
            full_score_guaranteed=False,scope='offline_demonstration_only',
            npy_limit_is_unmodified_legal_limit_claimed=False)
        pilot.config.ratio_target_speed_limit = selected_ratio
        self._pilot,self._carla = pilot,carla

    def step(self, input_data, timestamp):
        try:
            return self._step(input_data,timestamp)
        except Exception as exc:
            self._failures.append(dict(stage='step',error=type(exc).__name__,message=str(exc)))
            raise

    def _step(self, input_data, timestamp):
        if self._pilot is None or self._closed:
            raise RuntimeError('Expert is not set up or has already closed')
        if not _number(timestamp):
            raise ValueError('Expert source timestamp must be finite')
        imu = input_data.get('imu')
        if not isinstance(imu,(list,tuple)) or len(imu) != 2 or type(imu[0]) is not int:
            raise ValueError('Expert requires an actual same-frame IMU sample')
        frame = imu[0]
        world = self._provider.get_world()
        snapshot = world.get_snapshot()
        if snapshot.frame != frame:
            raise ValueError('Expert IMU and world source frame differ')
        world_time = float(snapshot.timestamp.elapsed_seconds)
        if not _number(world_time):
            raise ValueError('Expert frozen world timestamp must be finite')
        raw = imu[1]
        try:
            compass = float(raw[-1])
        except (TypeError,ValueError,IndexError) as exc:
            raise ValueError('Expert IMU has no finite raw compass') from exc
        if not math.isfinite(compass):
            raise ValueError('Expert IMU compass is not finite')
        if self._last_frame is not None and (frame != self._last_frame+1
                or abs(world_time-self._last_time-.05) > 1e-5
                or abs(timestamp-self._last_game_time-.05) > 1e-5):
            raise ValueError('Expert source frames/times must be consecutive official 20Hz')
        self._provider.refresh()
        try:
            control = self._pilot.run_step(input_data,timestamp)
        finally:
            self._provider.finish()
        if world.get_snapshot().frame != frame:
            raise RuntimeError('World advanced during expert control calculation')
        if (not isinstance(control,self._carla.VehicleControl)
                or any(not _number(getattr(control,name,None)) for name in ('throttle','brake','steer'))
                or not 0 <= control.throttle <= 1 or not 0 <= control.brake <= 1 or not -1 <= control.steer <= 1
                or control.throttle > 0 and control.brake > 0 or control.hand_brake or control.reverse):
            raise ValueError('Official expert returned an invalid ego control')
        self._last_frame,self._last_time,self._calls = frame,world_time,self._calls+1
        self._last_game_time = float(timestamp)
        decision=getattr(self._pilot,'_ep_last_decision',None)
        self._last_decision=(dict(decision,schema=DECISION_SCHEMA,source_frame=frame,
            source_time=world_time,official_game_time=float(timestamp),speed_unit='m/s',
            speed_semantics='CARLA.get_velocity().length_xyz',
            archived_ego_velocity_semantics='world_xy_m_per_s',
            caution_policy=_container_copy(getattr(self._pilot,'_ep_caution_decision',None)),
            maneuver_policy=_container_copy(getattr(self._pilot,'_ep_maneuver_decision',None))) if decision is not None else None)
        return control

    def decision_evidence(self):
        """Return a copy of the last successful source-bound scalar decision."""
        return _json_decision_copy(self._last_decision) if self._last_decision is not None else None

    def evidence(self):
        return dict(self._identity,teacher='pinned_official_AutoPilot_with_confined_offline_adapter',
            complete=self._guard_complete,world_mutations_allowed=False,
            non_ego_mutation_attempts=list(self._mutation_attempts),failures=list(self._failures),
            scope='ego_control_only',
            enforcement='pinned_source_AST_call_review_and_explicit_NPC_cleanup_rejection;not_a_CARLA_runtime_sandbox',
            formal_evaluation_control=False,success_certified=False,private_seed=self.seed,
            source_frame=self._last_frame,source_time=self._last_time,control_calls=self._calls,
            official_game_time=self._last_game_time,
            adaptations=dict(npc_cleanup='reject_instead_of_destroy',scenario_bookkeeping='private_container_copy',
                numpy_randomness='private_RandomState_not_global',steer_noise=0.,
                cruise_target_speed=dict(self._cruise_policy) if self._cruise_policy is not None else None,
                cautious_driving=dict(self._route_policy) if hasattr(self,'_route_policy') else None,
                decision_logging='finite_existing_scalars;XYZ_expert_speed_distinct_from_XY_archive_speed',
                anti_blocking_forced_throttle='disabled',official_datagen=False,official_logger=False,
                official_recorder=False,utilities='six_pinned_pure_functions_no_torch',
                serialization='stdlib_json_for_disabled_official_logger',
                environment='private_facade_pinned_scenario_runner_root',
                base_class='upstream_local_AutonomousAgent_setup_and_plan_hooks'),
            uses_privileged_scenario_bookkeeping=True,expert_bookkeeping_is_model_input=False,
            world_or_npc_mutation_calls_allowed=False,official_source_files_modified=False)

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                if self._pilot is not None:
                    self._pilot.destroy()
            except Exception as exc:
                self._failures.append(dict(stage='close',error=type(exc).__name__,message=str(exc)))
                raise
            else:
                self._guard_complete = True
