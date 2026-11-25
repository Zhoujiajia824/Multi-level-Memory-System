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
            # Keep all existing route, hazard, IDM and actuator calculations.
            # Read their final local scalars immediately before the sole return.
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
    if patched != 1:
        raise ValueError('Unexpected NPC cleanup shape in pinned expert')
    if decision_patches != 1:
        raise ValueError('Unexpected decision recording shape in pinned expert')
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
            _decision_record=_decision_record)
        exec(compile(_expert_tree(path.read_text(encoding='utf-8'),str(path)),str(path),'exec'),module.__dict__)
        pilot = module.AutoPilot(None)
        pilot.set_global_plan(global_plan_gps,dense_plan)
        pilot.setup('offline_official_demonstration')
        # The upstream anti-blocking full-throttle action is inappropriate for
        # a demonstration waiting behind an active 60s blockage. It is disabled
        # only for this teacher, never added to the paper planner.
        pilot.config.max_blocked_ticks = 2**63-1
        pilot.config.steer_noise = 0.
        # The NPY map limit has upstream preprocessing and can differ from the
        # vehicle's live limit. Preserve the original 72% default; changing the
        # offline cruise fraction must be explicit, never score-driven.
        upstream_ratio=float(pilot.config.ratio_target_speed_limit)
        selected_ratio=(upstream_ratio if self._requested_cruise_ratio is None else float(self._requested_cruise_ratio))
        self._cruise_policy = dict(upstream_ratio=upstream_ratio,adapted_ratio=selected_ratio,
            selection='pinned_default' if self._requested_cruise_ratio is None else 'explicit_offline_argument',
            maximum_target_speed_mps=20.,maximum_junction_speed_mps=float(pilot.config.max_speed_in_junction),
            hazard_and_idm_reductions='unchanged',score_feedback_used=False,
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
            archived_ego_velocity_semantics='world_xy_m_per_s') if decision is not None else None)
        return control

    def decision_evidence(self):
        """Return a copy of the last successful source-bound scalar decision."""
        return dict(self._last_decision) if self._last_decision is not None else None

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
