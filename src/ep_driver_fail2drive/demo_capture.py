"""Explicitly invoked offline demonstration collector, not a benchmark Agent.

No module-level CARLA import, connection, tick, model call or memory write.
The caller owns a dedicated idle simulator. Only owned actors are destroyed.
"""
import hashlib
import json
import logging
import math
from pathlib import Path
import queue
import time
import traceback

from .archive import RouteArchive, _png
from .demo_catalog import capture_implementation
from .demo_events import DemoEvents, NoWalkableCrossingSurface, requires_borrow_proof, verify_borrow_capture
from .demo_routes import _bypass_stations, select_demonstration_site, teaching_path, validate_teaching_road_path
from .demo_teacher import DemoTeacher, SCHEMA as TEACHER_SCHEMA
from .execution import SignedAccelerationController
from .public_state import PublicStateExtractor
from .route import xy, point_in_polygon, project, lane_key
from .vehicle_geometry import measure_vehicle_geometry
from .camera_calibration import source_camera_calibration
from nine_grid.paths import filesystem_path

HZ = 20
LOG = logging.getLogger(__name__)
VIEWS = (('front_rgb', 0), ('front_left_rgb', -60), ('front_right_rgb', 60),
         ('back_rgb', 180), ('back_left_rgb', -120), ('back_right_rgb', 120))


class InteractionCaptureWindow:
    """Observed short-recovery tail, independent of benchmark completion.

    Completion does not generate controls: an actual event must finish, the
    car must pass its position and return from an authored bypass, and four
    actual 1Hz source observations must cover a continuous recovery interval.
    """
    SCHEMA='independent-demo-interaction-segment-1.0'

    def __init__(self,spec,waypoints,actor_rows):
        self.route=[xy(wp.transform.location) for wp in waypoints]
        self.lanes=[lane_key(wp) for wp in waypoints]
        interaction_rows=[row for row in actor_rows if row['config'].get('role') not in
                          ('oncoming_vehicle', 'construction_advance_warning')]
        stations=[float(row['interaction_station_m']) for row in interaction_rows]
        if not stations or not all(math.isfinite(s) for s in stations):
            raise ValueError('Interaction collection requires actual finite hazard stations')
        self.gate=max(stations)+5.
        if spec['interaction_kind'] in ('parked_bypass','construction_bypass','construction_crossing'):
            route_spec=dict(spec,actor_specs=[row['config'] for row in interaction_rows])
            self.gate=max(self.gate,_bypass_stations(route_spec)[-1])
        self.start = None
        self.anchor = None
        self.sources = []
        self.complete = False
        self.last = None

    def update(self,scene,event_complete,*,saved_source,audit):
        if self.complete:raise ValueError('Interaction collection already has a completed recovery tail')
        frame=scene['diagnostics']['carla_frame'];stamp=scene['time']
        if type(frame) is not int or type(stamp) not in (int,float) or not math.isfinite(stamp):
            raise ValueError('Invalid actual recovery source frame/time')
        if self.last is not None and (frame!=self.last['frame']+1 or abs(stamp-self.last['time']-1/HZ)>1e-5):
            raise ValueError('Interaction recovery requires consecutive actual physical sources')
        observed=scene['ego'];speed=math.hypot(*observed['velocity'])
        projection=project(observed['position'],self.route,heading=observed['heading'])
        i=projection['segment_index']
        lane_match=observed.get('lane_id') in self.lanes[max(0,i-1):min(len(self.lanes),i+3)]
        valid=(scene['diagnostics']['route_projection'].get('valid') is True and audit.get('projection_valid') is True)
        progress=scene['diagnostics']['route_projection'].get('trusted_progress_m')
        finite_progress=type(progress) in (int,float) and math.isfinite(progress)
        safe=all(audit.get(k) is False for k in ('collision','offroad','rule_violation'))
        past_gate=finite_progress and progress>=self.gate
        stable=(event_complete and past_gate and valid and safe and lane_match and speed>=.5
                and projection['distance_m']<=1. and projection['heading_error_rad']<=.35)
        evidence=dict(source_frame=frame,source_time=stamp,interaction_complete=bool(event_complete),
            progress_m=progress,pass_gate_s_m=self.gate,past_gate=bool(past_gate),projection_valid=valid,
            route_lane_match=lane_match,projection_distance_m=projection['distance_m'],
            heading_error_rad=projection['heading_error_rad'],speed_mps=speed,stable_recovery=bool(stable))
        events=[];self.last=dict(frame=frame,time=stamp)
        if stable:
            if self.start is None:
                self.start=dict(frame=frame,time=stamp)
                events.append(dict(phase='recovery',event='recovery_started',completed=False,
                    observed_progress_m=progress,pass_gate_s_m=self.gate))
            if saved_source:
                if self.sources and frame - self.sources[-1]['frame'] != HZ:
                    self.sources.clear()
                if not self.sources:
                    self.anchor = dict(frame=frame,time=stamp)
                self.sources.append(dict(frame=frame,time=stamp))
                if len(self.sources) == 4:
                    self.complete = True
                    events.append(dict(phase='recovery',event='recovery_completed',completed=True,
                        observed_progress_m=progress,pass_gate_s_m=self.gate,
                        recovery_anchor_frame=self.anchor['frame'],recovery_anchor_time=self.anchor['time'],
                        future_source_frames=[row['frame'] for row in self.sources],
                        future_source_times=[row['time'] for row in self.sources]))
        elif self.start is not None:
            events.append(dict(phase='recovery',event='recovery_interrupted',completed=False,
                observed_progress_m=progress,pass_gate_s_m=self.gate))
            self.start=self.anchor=None;self.sources=[]
        return evidence,events

    def report(self):
        def field(row,key):return None if row is None else row[key]
        return dict(schema=self.SCHEMA,complete=self.complete,pass_gate_s_m=self.gate,
            recovery_start_frame=field(self.start,'frame'),recovery_start_time=field(self.start,'time'),
            recovery_anchor_frame=field(self.anchor,'frame'),recovery_anchor_time=field(self.anchor,'time'),
            future_source_frames=[r['frame'] for r in self.sources],future_source_times=[r['time'] for r in self.sources],
            end_frame=field(self.last,'frame'),end_time=field(self.last,'time'),
            official_score=False,scope='independent_interaction_segment')


def atomic_json(path, value):
    target = Path(path)
    temporary = target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)


class FrameBarrier:
    """Exact frames only, including per-image timestamp and raw transform."""
    def __init__(self, names):
        if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError('Camera barrier requires distinct nonempty camera names')
        self.queues = {name: queue.Queue(maxsize=4) for name in names}
        self.errors = []
        self.last_frame, self.last_stamp = None, None

    def callback(self, name):
        if name not in self.queues:
            raise ValueError('Unregistered camera: '+str(name))
        def receive(image):
            try:
                self.queues[name].put_nowait(image)
            except queue.Full:
                self.errors.append('Camera queue overflow: '+name)
        return receive

    def get(self, frame, stamp, timeout=30.):
        number = lambda value: type(value) in (int, float) and math.isfinite(value)
        if (type(frame) is not int or frame < 0 or not number(stamp)
                or not number(timeout) or timeout <= 0):
            raise ValueError('Requested camera frame, time and timeout must be finite and valid')
        if self.last_frame is not None and (frame != self.last_frame+1 or stamp <= self.last_stamp):
            raise ValueError('Camera barrier requests must use consecutive increasing source frames')
        deadline, result = time.monotonic()+timeout, {}
        for name, rows in self.queues.items():
            if self.errors:
                raise RuntimeError(self.errors[0])
            try:
                image = rows.get(timeout=max(.001, deadline-time.monotonic()))
            except queue.Empty as exc:
                raise TimeoutError('Missing exact camera frame: '+name+' '+str(frame)) from exc
            if (type(image.frame) is not int or image.frame != frame or not number(image.timestamp)
                    or abs(image.timestamp-stamp) > 1e-6):
                raise ValueError('Camera/world frame or timestamp mismatch: '+name)
            try:
                matrix = image.transform.get_matrix()
                valid = (len(matrix) == 4 and all(len(row) == 4 and all(number(x) for x in row) for row in matrix))
            except (AttributeError, TypeError):
                valid = False
            if not valid:
                raise ValueError('Camera lacks a finite raw image transform: '+name)
            result[name] = image
        if self.errors:
            raise RuntimeError(self.errors[0])
        self.last_frame, self.last_stamp = frame, stamp
        return result


def _control(control):
    return {key: getattr(control, key) for key in
            ('throttle', 'brake', 'steer', 'hand_brake', 'reverse', 'manual_gear_shift', 'gear')}


def _body_audit(scene, previous, collisions, line_events):
    """Independent current-state audit; unknown evidence cannot certify success.

    Body coverage uses a <=0.25m grid, not a continuous-geometry proof. Sensor
    events retain their true source frames even when their callbacks arrive late.
    """
    number = lambda value: type(value) in (int, float) and math.isfinite(value)
    errors, reasons, rule_evidence = [], [], []
    try:
        ego, rules, diagnostics = scene['ego'], scene['rules'], scene['diagnostics']
        frame = diagnostics['carla_frame']
        if type(frame) is not int or frame < 0:
            raise ValueError('invalid physical source frame')
        def body(row):
            values = [*row['position'], row['heading'], row['length'], row['width']]
            if len(row['position']) != 2 or not all(number(v) for v in values) or min(row['length'],row['width']) <= 0:
                raise ValueError('invalid measured body')
            c, s = math.cos(row['heading']), math.sin(row['heading'])
            def at(u, v):
                return [row['position'][0]+u*c-v*s,row['position'][1]+u*s+v*c]
            return at
        at = body(ego)
        nx, ny = max(2,math.ceil(ego['length']/.25)), max(2,math.ceil(ego['width']/.25))
        samples = [at(ego['length']*(i/nx-.5), ego['width']*(j/ny-.5))
                   for i in range(nx+1) for j in range(ny+1)]
        samples.append(list(ego['position']))
        polygons = scene['drivable_polygons']
        if not polygons:
            raise ValueError('missing drivable polygons')
        outside = [p for p in samples if not any(point_in_polygon(p, poly) for poly in polygons)]
        offroad = bool(outside)
        road_coverage = dict(source='same_frame_drivable_polygon_union',source_frame=frame,
            ego_lane_id=ego.get('lane_id'),outside_sample_count=len(outside),
            outside_sample_limit=8,outside_samples=[],
            outside_distance_min_m=None,outside_distance_max_m=None)
        if outside:
            # Distance is diagnostic only: the existing exact inclusion test
            # above decides offroad, with no road enlargement or grace margin.
            def boundary_distance(point):
                distances = []
                for polygon in polygons:
                    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                        dx,dy=b[0]-a[0],b[1]-a[1]
                        norm=dx*dx+dy*dy
                        t=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/norm)) if norm else 0.
                        distances.append(math.hypot(point[0]-a[0]-t*dx,point[1]-a[1]-t*dy))
                return min(distances) if distances else None
            measured=[dict(position_xy_m=point,distance_to_drivable_m=boundary_distance(point)) for point in outside]
            finite=[row['distance_to_drivable_m'] for row in measured if row['distance_to_drivable_m'] is not None]
            road_coverage['outside_samples']=sorted(measured,
                key=lambda row:row['distance_to_drivable_m'] or 0.,reverse=True)[:8]
            if finite:
                road_coverage.update(outside_distance_min_m=min(finite),outside_distance_max_m=max(finite))
        current_events = []
        for event in [*collisions,*line_events]:
            if type(event.get('frame')) is not int or event['frame'] > frame or event['frame'] < 0:
                errors.append('sensor event has invalid or future source frame')
            else:
                current_events.append(event)
        collision_frames = sorted({row['frame'] for row in collisions if row in current_events})
        reasons.extend(row['kind'] for row in line_events if row in current_events)
        if rules.get('navigation_input_valid') is False or rules.get('unmapped_current_red'):
            errors.append('current public rule or navigation association unresolved')
        if rules.get('public_road_rules',{}).get('unmapped_current_stops'):
            errors.append('current stop-sign association unresolved')
        if number(rules.get('speed_limit')):
            velocity = ego.get('velocity')
            if not isinstance(velocity,(tuple,list)) or len(velocity)!=2 or not all(number(v) for v in velocity):
                errors.append('missing finite observed velocity for speed audit')
            elif math.hypot(*velocity) > rules['speed_limit']+.2:
                reasons.append('speed_limit_exceeded')
        else:
            errors.append('missing finite public speed limit')
        if previous is not None:
            old = previous['ego']
            if previous['diagnostics']['carla_frame'] != frame-1:
                errors.append('physical audit sources are not consecutive')
            old_at = body(old)
            def front_station(row, transform):
                trusted = (scene if row is ego else previous)['diagnostics']['route_projection'].get('trusted_progress_m')
                return max(project(transform(row['length']/2,v), scene['route'],
                    previous_s=trusted, window=row['length']+6 if trusted is not None else None,
                    heading=row['heading'])['s_m'] for v in (-row['width']/2,row['width']/2))
            before, now = front_station(old,old_at),front_station(ego,at)
            lanes = {row['id']:row for row in scene.get('lanes',[])}
            permitted_lanes = {ego.get('lane_id')}
            old_lane = old.get('lane_id')
            if ego.get('lane_id') in lanes.get(old_lane,{}).get('successors',[]):
                permitted_lanes.add(old_lane)
            current_lights = {row['id']:row for row in rules.get('traffic_lights',[])}
            prior_lights = {row['id']:row for row in previous['rules'].get('traffic_lights',[])}
            for identity in current_lights.keys() | prior_lights.keys():
                light = current_lights.get(identity)
                old_light = prior_lights.get(identity,{})
                stops = (light or {}).get('stops',[]) + old_light.get('stops',[])
                checked = set()
                for stop in stops:
                    if stop.get('lane_id') not in permitted_lanes:
                        continue
                    station = stop.get('task_stop_s')
                    if not number(station) and isinstance(stop.get('position'),(list,tuple)):
                        station = project(stop['position'],scene['route'],heading=stop.get('heading_rad'))['s_m']
                    if not number(station):
                        errors.append('controlled traffic-light stop lacks finite public geometry')
                        continue
                    key = (stop.get('lane_id'),round(station,5))
                    if key in checked:
                        continue
                    checked.add(key)
                    if not before <= station < now:
                        continue
                    state = light.get('state') if light else None
                    source = light.get('source_frame') if light else None
                    rule_evidence.append(dict(kind='traffic_light_crossing',id=identity,
                        lane_id=stop.get('lane_id'),stop_s=station,current_state=state,source_frame=source,
                        mapped_stop=number(stop.get('task_stop_s')),position_xyz_carla=stop.get('position_xyz_carla'),
                        stop_heading_rad=stop.get('heading_rad')))
                    if source != frame or state not in ('red','yellow','green','off'):
                        errors.append('crossed light lacks resolved current-frame state')
                    elif state in ('red','yellow'):
                        reasons.append('active_traffic_light_crossing')
            current_stops = {row['id']:row for row in rules.get('stop_sign_obligations',[])}
            prior_stops = {row['id']:row for row in previous['rules'].get('stop_sign_obligations',[])}
            for identity in current_stops.keys() | prior_stops.keys():
                obligation = current_stops.get(identity)
                reference = obligation or prior_stops[identity]
                if reference.get('lane_id') not in permitted_lanes:
                    continue
                station = reference.get('stop_s')
                if not number(station):
                    errors.append('stop-sign obligation lacks finite stop geometry')
                    continue
                if reference.get('status') == 'passed_without_observed_stop':
                    reasons.append('stop_sign_passed_without_observed_stop')
                if not before <= station < now:
                    continue
                if obligation is None or obligation.get('source_frame') != frame:
                    errors.append('crossed stop sign lacks current-frame obligation evidence')
                elif obligation.get('must_stop') is True:
                    reasons.append('unfulfilled_stop_sign_crossing')
                elif obligation.get('status') != 'fulfilled':
                    errors.append('crossed stop sign has unknown fulfilment status')
            # Aggregate stop_s alone is not enough to infer a light identity or
            # to replay yesterday's red state after the current signal is green.
            if not current_lights and not current_stops and rules.get('stop_s') is not None:
                errors.append('active aggregate stop has no auditable rule identity')
        return dict(source_frame=frame,collision=bool(collision_frames),collision_source_frames=collision_frames,
            late_sensor_event_frames=sorted({row['frame'] for row in current_events if row['frame']<frame}),
            offroad=offroad,rule_violation=bool(reasons),rule_reasons=sorted(set(reasons)),
            rule_evidence=rule_evidence,body_sample_count=len(samples),body_sample_spacing_max_m=.25,
            road_coverage=road_coverage,
            audit_input_valid=not errors,audit_errors=errors,
            projection_valid=bool(diagnostics['route_projection']['valid']) and not errors)
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        return dict(source_frame=frame if 'frame' in locals() and type(frame) is int else None,
            collision=bool(collisions),offroad=False,rule_violation=False,rule_reasons=[],
            projection_valid=False,audit_input_valid=False,audit_errors=['invalid physical audit input: '+str(exc)])


def _archive_failure_scene(archive,scene,audit):
    """Retain the last frozen state, including failures between 1Hz archives."""
    if scene is None:
        return
    frame=scene['diagnostics']['carla_frame']
    archive.json('failure_scene.json',dict(source_frame=frame,source_time=scene['time'],scene=scene,
        audit=audit if isinstance(audit,dict) and audit.get('source_frame')==frame else None,
        evidence='last_frozen_public_state;not_a_new_observation',memory_published=False))


def _world_preflight(world):
    """Same exclusivity predicate as capture; all listed actors predate this attempt."""
    synchronous = bool(world.get_settings().synchronous_mode)
    counts = {'vehicle': 0, 'walker': 0, 'sensor': 0}
    examples = []
    for actor in world.get_actors():
        kind, sep, _ = actor.type_id.partition('.')
        if not sep or kind not in counts:
            continue
        counts[kind] += 1
        if len(examples) < 8:
            examples.append(dict(id=getattr(actor, 'id', None), type_id=actor.type_id,
                role_name=getattr(actor, 'attributes', {}).get('role_name', '')))
    count = sum(counts.values())
    return dict(idle=not synchronous and count == 0, synchronous_mode=synchronous,
        blocking_actor_count=count, blocking_actor_counts=counts, blocking_actor_examples=examples)


def _destroy_owned_ids(client, carla, ids, already_destroyed):
    """Destroy registered actors even when their spawn is newer than the last snapshot."""
    ids = list(dict.fromkeys(ids))
    pending = [actor_id for actor_id in ids if actor_id not in already_destroyed]
    report = dict(owned_ids=ids, previously_destroyed_ids=[i for i in ids if i in already_destroyed],
                  requested_ids=pending, acknowledged_ids=[], errors=[])
    if not pending:
        return report
    try:
        responses = client.apply_batch_sync([carla.command.DestroyActor(i) for i in pending], False)
        for index, actor_id in enumerate(pending):
            error = responses[index].error if index < len(responses) else 'missing server response'
            if error:
                report['errors'].append(f'DestroyActor({actor_id}): {error}')
            else:
                report['acknowledged_ids'].append(actor_id)
    except Exception as exc:
        report['errors'].append(f'DestroyActor{pending}: {exc}')
    return report


def capture(spec, output_dir, *, host='127.0.0.1', port=39000, allow_map_load=False, video=True,
            interaction_only=False, allow_similar_road=False):
    if type(interaction_only) is not bool:
        raise ValueError('Interaction-only collection selection must be explicitly boolean')
    import carla
    import numpy as np
    output = filesystem_path(output_dir)
    archive = RouteArchive(output, video=video, fps=HZ)
    (output/'native').mkdir()
    run = dict(protocol='ep-driver-f2d-demo-capture-1.0', id=output.name,
        demo_suite='fail2drive-targeted-demonstrations-1.0', scenario_family=spec['family'],
        scenario=spec['id'], source='carla', source_version=None, split='independent_development',
        reference_driver=TEACHER_SCHEMA, seed=spec['seed'], test_only=False,
        complete=False, official_evaluation=False, record_hz=1, physics_hz=HZ,
        capture_mode='interaction_segment' if interaction_only else 'full_route',
        frames_path='frames.jsonl', model_calls=0, retrieval_enabled=False,
        read_future_during_current_representation=False, parameters=spec,
        weather_preset='ClearNoon',seed_purpose='independent_site_order;not_an_engine_determinism_claim')
    run['implementation_sha256'] = capture_implementation()
    archive.json('run.json', run)
    owned, sensors, world, saved, saved_weather, events = [], [], None, None, None, None
    restored, complete, reached = False, False, False
    collisions, line_events, errors = [], [], []
    audit_rows, recovery_complete, geometry = [], False, None
    interaction_window=None
    preflight = None
    cleanup_reports, destroyed_ids = [], set()
    def release_site(stage):
        event_actors = [] if events is None else events.actors
        if events is not None:
            destroyed_ids.update(events.released_actor_ids)
            try:
                events.close(destroy_actors=False)
            except Exception as exc:
                errors.append('restore event signals: '+str(exc))
        ids = [actor.id for actor in (*reversed(owned), *reversed(event_actors))]
        if not ids:
            return
        report = _destroy_owned_ids(client, carla, ids, destroyed_ids)
        report['stage'] = stage
        cleanup_reports.append(report)
        destroyed_ids.update(report['acknowledged_ids'])
        errors.extend(report['errors'])

    def startup(operation, timeout_s=30., status='entered', **details):
        # Survives a native SDK abort, which bypasses Python exception handling.
        atomic_json(output/'startup_stage.json', dict(operation=operation, timeout_s=timeout_s,
            status=status, host=host, port=int(port), **details))

    try:
        startup('connect_and_check_world')
        client = carla.Client(host, int(port))
        client.set_timeout(30.)
        if client.get_client_version() != client.get_server_version():
            raise RuntimeError('CARLA client and server versions differ')
        run['source_version'] = client.get_server_version()
        world = client.get_world()
        preflight = _world_preflight(world)
        if not preflight['idle']:
            raise RuntimeError('Demonstration collection requires an idle dedicated world: '
                'synchronous_mode={synchronous_mode}, pre-existing actors={blocking_actor_count} '
                '(vehicles={vehicle}, walkers={walker}, sensors={sensor}).'.format(
                    **preflight, **preflight['blocking_actor_counts']))
        if world.get_map().name.split('/')[-1] != spec['map']:
            if not allow_map_load:
                raise RuntimeError('Required map '+spec['map']+'; explicitly allow an idle-world map load')
            # Town13 cold loads use the official evaluator's 300 s allowance.
            # A timed-out load can still finish server-side; issue it only once.
            load_started = time.monotonic()
            run['map_load'] = dict(map=spec['map'], timeout_s=300., status='loading')
            archive.json('map_load.json', run['map_load'])
            startup('load_world', 300.)
            LOG.info('加载地图 %s，最多等待300秒', spec['map'])
            client.set_timeout(300.)
            try:
                world = client.load_world(spec['map'])
                actual_map = world.get_map().name.split('/')[-1]
                if actual_map != spec['map']:
                    raise RuntimeError('Loaded map differs from requested map: '+actual_map)
                run['map_load']['status'] = 'ready'
            except Exception as exc:
                run['map_load'].update(status='failed', error=str(exc))
                raise RuntimeError('Map load '+spec['map']+' failed: '+str(exc)) from exc
            finally:
                client.set_timeout(30.)
                run['map_load']['wall_seconds'] = time.monotonic()-load_started
                archive.json('map_load.json', run['map_load'])
            LOG.info('地图 %s 已就绪，加载耗时 %.1f 秒', spec['map'], run['map_load']['wall_seconds'])
        startup('configure_world')
        saved = world.get_settings()
        saved_weather = world.get_weather()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1/HZ
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = .01
        settings.max_substeps = 5
        large_map = spec['map'] == 'Town13'
        if large_map:
            settings.tile_stream_distance = 650.
            settings.actor_active_distance = 650.
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        unusable_site = None
        while True:
            startup('select_site')
            waypoints, site = select_demonstration_site(world, spec, allow_similar_road=allow_similar_road,
                                                       unusable_site=unusable_site)
            run['selected_site'] = site
            authored = teaching_path(waypoints, spec)
            opposing_borrow = spec['teacher_policy'].get('allow_opposite_lane', False)
            early_authored = teaching_path(waypoints, spec, early_return=True) if opposing_borrow else None
            route = [xy(wp.transform.location) for wp in waypoints]
            archive.json('demonstration_design.json', dict(spec=spec, route_xy=route, teaching_path_xy=authored,
                early_return_path_xy=early_authored,
                source=site['mode'], selected_site=site, collection_route_reference=spec.get('route_source'),
                official_route_used=False, official_event_replayed=False,
                formal_evaluation_control=False, runtime_success_not_yet_verified=True))
            spawn_timeout = 120. if large_map else 30.
            if large_map:
                client.set_timeout(spawn_timeout)
            try:
                startup('prepare_ego_spawn', spawn_timeout)
                bp = world.get_blueprint_library().find('vehicle.tesla.model3')
                bp.set_attribute('role_name', 'hero' if large_map else 'offline_demonstration_ego')
                start = waypoints[0].transform
                pose = carla.Transform(carla.Location(x=start.location.x,y=start.location.y,z=start.location.z+.4),
                                       carla.Rotation(pitch=start.rotation.pitch,yaw=start.rotation.yaw,roll=start.rotation.roll))
                startup('try_spawn_ego', spawn_timeout)
                ego = world.try_spawn_actor(bp, pose)
                if ego is None:
                    raise RuntimeError('Development ego spawn failed')
                owned.append(ego)
                startup('apply_initial_brake', spawn_timeout, ego_actor_id=ego.id)
                ego.apply_control(carla.VehicleControl(brake=1.))
                if large_map:
                    startup('register_streaming_hero', spawn_timeout, ego_actor_id=ego.id)
                    world.tick(spawn_timeout)
                startup('spawn_interaction_actors', spawn_timeout, ego_actor_id=ego.id)
                events = DemoEvents(world, waypoints, spec, carla)
                archive.json('demonstration_actors.json', events.spawn())
                break
            except NoWalkableCrossingSurface as exc:
                if (unusable_site is not None or not allow_similar_road or not spec.get('route_source')
                        or spec['family'] != 'PedestrianOtherBlocker'):
                    raise
                # No cameras or driving frames exist yet. Release only this attempt's actors.
                archive.json('walker_site_rejection.json', dict(selected_site=site, reason=str(exc),
                    walker_placements=events.walker_placements))
                release_site('replace_site')
                if errors:
                    raise RuntimeError('Failed to release actors or signals before replacing the site: '+errors[-1])
                events = None
                owned.clear()
                unusable_site = site
                LOG.info('行人横穿起点没有可验证地面，保留现场并尝试另一处已允许的同图路段')
            finally:
                if large_map:
                    client.set_timeout(30.)
        startup('register_sensors', ego_actor_id=ego.id)
        if interaction_only:
            interaction_window=InteractionCaptureWindow(spec,waypoints,events.rows)
        barrier = FrameBarrier([name for name, _ in VIEWS]+['rgb'])
        mounts = {}
        for name, yaw in (*VIEWS, ('rgb', 0)):
            camera = world.get_blueprint_library().find('sensor.camera.rgb')
            camera.set_attribute('image_size_x', '1920' if name == 'rgb' else '800')
            camera.set_attribute('image_size_y', '1080' if name == 'rgb' else '450')
            camera.set_attribute('fov', '110' if name == 'rgb' else '100')
            camera.set_attribute('sensor_tick', '0.0')
            mount = (carla.Transform(carla.Location(x=-5.5, z=3.5), carla.Rotation(pitch=-8))
                     if name == 'rgb' else carla.Transform(carla.Location(z=1.8), carla.Rotation(yaw=yaw)))
            sensor = world.spawn_actor(camera, mount, attach_to=ego)
            owned.append(sensor); sensors.append(sensor); mounts[name] = mount
            sensor.listen(barrier.callback(name))
        for kind, sink in (('sensor.other.collision', collisions), ('sensor.other.lane_invasion', line_events)):
            sensor = world.spawn_actor(world.get_blueprint_library().find(kind), carla.Transform(), attach_to=ego)
            owned.append(sensor); sensors.append(sensor)
            def receive(event, target=sink, sensor_kind=kind):
                if sensor_kind.endswith('collision'):
                    row = dict(frame=int(event.frame), time=float(event.timestamp), kind='collision', other_actor=event.other_actor.id)
                else:
                    names = [str(mark.type).split('.')[-1].lower() for mark in event.crossed_lane_markings]
                    if not any('solid' in name or name in ('curb', 'grass') for name in names):
                        return
                    row = dict(frame=int(event.frame), time=float(event.timestamp), kind='noncrossable_line', markings=names)
                target.append(row)
            sensor.listen(receive)
        last_frame = int(world.get_snapshot().frame)
        startup('await_first_observation', ego_actor_id=ego.id)
        first_time, previous, extractor, teacher, controller = None, None, None, None, None
        arrived_time, recovery_start = None, None
        previous_borrow_stage = None
        return_update_recorded = False
        while True:
            frame = int(world.tick(30.))
            snapshot = world.get_snapshot()
            if frame != last_frame+1 or snapshot.frame != frame:
                raise RuntimeError('Another client advanced the demonstration world')
            last_frame = frame
            stamp = float(snapshot.timestamp.elapsed_seconds)
            raw_images = barrier.get(frame, stamp)
            if first_time is None:
                startup('first_observation_received', status='complete', frame=frame)
            first_time = stamp if first_time is None else first_time
            elapsed = stamp-first_time
            # Integer physical-frame schedule; never infer camera identity from time.
            first_frame = frame if previous is None else first_frame
            saved_source = (frame-first_frame) % HZ == 0
            arrays = {name: (frame, np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4))
                      for name, image in raw_images.items()}
            images = archive.capture_images(arrays, frame, stamp, [name for name, _ in VIEWS], save=saved_source)
            archive.capture_display(arrays['rgb'], frame, stamp)
            if extractor is None:
                box, physics = ego.bounding_box, ego.get_physics_control()
                frozen = snapshot.find(ego.id)
                geometry = measure_vehicle_geometry(actor_transform_matrix_m=frozen.get_transform().get_matrix(),
                    wheels=physics.wheels, bbox_center_m=[box.location.x,box.location.y,box.location.z],
                    bbox_extent_m=[box.extent.x,box.extent.y,box.extent.z], bbox_heading_deg=box.rotation.yaw,
                    sensor_pose=dict(frame=frame, timestamp=stamp,
                        camera_world_matrix_m=raw_images['front_rgb'].transform.get_matrix(),
                        camera_mount_matrix_m=mounts['front_rgb'].get_matrix()), frame=frame, timestamp=stamp)
                archive.json('ego_geometry.json', geometry)
                navigation_config = dict(desired_speed_mps=spec['cruise_speed_mps'],
                    teaching_junction_overlap=True, teaching_opposing_shared_edges=True)
                if opposing_borrow:
                    change_start, _, _, return_end = _bypass_stations(spec)
                    navigation_config['teaching_opposing_borrow_interval_m'] = [change_start, return_end]
                extractor = PublicStateExtractor(world, ego, [wp.transform for wp in waypoints],
                    config=navigation_config, route_id=output.name, ego_geometry=geometry)
                if spec['interaction_kind'] == 'stop_sign':
                    # A real public trigger is still required. This stricter
                    # teaching dwell applies only to this independent collector.
                    extractor.traffic_rules.hold_s = max(.1,float(spec['wait_s']))
                    archive.json('demonstration_stop_policy.json',dict(real_stop_sign_required=True,
                        observed_hold_s=extractor.traffic_rules.hold_s,formal_rules_modified=False))
                teacher_settings = dict(cruise_speed_mps=spec['cruise_speed_mps'],corridor_margin_m=.2)
                teacher_settings.update(spec['teacher_policy'].get('settings', {}))
                if spec['interaction_kind'] == 'stop_sign':
                    # The public stop trigger requires an observed stop within
                    # 4m. A 2.5m front margin plus half a car would stop outside
                    # that trigger forever. Keep the body behind the real line
                    # while allowing the measured public obligation to complete.
                    teacher_settings['stop_margin_m'] = .5
                teacher = DemoTeacher(route, geometry['wheelbase_m'],settings=teacher_settings,bypass_reference=authored,
                    opposing_borrow=opposing_borrow,early_return_reference=early_authored)
                archive.json('demonstration_teacher.json',dict(schema=TEACHER_SCHEMA,settings=teacher.settings,
                    formal_evaluation_control=False,safety_certified=False))
                steering = math.radians(max(w.max_steer_angle for w in physics.wheels))
                controller = SignedAccelerationController(steering, dt=1/HZ,
                    acceleration_min=-5., acceleration_max=2., steering_rate=.7)
                archive.append('event_evidence', dict(frame=frame,time=stamp,phase='approach',event='collection_started',completed=True))
            scene = extractor.capture(frame, camera_images=images, world_snapshot=snapshot)
            audit = _body_audit(scene, previous, collisions, line_events)
            extractor.remember_audited_green_crossings(scene, audit)
            audit_rows.append(audit)
            if previous is None:
                # Check the same measured footprint used by the execution audit,
                # including the full return and recovery, before driving starts.
                measured = scene['ego']
                body_evidence = events.validate_bodies(authored, measured['width'], ego_length=measured['length'])
                body_evidence['road_coverage'] = validate_teaching_road_path(
                    authored, scene['drivable_polygons'], measured['length'], measured['width'])
                archive.json('development_site_geometry.json', body_evidence)
                if not body_evidence['road_coverage']['valid']:
                    raise RuntimeError('Teaching reference failed full-body road preflight: '+
                        json.dumps(body_evidence['road_coverage']['first_failure'], ensure_ascii=False))
                if early_authored is not None:
                    early_evidence = events.validate_bodies(early_authored, measured['width'],
                        ego_length=measured['length'], path_name='early_return')
                    early_evidence['road_coverage'] = validate_teaching_road_path(
                        early_authored, scene['drivable_polygons'], measured['length'], measured['width'])
                    body_evidence['early_return'] = early_evidence
                    archive.json('development_site_geometry.json', body_evidence)
                    if not early_evidence['road_coverage']['valid']:
                        raise RuntimeError('Early return reference failed full-body road preflight: '+
                            json.dumps(early_evidence['road_coverage']['first_failure'], ensure_ascii=False))
            actual = ego.get_control()
            command = teacher.step(scene, 1/HZ)
            borrow = command.get('borrow')
            if borrow is not None and borrow['stage'] != previous_borrow_stage:
                archive.append('event_evidence', dict(frame=frame,time=stamp,phase='interact',
                    event='observed_borrow_'+borrow['stage'],completed=False,borrow=borrow))
                previous_borrow_stage = borrow['stage']
            if borrow is not None and borrow.get('early_return_trigger_id') is not None and not return_update_recorded:
                archive.append('event_evidence', dict(frame=frame,time=stamp,phase='interact',
                    event='observed_oncoming_early_return_selection',completed=False,borrow=borrow))
                return_update_recorded = True
            speed = math.hypot(*scene['ego']['velocity'])
            heading = scene['ego']['heading']
            acceleration = sum(v*d for v,d in zip(scene['ego']['acceleration'], (math.cos(heading),math.sin(heading))))
            control = controller.step(frame, command['acceleration_mps2'], command['steering_rad_left'],
                command['target_speed_mps'], speed, acceleration,
                actual_steering_rad=-actual.steer*steering)
            returned = carla.VehicleControl(throttle=control['throttle'],brake=control['brake'],
                                           steer=control['steer_normalized_right'],hand_brake=False)
            execution=dict(frame=frame,time=stamp,applied_control=_control(actual),
                returned_control=_control(returned), observed_ego=scene['ego'],
                teacher=command,controller=control,audit=audit)
            for event in events.update(scene,1/HZ,snapshot=snapshot):
                archive.append('event_evidence', dict(frame=frame,time=stamp,**event))
            progress = scene['diagnostics']['route_projection'].get('trusted_progress_m')
            if interaction_window is not None:
                recovery_evidence,recovery_events=interaction_window.update(scene,events.completed,
                    saved_source=saved_source,audit=audit)
                execution['interaction_recovery']=recovery_evidence
                for event in recovery_events:
                    archive.append('event_evidence',dict(frame=frame,time=stamp,**event))
                recovery_complete=interaction_window.complete
            elif events.completed and progress is not None:
                completion_s = events.completion_progress_m
                if completion_s is None:
                    raise RuntimeError('Completed event lacks actual completion progress')
                if recovery_start is None:
                    recovery_start = stamp
                    archive.append('event_evidence', dict(frame=frame,time=stamp,phase='recovery',event='recovery_started',completed=False))
                if not recovery_complete and progress >= completion_s+float(spec['recovery_m']):
                    recovery_complete = True
                    archive.append('event_evidence', dict(frame=frame,time=stamp,phase='recovery',event='recovery_completed',completed=True))
            archive.append('execution',execution)
            arrival = math.dist(scene['ego']['position'], route[-1]) < 1.5 and speed < .2
            correct_lane = scene['ego']['lane_id'] == '{}|{}|{}'.format(waypoints[-1].road_id,waypoints[-1].section_id,waypoints[-1].lane_id)
            if arrival and correct_lane and events.completed and recovery_complete:
                reached = True
                arrived_time = stamp if arrived_time is None else arrived_time
            else:
                reached, arrived_time = False, None
            if saved_source:
                views = dict((name, arrays[name][1][:,:,2::-1]) for name,_ in VIEWS)
                mosaic = np.concatenate((np.concatenate([views[n] for n in ('front_left_rgb','front_rgb','front_right_rgb')],axis=1),
                    np.concatenate([views[n] for n in ('back_left_rgb','back_rgb','back_right_rgb')],axis=1)),axis=0)
                native = 'native/%010d.png'%frame
                _png(output/native, mosaic)
                manifest_path = 'native/%010d.cameras.json'%frame
                origin=scene['diagnostics']['frozen_actor_origin_pose']['matrix_xyz_carla']
                cameras=[]
                for image in images:
                    raw=raw_images[image['id']]
                    calibration=source_camera_calibration(raw.transform.get_matrix(),mounts[image['id']].get_matrix(),origin)
                    cameras.append(dict(view=image['id'],rgb_path=Path(image['path']).relative_to(output).as_posix(),
                        frame=int(raw.frame),timestamp=float(raw.timestamp),width=image['width'],height=image['height'],fov_deg=100.,
                        calibration_source='same_frame_raw_image_transform_and_frozen_actor_origin',**calibration))
                residual=max(math.dist(c['api_global_minus_ue_origin_m'],cameras[0]['api_global_minus_ue_origin_m']) for c in cameras)
                if residual>.02:
                    raise ValueError('Same-frame rigid cameras disagree on the UE/global origin by %.6fm'%residual)
                atomic_json(output/manifest_path,dict(frame_id=scene['frame_id'],time=stamp,carla_frame=frame,
                    carla_timestamp=stamp,cameras=cameras,rigid_origin_residual_max_m=residual,rigid_origin_tolerance_m=.02))
                archive.append('raw_frames',dict(frame=frame,time=stamp,route_id=output.name,scene=scene,images=images,
                    native_image=native,camera_manifest=manifest_path,audit=audit))
                LOG.info('%s: sim=%.1fs frame=%d progress=%.2fm speed=%.2fm/s phase=%s interaction_complete=%s',
                    spec['id'],elapsed,frame,progress or 0.,speed,command['phase'],events.completed)
            if any(audit[k] for k in ('collision','offroad','rule_violation')) or not audit['projection_valid']:
                raise RuntimeError('Demonstration failed actual physical audit: '+json.dumps(audit,ensure_ascii=False))
            if interaction_window is not None and interaction_window.complete:
                complete=True
                break
            if reached and stamp-arrived_time >= 4.:
                complete = True
                break
            if elapsed >= spec['max_duration_s']:
                complete = True
                break
            if world.get_snapshot().frame != frame:
                raise RuntimeError('World advanced during source capture/control calculation')
            ego.apply_control(returned)
            previous = scene
    except BaseException as exc:
        errors.append(type(exc).__name__+': '+str(exc))
        try:
            busy_evidence = (dict(failure_code='world_not_idle', world_preflight=preflight)
                             if preflight is not None and not preflight['idle'] else {})
            archive.json('capture_failure.json',dict(error=errors[-1],traceback=traceback.format_exc(),
                walker_placements=events.walker_placements if events is not None else [],
                capture_actor_ids=[a.id for a in owned],
                interaction_actor_ids=[a.id for a in events.actors] if events is not None else [],
                frame=locals().get('frame'),time=locals().get('stamp'),successful_memory_created=False,
                **busy_evidence))
        except Exception as write_error:
            errors.append('failure evidence write: '+str(write_error))
        try:
            _archive_failure_scene(archive,locals().get('scene'),locals().get('audit'))
        except Exception as write_error:
            errors.append('failure scene write: '+str(write_error))
        raise
    finally:
        if events is not None and events.walker_placements:
            try:
                archive.json('walker_placements.json', events.walker_placements)
            except Exception as exc:
                errors.append('walker placement evidence: '+str(exc))
        if events is not None and events.geometry_audit is not None:
            try:
                archive.json('development_site_geometry.json',events.geometry_audit)
            except Exception as exc:
                errors.append('geometry evidence: '+str(exc))
        # Drain no synthetic physics steps. Sensor callbacks are finalized before
        # their evidence is summarized; close/restore failures invalidate a run.
        for sensor in reversed(sensors):
            try:
                sensor.stop()
            except Exception as exc:
                errors.append('sensor stop: '+str(exc))
        release_site('final')
        remaining_check = None
        if saved is not None:
            if saved_weather is not None:
                try:
                    world.set_weather(saved_weather)
                except Exception as exc:
                    errors.append('restore weather: '+str(exc))
            try:
                world.apply_settings(saved)
                restored = True
            except Exception as exc:
                errors.append('restore settings: '+str(exc))
        if cleanup_reports and restored and not saved.synchronous_mode:
            # Consume a naturally published snapshot after restoration. Looking up actors
            # before the world advances can return the stale, pre-destruction cache.
            try:
                snapshot = world.wait_for_tick(3.)
                owned_ids = {i for report in cleanup_reports for i in report['owned_ids']}
                remaining = sorted(i for i in owned_ids if snapshot.find(i) is not None)
                remaining_check = dict(frame=snapshot.frame, remaining_owned_ids=remaining)
                if remaining:
                    errors.append('Owned actors still present after restoration: '+str(remaining))
            except Exception as exc:
                remaining_check = dict(error=str(exc), remaining_owned_ids=None)
        archive.json('physical_events.json',dict(collisions=collisions,line_events=line_events))
        try:
            archive.close()
            if video:
                status = json.loads((output/'planning'/'video_recording.json').read_text(encoding='utf-8'))
                init_failed_before_video = errors and not audit_rows and status.get('status') == 'no_frames'
                if not init_failed_before_video and (status.get('status') != 'complete'
                        or status.get('playback_verified') is not True or status.get('frames') != len(audit_rows)
                        or status.get('decoded_frames') != len(audit_rows)):
                    errors.append('Recorded video did not pass full physical-frame decode verification')
        except Exception as exc:
            errors.append('archive close: '+str(exc))
        # Video errors, teardown errors and unknown audits cannot certify data.
        evaluation = dict(protocol='ep-driver-f2d-demo-evaluation-1.0',
            capture_mode=run['capture_mode'],
            reached=reached, interaction_complete=bool(events and events.completed),
            recovery_complete=recovery_complete, collision_count=len(collisions),
            offroad_count=sum(a['offroad'] for a in audit_rows),
            rule_violation_count=max(len(line_events),sum(a['rule_violation'] for a in audit_rows)),
            projection_valid=bool(audit_rows) and all(a['projection_valid'] for a in audit_rows),
            world_settings_restored=restored,errors=errors,official_score=False,
            actor_cleanup=dict(attempts=cleanup_reports, post_restore_snapshot=remaining_check),
            evidence='actual_20hz_execution_and_sensors_not_designed_reference',
            visual_proxy=spec['proxy'])
        if interaction_window is not None:
            evaluation['interaction_segment']=interaction_window.report()
        goal_complete=(interaction_window.complete if interaction_window is not None else reached)
        if complete and requires_borrow_proof(spec):
            try:
                evaluation['opposing_borrow_proof']=verify_borrow_capture(output,spec)
            except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
                errors.append('Authored opposing borrow verification: '+str(error))
        evaluation['passed'] = bool(complete and not errors and goal_complete and evaluation['interaction_complete']
            and recovery_complete and not evaluation['collision_count'] and not evaluation['offroad_count']
            and not evaluation['rule_violation_count'] and evaluation['projection_valid'] and restored)
        atomic_json(output/'demo_evaluation.json',evaluation)
        run['demonstration_verification'] = dict(passed=evaluation['passed'],
            evaluation_sha256=hashlib.sha256((output/'demo_evaluation.json').read_bytes()).hexdigest(),
            physical_events_sha256=hashlib.sha256((output/'physical_events.json').read_bytes()).hexdigest(),
            evidence_protocol=evaluation['protocol'], actual_physics_hz=HZ,
            official_evaluation=False)
        run.update(complete=complete and not errors, raw_frame_count=sum(1 for _ in (output/'raw_frames.jsonl').open(encoding='utf-8'))
                   if (output/'raw_frames.jsonl').is_file() else 0, errors=errors,
                   normalization_protocol='observable-driving-recording-1.0')
        atomic_json(output/'run.json',run)
    return evaluation
