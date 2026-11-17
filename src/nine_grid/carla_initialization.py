"""Initialization hold/release helpers shared by the v2 experiment runtime. Python 3.7."""
import math
import sys

def finalize_ego_initial_speed(experiment, carla):
    """注册物理帧后初始化合法自车速度；本函数不推进世界时钟。

    spawn时自车保持静止。桥在一次注册tick之后调用本函数，再以随后的
    同步帧建立t0。地图限速尚未注册时不能拿零或自造默认值继续实验。
    YAML的请求值不变，实际采用值和限速单独审计；不改变NPC速度策略。
    """
    if getattr(experiment, 'initial_speed_audit', None) is not None:
        raise RuntimeError('自车初始速度已经初始化；禁止在运行中重复设置目标速度')
    requested = experiment.settings.get('initial_speed_mps', 10.)
    if type(requested) not in (int, float) or not math.isfinite(requested) or requested < 0:
        raise ValueError('自车请求初始速度必须为有限非负m/s')
    limit_kmh = experiment.ego.get_speed_limit()
    if type(limit_kmh) not in (int, float) or not math.isfinite(limit_kmh) or limit_kmh <= 0:
        raise ValueError('自车注册帧后的CARLA地图限速未就绪或非法：{}；不补默认限速'.format(limit_kmh))
    limit_mps = float(limit_kmh) / 3.6
    applied = min(float(requested), limit_mps)
    forward = experiment.ego.get_transform().get_forward_vector()
    values = (forward.x, forward.y, forward.z)
    norm = math.sqrt(sum(value*value for value in values))
    if not all(math.isfinite(value) for value in values) or not math.isfinite(norm) or norm <= 0:
        raise ValueError('自车注册帧的前向向量非法，不能初始化速度')
    experiment.ego.set_target_velocity(carla.Vector3D(
        x=applied*forward.x/norm, y=applied*forward.y/norm, z=applied*forward.z/norm))
    audit = {'version': 'carla-legal-initial-speed-1.0',
        'requested_initial_speed_mps': float(requested),
        'applied_initial_speed_mps': applied,
        'map_speed_limit_mps': limit_mps, 'map_speed_limit_kmh': float(limit_kmh),
        'limited_by_map': applied < float(requested),
        'source': 'CARLA.Vehicle.get_speed_limit_after_registration_tick',
        'policy': 'min(requested_initial_speed,map_speed_limit);yaml_unchanged',
        'scope': 'initialization_only;not_an_online_control_or_speed_recovery'}
    experiment.initial_speed_audit = audit
    experiment.events.append(dict(audit, event='ego_initial_speed_applied'))
    print('自车初始速度：请求{:.6f}m/s，地图限速{:.6f}m/s，实际设置{:.6f}m/s；YAML请求值保留。'.format(
        requested, limit_mps, applied), file=sys.stderr, flush=True)
    return dict(audit)


def _initial_motion_snapshot(experiment):
    """只读注册后的演员状态；垂向落地单列，不当作平面布局漂移。"""
    actors = {}
    for actor in experiment.actors:
        pose, velocity = actor.get_transform(), actor.get_velocity()
        position = [float(pose.location.x), float(pose.location.y), float(pose.location.z)]
        speed = [float(velocity.x), float(velocity.y), float(velocity.z)]
        yaw = float(pose.rotation.yaw)
        if not all(math.isfinite(value) for value in position+speed+[yaw]):
            raise ValueError('初始化保持期间演员状态非有限：{}'.format(actor.id))
        actors[str(actor.id)] = {'type_id': str(actor.type_id), 'position_m': position,
            'yaw_deg': yaw, 'velocity_mps': speed, 'planar_speed_mps': math.hypot(*speed[:2])}
    ego = actors[str(experiment.ego.id)]
    for state in actors.values():
        state['relative_to_ego_xy_m'] = [state['position_m'][i]-ego['position_m'][i] for i in (0, 1)]
    return actors


def hold_initial_motion(experiment, carla):
    """注册tick之后制动保持演员；不推进世界，也不调用场景触发逻辑。"""
    if getattr(experiment, 'initial_motion_audit', None) is not None:
        raise RuntimeError('初始化运动已经进入保持流程，不能重复保持')
    audit = {'version': 'carla-initial-motion-hold-1.0', 'state': 'holding',
        'xy_tolerance_m': .25, 'yaw_tolerance_deg': 3.,
        'policy': 'brake_hold;deferred_initial_velocities;no_teleport_or_physics_toggle',
        'pending_npc_velocity_count': len(experiment._pending_initial_velocities)}
    experiment.initial_motion_audit = audit
    experiment._initial_motion_controls = []
    experiment._initial_motion_release_attempted = False
    try:
        audit['before_hold'] = _initial_motion_snapshot(experiment)
        audit['original_vehicle_controls'] = {}
        # get_control与get_speed_limit一样读取最近注册帧；调用者必须先完成注册tick。
        for actor in experiment.actors:
            if actor.type_id.startswith('vehicle.'):
                control = actor.get_control()
                experiment._initial_motion_controls.append((actor, control))
                audit['original_vehicle_controls'][str(actor.id)] = {
                    name: getattr(control, name) for name in ('throttle', 'steer', 'brake',
                        'hand_brake', 'reverse', 'manual_gear_shift', 'gear')}
        for actor, _ in experiment._initial_motion_controls:
            actor.apply_control(carla.VehicleControl(brake=1., hand_brake=True))
        audit['state'] = 'held'
    except Exception as exc:
        audit.update(state='hold_failed', error=type(exc).__name__+': '+str(exc))
        raise
    return audit


def release_initial_motion(experiment, carla):
    """核查保持布局后仅激活一次；恢复原控制，再设置NPC及合法自车初速。"""
    audit = getattr(experiment, 'initial_motion_audit', None)
    if (audit is None or audit.get('state') != 'held' or
            getattr(experiment, '_initial_motion_release_attempted', False)):
        raise RuntimeError('初始化运动未处于保持状态或已经尝试激活，不能重复激活')
    experiment._initial_motion_release_attempted = True
    try:
        after = _initial_motion_snapshot(experiment)
        before = audit['before_hold']
        audit['before_release'] = after
        if set(before) != set(after):
            raise RuntimeError('初始化保持期间演员集合发生变化')
        drift = {}
        failures = []
        for identity, start in before.items():
            end = after[identity]
            absolute_xy = math.hypot(*(end['position_m'][i]-start['position_m'][i] for i in (0, 1)))
            relative_xy = math.hypot(*(end['relative_to_ego_xy_m'][i]-start['relative_to_ego_xy_m'][i]
                                      for i in (0, 1)))
            yaw = (end['yaw_deg']-start['yaw_deg']+180.) % 360.-180.
            drift[identity] = {'absolute_xy_m': absolute_xy, 'relative_to_ego_xy_m': relative_xy,
                'yaw_deg': yaw, 'vertical_m': end['position_m'][2]-start['position_m'][2],
                'planar_speed_before_release_mps': end['planar_speed_mps']}
            if max(absolute_xy, relative_xy) > audit['xy_tolerance_m'] or abs(yaw) > audit['yaw_tolerance_deg']:
                failures.append(identity)
        audit['drift'] = drift
        audit['layout_within_tolerance'] = not failures
        if failures:
            raise RuntimeError('初始化保持期间演员平面布局漂移超限：{}；不瞬移修正'.format(','.join(failures)))
        # parked/occluder的原手刹也在这里恢复；不能统一松开所有车辆手刹。
        for actor, control in experiment._initial_motion_controls:
            actor.apply_control(control)
        applied = []
        audit['applied_npc_initial_velocities'] = applied
        for actor, x, y, z in experiment._pending_initial_velocities:
            limit = float(actor.get_speed_limit()) / 3.6
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError('NPC map speed limit unavailable at initialization')
            requested = math.sqrt(x*x+y*y+z*z)
            factor = min(1., limit/max(requested, 1e-9))
            x, y, z = x*factor, y*factor, z*factor
            actor.set_target_velocity(carla.Vector3D(x=x, y=y, z=z))
            applied.append({'actor_id': str(actor.id), 'velocity_mps': [x, y, z]})
        ego_audit = finalize_ego_initial_speed(experiment, carla)
        experiment._pending_initial_velocities = []
        audit.update(state='released', ego_initial_speed=ego_audit)
        return ego_audit
    except Exception as exc:
        audit.update(state='release_failed', error=type(exc).__name__+': '+str(exc))
        raise
