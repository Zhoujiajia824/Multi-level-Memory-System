"""CARLA 0.9.15首批可控交互场景；兼容SDK的Python 3.7，不推进世界时钟。

局部交互段与完整自车导航分开检查；不能把无法生成当成实验成功。
周车使用物理Ackermann控制，行人使用WalkerControl，运行期间不瞬移对象。
"""
import math
import random
import json


SCENARIOS = ('slow_lead', 'hard_brake', 'cut_in', 'parked_obstacle', 'occluded_pedestrian', 'scripted_traffic')
REQUIRED_EXPERIMENT_FIELDS = frozenset(('name', 'seed', 'spawn_index', 'route_length_m', 'initial_speed_mps',
    'lead_speed_mps', 'lead_distance_m', 'trigger_distance_m', 'lane_change_seconds',
    'pedestrian_speed_mps', 'ego_blueprint', 'lead_blueprint', 'trigger_min_time_s',
    'hard_brake_amount', 'cut_in_side', 'pedestrian_longitudinal_offset_m',
    'occluder_lateral_fraction', 'occluder_blueprint', 'pedestrian_blueprint'))
OPTIONAL_EXPERIMENT_DEFAULTS = {'interaction_length_m': None, 'navigation_step_yaw_limit_deg': 30.}
EXPERIMENT_FIELDS = REQUIRED_EXPERIMENT_FIELDS | frozenset(OPTIONAL_EXPERIMENT_DEFAULTS)
ROUTE_SELECTION_VERSION = 'carla-local-corridor-navigation-1.1'


def _distance(a, b):
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)


def _angle(value):
    return (value+math.pi) % (2*math.pi)-math.pi


def _waypoint_key(wp):
    return (wp.road_id, wp.section_id, wp.lane_id, round(float(wp.s), 2))


def _trace_details(wp, length, interaction_length, spacing=2., max_step_yaw_deg=30.):
    """公开合法后继的有界深度搜索。近直非路口仅约束交互段，不约束远端导航。

    后续分支优先最小局部转角，断路时尝试其余公开后继；不是指定终点的A*。
    保持相对起点的同层限制，与桥的地图高度筛选一致；不允许重复路段回环。
    """
    points, arcs, choices = [wp], [0.], [None]
    visited = {_waypoint_key(wp)}
    longest, failure, expanded = 0., 'navigation_dead_end', 0
    while points:
        if arcs[-1] >= length:
            return list(points), 'accepted', arcs[-1]
        if choices[-1] is None:
            if expanded >= 5000:
                return None, 'navigation_search_limit', longest
            expanded += 1
            current = points[-1]
            choices[-1] = sorted(current.next(spacing), key=lambda p: (
                abs(_angle(math.radians(p.transform.rotation.yaw-current.transform.rotation.yaw))),
                p.road_id, p.section_id, p.lane_id, p.s))
            if not choices[-1] and arcs[-1] >= longest:
                longest, failure = arcs[-1], 'navigation_dead_end'
        if not choices[-1]:
            visited.remove(_waypoint_key(points.pop()))
            arcs.pop(); choices.pop()
            continue
        nxt = choices[-1].pop(0)
        current, arc = points[-1], arcs[-1]
        step = _distance(nxt.transform.location, current.transform.location)
        reason = None
        if str(nxt.lane_type) != 'Driving':
            reason = 'navigation_not_driving'
        elif abs(nxt.transform.location.z-wp.transform.location.z) >= 1.5:
            reason = 'navigation_other_level'
        elif not .1 <= step <= spacing*1.5:
            reason = 'navigation_discontinuous'
        elif abs(_angle(math.radians(nxt.transform.rotation.yaw-current.transform.rotation.yaw))) > math.radians(max_step_yaw_deg):
            reason = 'navigation_step_turn'
        elif (_waypoint_key(nxt) in visited or
              any(_distance(nxt.transform.location, p.transform.location) < .5 for p in points[:-2])):
            reason = 'navigation_loop'
        elif arc < interaction_length and nxt.is_junction:
            reason = 'interaction_junction'
        elif arc < interaction_length and abs(_angle(math.radians(nxt.transform.rotation.yaw-wp.transform.rotation.yaw))) > .3:
            reason = 'interaction_bend'
        if reason:
            if arc >= longest:
                longest, failure = arc, reason
            continue
        points.append(nxt); arcs.append(arc+step); choices.append(None)
        visited.add(_waypoint_key(nxt))
    return None, failure, longest


def interaction_length(settings):
    """交互走廊须覆盖实际布置及模板核查段；10m为开发起点的布置余量。"""
    lead = float(settings.get('lead_distance_m', 35.))
    required = lead+10.
    if settings.get('name', 'slow_lead') == 'occluded_pedestrian':
        required += float(settings.get('pedestrian_longitudinal_offset_m', 6.))
    elif settings.get('name') == 'cut_in':
        required = max(required, lead+float(settings.get('lead_speed_mps', 6.))*(
            float(settings.get('lane_change_seconds', 4.))+8.))
    explicit = settings.get('interaction_length_m')
    return required if explicit is None else float(explicit)


def _at(points, distance):
    arc = 0.0
    for a, b in zip(points, points[1:]):
        arc += _distance(a.transform.location, b.transform.location)
        if arc >= distance:
            return b
    raise ValueError('场景路线长度不足，不能裁剪摆放位置')


def _same_direction(a, b):
    return (b is not None and abs(_angle(math.radians(a.transform.rotation.yaw-b.transform.rotation.yaw))) < .3
            and abs(a.transform.location.z-b.transform.location.z) < 1.5)


def _adjacent(wp, side):
    return wp.get_left_lane() if side == 'left' else wp.get_right_lane()


def _can_cross(wp, carla, side='left', require_return=False):
    other = _adjacent(wp, side)
    outgoing = carla.LaneChange.Left if side == 'left' else carla.LaneChange.Right
    returning = carla.LaneChange.Right if side == 'left' else carla.LaneChange.Left
    return (_same_direction(wp,other) and str(other.lane_type)=='Driving'
            and bool(int(wp.lane_change) & int(outgoing))
            and (not require_return or bool(int(other.lane_change) & int(returning))))


def validate_experiment(settings):
    if not isinstance(settings, dict) or set(settings)-EXPERIMENT_FIELDS:
        raise ValueError('CARLA场景包含未实现的参数')
    name = settings.get('name', 'slow_lead')
    if name not in SCENARIOS:
        raise ValueError('未知CARLA交互模板：'+str(name))
    if type(settings.get('seed',1)) is not int or settings.get('seed',1)<0:
        raise ValueError('场景seed必须是非负整数')
    for key, default in (('route_length_m', 180.), ('lead_distance_m', 35.), ('trigger_distance_m', 28.),
                         ('lane_change_seconds', 4.), ('pedestrian_speed_mps', 1.4),
                         ('pedestrian_longitudinal_offset_m', 6.)):
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('CARLA场景参数必须为有限正数：'+key)
    for key in ('initial_speed_mps', 'lead_speed_mps'):
        value = settings.get(key, 10. if key == 'initial_speed_mps' else 6.)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('CARLA场景速度必须为有限非负数：'+key)
    if settings.get('lead_distance_m', 35.)+30 >= settings.get('route_length_m', 180.):
        raise ValueError('路线必须覆盖前车摆放位置并保留至少30米')
    if (settings.get('lead_distance_m',35.)+settings.get('pedestrian_longitudinal_offset_m',6.)
            >= settings.get('route_length_m',180.)):
        raise ValueError('路线必须覆盖行人摆放位置')
    for key, default, low, high in (('trigger_min_time_s', .5, 0., None),
                                  ('hard_brake_amount', 1., 0., 1.),
                                  ('occluder_lateral_fraction', .35, -.5, .5)):
        value = settings.get(key,default)
        if (type(value) not in (int,float) or not math.isfinite(value) or value<low
                or high is not None and value>high or key=='hard_brake_amount' and value==0):
            raise ValueError('CARLA场景参数超出有效范围：'+key)
    if settings.get('cut_in_side','left') not in ('left','right'):
        raise ValueError('cut_in_side只能为left或right')
    for key in ('ego_blueprint','lead_blueprint','occluder_blueprint'):
        if key in settings and (not isinstance(settings[key],str) or not settings[key].startswith('vehicle.')):
            raise ValueError(key+'必须指定vehicle车辆蓝图')
    pedestrian = settings.get('pedestrian_blueprint')
    if pedestrian is not None and (not isinstance(pedestrian,str) or not pedestrian.startswith('walker.pedestrian.')):
        raise ValueError('pedestrian_blueprint必须为行人蓝图或null')
    index = settings.get('spawn_index')
    if index is not None and (type(index) is not int or index<0):
        raise ValueError('spawn_index必须为null或非负整数')
    explicit = settings.get('interaction_length_m')
    if explicit is not None:
        if (type(explicit) not in (int,float) or not math.isfinite(explicit)
                or explicit < interaction_length(dict(settings, interaction_length_m=None))):
            raise ValueError('interaction_length_m必须覆盖完整交互布置走廊，或使用null自动计算')
    if interaction_length(settings) > settings.get('route_length_m',180.):
        raise ValueError('interaction_length_m不能超过route_length_m')
    yaw_limit = settings.get('navigation_step_yaw_limit_deg',30.)
    if type(yaw_limit) not in (int,float) or not math.isfinite(yaw_limit) or not 0 < yaw_limit <= 45.:
        raise ValueError('navigation_step_yaw_limit_deg必须在(0,45]度内')
    return name


def _sidewalk(wp):
    other = wp.get_right_lane()
    for _ in range(5):
        if other is None or str(other.lane_type) == 'Sidewalk':
            break
        other = other.get_right_lane()
    return other if other is not None and str(other.lane_type) == 'Sidewalk' else None


def select_route(road_map, settings, carla, audit_all=False):
    """纯地图选址，不创建演员、不推进时钟；检查与真实初始化使用同一实现。"""
    if settings.get('name') == 'scripted_traffic':
        try:
            from .carla_scripted_scenarios import select_scripted_route
        except ImportError:
            from carla_scripted_scenarios import select_scripted_route
        return select_scripted_route(road_map, settings, carla, audit_all=audit_all)
    name = validate_experiment(settings)
    spawn_points = list(road_map.get_spawn_points())
    index = settings.get('spawn_index')
    if index is not None:
        if not 0 <= index < len(spawn_points):
            raise ValueError('spawn_index不在当前地图生成点范围内')
        candidates = [(index, spawn_points[index])]
    else:
        candidates = list(enumerate(spawn_points))
        random.Random(settings.get('seed',1)).shuffle(candidates)
    length = float(settings.get('route_length_m',180.))
    local_length = interaction_length(settings)
    report = {'version': ROUTE_SELECTION_VERSION, 'success': False,
              'map': str(getattr(road_map,'name','unknown')), 'seed': settings.get('seed',1),
              'template': name, 'route_length_m': length, 'interaction_length_m': local_length,
              'total_spawn_points': len(spawn_points), 'selected_spawn_index': None,
              'rejection_counts': {}, 'candidates': [], 'audit_all': bool(audit_all),
              'scope': '交互区同层近直非路口；导航允许公开路口/温和弯曲；未验证完整路口路权'}
    selected = None
    for idx, pose in candidates:
        wp = road_map.get_waypoint(pose.location)
        route, prefix, reason, lead_wp = None, 0., None, None
        if wp is None or str(wp.lane_type) != 'Driving':
            reason = 'start_not_driving'
        elif wp.is_junction:
            reason = 'start_in_junction'
        else:
            route, reason, prefix = _trace_details(wp,length,local_length,
                max_step_yaw_deg=float(settings.get('navigation_step_yaw_limit_deg',30.)))
            if route is not None:
                lead_wp = _at(route,float(settings.get('lead_distance_m',35.)))
                if name in ('cut_in','parked_obstacle'):
                    side = settings.get('cut_in_side','left') if name=='cut_in' else 'left'
                    if not _can_cross(lead_wp,carla,side,name=='cut_in'):
                        reason = 'interaction_adjacent_lane_or_crossing'
                    elif name == 'cut_in':
                        end_s = float(settings.get('lead_distance_m',35.))+float(settings.get('lead_speed_mps',6.))*(
                            float(settings.get('lane_change_seconds',4.))+8.)
                        checked = [_at(route,d) for d in range(int(settings.get('lead_distance_m',35.)),int(end_s),2)]
                        if not all(_can_cross(p,carla,side,True) for p in checked):
                            reason = 'interaction_cut_in_crossing_range'
                elif name == 'occluded_pedestrian':
                    pedestrian_wp = _at(route,float(settings.get('lead_distance_m',35.))+
                        float(settings.get('pedestrian_longitudinal_offset_m',6.)))
                    if _sidewalk(pedestrian_wp) is None:
                        reason = 'interaction_pedestrian_sidewalk'
        row = {'spawn_index': idx, 'reason': reason, 'valid_prefix_m': prefix}
        if wp is not None:
            t = wp.transform
            row['start_pose_carla'] = {'x_m': float(t.location.x), 'y_m': float(t.location.y),
                'z_m': float(t.location.z), 'yaw_deg': float(t.rotation.yaw)}
        report['candidates'].append(row)
        if reason == 'accepted':
            if selected is None:
                selected = (idx,wp,route,lead_wp)
                report.update(success=True, selected_spawn_index=idx)
                report['selected_route'] = [{'x_m': float(p.transform.location.x),
                    'y_m': float(p.transform.location.y), 'z_m': float(p.transform.location.z),
                    'yaw_deg': float(p.transform.rotation.yaw), 'junction': bool(p.is_junction),
                    'road_id': p.road_id, 'section_id': p.section_id, 'lane_id': p.lane_id,
                    's_m': float(p.s)} for p in route]
            if not audit_all:
                break
        else:
            counts = report['rejection_counts']
            counts[reason] = counts.get(reason,0)+1
    report['checked_spawn_points'] = len(report['candidates'])
    report['longest_valid_prefix_m'] = max([row['valid_prefix_m'] for row in report['candidates']] or [0.])
    return selected, report


class RouteSelectionError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__('场景选址失败：检查{}个起点，交互段{}m，导航{}m；原因={}。'
                         '请用--inspect-scene查看同一地图的逐点诊断，不会更换模板或缩短路线'.format(
            report['checked_spawn_points'],report['interaction_length_m'],report['route_length_m'],
            json.dumps(report['rejection_counts'],ensure_ascii=False,sort_keys=True)))


class Experiment:
    def __init__(self, world, client, settings, carla):
        self.world, self.settings, self.carla = world, dict(settings), carla
        self.name = validate_experiment(settings)
        self.actors, self.events, self.controllers = [], [], []
        self.ego = None
        self.triggered_at = None
        self.rng = random.Random(int(settings.get('seed', 1)))
        self.route_waypoints = []
        self._closed = False
        try:
            self._setup()
        except BaseException:
            self.close()
            raise

    def _spawn(self, blueprint_name, wp, speed=0., shift=0., role='scenario'):
        blueprint = self.world.get_blueprint_library().find(blueprint_name)
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', 'ep_driver_'+role)
        transform = wp.transform
        right = transform.get_right_vector()
        location = self.carla.Location(x=transform.location.x+shift*right.x,
                                      y=transform.location.y+shift*right.y, z=transform.location.z+.3)
        actor = self.world.try_spawn_actor(blueprint, self.carla.Transform(location, transform.rotation))
        if actor is None:
            raise RuntimeError('场景对象生成失败：'+role+'；请更换spawn_index或调整场景参数')
        self.actors.append(actor)
        if speed:
            forward = transform.get_forward_vector()
            actor.set_target_velocity(self.carla.Vector3D(x=speed*forward.x, y=speed*forward.y, z=0.))
        return actor

    def _setup(self):
        road_map = self.world.get_map()
        selected, self.selection_report = select_route(road_map,self.settings,self.carla)
        if selected is None:
            raise RouteSelectionError(self.selection_report)
        idx, wp, self.forward_route, lead_wp = selected
        self.selected_spawn_index = idx
        self.ego = self._spawn(self.settings.get('ego_blueprint', 'vehicle.tesla.model3'), wp,
                               float(self.settings.get('initial_speed_mps', 10.)), role='ego')
        # 自车已知导航参考延伸到后方；不读取周车目标车道或私有路线。
        behind, prev = [], wp
        for _ in range(30):
            options = [p for p in prev.previous(2.) if not p.is_junction and _same_direction(wp,p)]
            if not options:
                break
            prev = min(options, key=lambda p: (abs(_angle(math.radians(p.transform.rotation.yaw-wp.transform.rotation.yaw))), p.s))
            behind.append(prev)
        self.route_waypoints = list(reversed(behind))+self.forward_route
        pose=wp.transform
        self.events.append({'event':'created', 'scenario':self.name, 'spawn_index':idx,
                            'seed':int(self.settings.get('seed', 1)),
                            'start_waypoint':{'road_id':wp.road_id,'section_id':wp.section_id,
                                'lane_id':wp.lane_id,'s_m':float(wp.s)},
                            'start_pose_carla':{'x_m':float(pose.location.x),'y_m':float(pose.location.y),
                                'z_m':float(pose.location.z),'yaw_deg':float(pose.rotation.yaw)},
                            'source':'公开地图与预设交互策略'})
        speed = float(self.settings.get('lead_speed_mps', 6.))
        if self.name == 'occluded_pedestrian':
            self.lead = self._spawn(self.settings.get('occluder_blueprint','vehicle.mercedes.sprinter'),
                lead_wp, shift=lead_wp.lane_width*float(self.settings.get('occluder_lateral_fraction',.35)), role='occluder')
            self.lead.apply_control(self.carla.VehicleControl(brake=1., hand_brake=True))
            pedestrian_wp = _at(self.forward_route, float(self.settings.get('lead_distance_m',35.))+
                                float(self.settings.get('pedestrian_longitudinal_offset_m',6.)))
            sidewalk = _sidewalk(pedestrian_wp)
            if sidewalk is None:
                raise RuntimeError('行人所在位置缺少真实人行道，不在车道内伪造遮挡行人')
            pedestrians = sorted(self.world.get_blueprint_library().filter('walker.pedestrian.*'), key=lambda b:b.id)
            if not pedestrians:
                raise RuntimeError('CARLA资源缺少行人蓝图')
            blueprint = self.settings.get('pedestrian_blueprint') or pedestrians[0].id
            if blueprint not in {p.id for p in pedestrians}:
                raise ValueError('当前地图资源缺少指定行人蓝图：'+blueprint)
            self.walker = self._spawn(blueprint, sidewalk, role='pedestrian')
            right = pedestrian_wp.transform.get_right_vector()
            self.walk_direction = self.carla.Vector3D(x=-right.x,y=-right.y,z=0.)
            self.walker.apply_control(self.carla.WalkerControl(speed=0.))
        else:
            spawn_wp = _adjacent(lead_wp,self.settings.get('cut_in_side','left')) if self.name == 'cut_in' else lead_wp
            self.lead = self._spawn(self.settings.get('lead_blueprint','vehicle.audi.a2'), spawn_wp,
                                    0. if self.name == 'parked_obstacle' else speed, role='lead')
            if self.name == 'parked_obstacle':
                self.lead.apply_control(self.carla.VehicleControl(brake=1.,hand_brake=True))

    def _follow(self, actor, speed, elapsed):
        location = actor.get_location()
        nearest = min(range(len(self.forward_route)), key=lambda i:_distance(location,self.forward_route[i].transform.location))
        velocity = actor.get_velocity()
        actual_speed = math.sqrt(velocity.x**2+velocity.y**2)
        lookahead = max(5., actual_speed*.8)
        target_index = min(nearest+max(2,int(lookahead/2)),len(self.forward_route)-1)
        target = self.forward_route[target_index].transform
        target_x,target_y=target.location.x,target.location.y
        if self.name == 'cut_in':
            side_name=self.settings.get('cut_in_side','left')
            fraction = 0. if self.triggered_at is None else min(1.,max(0.,(elapsed-self.triggered_at)/float(self.settings.get('lane_change_seconds',4.))))
            blend = fraction*fraction*(3.-2.*fraction)
            if blend<1.:
                if not _can_cross(self.forward_route[nearest],self.carla,side_name,True) or not _can_cross(self.forward_route[target_index],self.carla,side_name,True):
                    raise RuntimeError('预设切入区间已遇到不可跨越边界或地图缺口，终止并记录场景失败')
                side=_adjacent(self.forward_route[target_index],side_name).transform.location
                target_x=(1.-blend)*side.x+blend*target.location.x
                target_y=(1.-blend)*side.y+blend*target.location.y
        dx = target_x-location.x
        dy = target_y-location.y
        alpha = _angle(math.atan2(dy,dx)-math.radians(actor.get_transform().rotation.yaw))
        steering = math.atan2(2.*2.7*math.sin(alpha), max(1.,math.hypot(dx,dy)))
        steering = max(-.5,min(.5,steering))
        # 脚本车到公开路线末端前按2m/s²开发制动值收速，避免越过末点后反向追点。
        remaining = sum(_distance(a.transform.location,b.transform.location)
                        for a,b in zip(self.forward_route[nearest:],self.forward_route[nearest+1:]))
        speed = min(speed, math.sqrt(4.*max(0.,remaining-2.)))
        if remaining <= 2.:
            steering = 0.
        actor.apply_ackermann_control(self.carla.VehicleAckermannControl(steer=steering,speed=speed))

    def before_tick(self, elapsed_s, dt):
        if self._closed:
            raise RuntimeError('场景已关闭')
        if (not math.isfinite(elapsed_s) or elapsed_s<0 or not math.isfinite(dt) or dt<=0):
            raise ValueError('场景步进时间必须有限，累计时间非负且dt为正')
        ego_location = self.ego.get_location()
        actor_location = self.lead.get_location()
        distance = _distance(ego_location,actor_location)
        if (self.triggered_at is None and elapsed_s >= float(self.settings.get('trigger_min_time_s',.5))
                and distance <= float(self.settings.get('trigger_distance_m',28.))
                and self.name in ('hard_brake','cut_in','occluded_pedestrian')):
            self.triggered_at=elapsed_s
            self.events.append({'event':'interaction_triggered','time':elapsed_s,'distance_m':distance,'scenario':self.name})
        if self.name == 'occluded_pedestrian':
            speed = float(self.settings.get('pedestrian_speed_mps',1.4)) if self.triggered_at is not None else 0.
            self.walker.apply_control(self.carla.WalkerControl(direction=self.walk_direction,speed=speed))
        elif self.name != 'parked_obstacle':
            speed = float(self.settings.get('lead_speed_mps',6.))
            if self.name == 'hard_brake' and self.triggered_at is not None:
                self.lead.apply_control(self.carla.VehicleControl(brake=float(self.settings.get('hard_brake_amount',1.))))
            else:
                self._follow(self.lead,speed,elapsed_s)

    def close(self):
        if self._closed:
            return
        self._closed = True
        failures=[]
        for actor in reversed(self.actors):
            try:
                if actor.is_alive:
                    if actor.destroy() is False:
                        failures.append('destroy返回False')
            except Exception as exc:
                failures.append(type(exc).__name__)
        if failures:
            self.events.append({'event':'cleanup_error','errors':failures})


def create_experiment(world, client, settings, carla):
    if settings.get('name') == 'scripted_traffic':
        try:
            from .carla_scripted_scenarios import ScriptedExperiment
        except ImportError:
            from carla_scripted_scenarios import ScriptedExperiment
        return ScriptedExperiment(world, client, settings, carla)
    return Experiment(world,client,settings,carla)
