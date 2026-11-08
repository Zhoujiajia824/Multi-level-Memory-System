"""CARLA 0.9.15首批可控交互场景；兼容SDK的Python 3.7，不推进世界时钟。

场景使用公开地图选择足够长、同层、非路口路段；不能把无法生成当成实验成功。
周车使用物理Ackermann控制，行人使用WalkerControl，运行期间不瞬移对象。
"""
import math
import random


SCENARIOS = ('slow_lead', 'hard_brake', 'cut_in', 'parked_obstacle', 'occluded_pedestrian')
EXPERIMENT_FIELDS = frozenset(('name', 'seed', 'spawn_index', 'route_length_m', 'initial_speed_mps',
    'lead_speed_mps', 'lead_distance_m', 'trigger_distance_m', 'lane_change_seconds',
    'pedestrian_speed_mps', 'ego_blueprint', 'lead_blueprint', 'trigger_min_time_s',
    'hard_brake_amount', 'cut_in_side', 'pedestrian_longitudinal_offset_m',
    'occluder_lateral_fraction', 'occluder_blueprint', 'pedestrian_blueprint'))


def _distance(a, b):
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)


def _angle(value):
    return (value+math.pi) % (2*math.pi)-math.pi


def _trace(wp, length, spacing=2.0):
    """只选公开合法后继中最接近当前方向的路段，拒绝路口和高度歧义。"""
    points = [wp]
    travelled = 0.0
    while travelled < length:
        options = [p for p in points[-1].next(spacing)
                   if not p.is_junction and abs(p.transform.location.z-wp.transform.location.z) < 1.5]
        if not options:
            return None
        current = points[-1]
        nxt = min(options, key=lambda p: (abs(_angle(math.radians(p.transform.rotation.yaw-current.transform.rotation.yaw))),
                                         p.road_id, p.section_id, p.lane_id, p.s))
        if abs(_angle(math.radians(nxt.transform.rotation.yaw-wp.transform.rotation.yaw))) > .3:
            return None
        distance = _distance(nxt.transform.location, current.transform.location)
        if distance < .1:
            return None
        travelled += distance
        points.append(nxt)
    return points


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
    return name


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
        spawn_points = list(road_map.get_spawn_points())
        index = self.settings.get('spawn_index')
        if index is not None:
            if type(index) is not int or not 0 <= index < len(spawn_points):
                raise ValueError('spawn_index不在当前地图生成点范围内')
            candidates = [(index, spawn_points[index])]
        else:
            candidates = list(enumerate(spawn_points))
            self.rng.shuffle(candidates)
        selected = None
        for idx, pose in candidates:
            wp = road_map.get_waypoint(pose.location)
            if wp is None or wp.is_junction:
                continue
            route = _trace(wp, float(self.settings.get('route_length_m', 180.)))
            if route is None:
                continue
            lead_wp = _at(route, float(self.settings.get('lead_distance_m', 35.)))
            if self.name in ('cut_in', 'parked_obstacle'):
                side = self.settings.get('cut_in_side','left') if self.name=='cut_in' else 'left'
                if not _can_cross(lead_wp,self.carla,side,self.name=='cut_in'):
                    continue
                if self.name == 'cut_in':
                    # 预先核对本次交互会经过的路段；实际控制阶段仍逐帧核对。
                    end_s = min(float(self.settings.get('route_length_m',180.)),
                                float(self.settings.get('lead_distance_m',35.))+
                                float(self.settings.get('lead_speed_mps',6.))*(float(self.settings.get('lane_change_seconds',4.))+8.))
                    checked=[_at(route,d) for d in range(int(self.settings.get('lead_distance_m',35.)),int(end_s),2)]
                    if not all(_can_cross(p,self.carla,side,True) for p in checked):
                        continue
            if self.name == 'occluded_pedestrian':
                sidewalk = lead_wp.get_right_lane()
                for _ in range(5):
                    if sidewalk is None or str(sidewalk.lane_type) == 'Sidewalk':
                        break
                    sidewalk = sidewalk.get_right_lane()
                if sidewalk is None or str(sidewalk.lane_type) != 'Sidewalk':
                    continue
            selected = idx, wp, route, lead_wp
            break
        if selected is None:
            raise RuntimeError('当前地图没有满足本模板的同层连续路段；明确指定适用地图或spawn_index，不会换成其他模板')
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
            sidewalk = pedestrian_wp.get_right_lane()
            for _ in range(5):
                if sidewalk is None or str(sidewalk.lane_type) == 'Sidewalk':
                    break
                sidewalk = sidewalk.get_right_lane()
            if sidewalk is None or str(sidewalk.lane_type) != 'Sidewalk':
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
    return Experiment(world,client,settings,carla)
