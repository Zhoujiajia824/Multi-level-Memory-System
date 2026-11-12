"""独立长路线脚本交通；Python 3.7，所有移动通过物理控制实现。

场景导演知道配置的演员路线，不向核心预测器提供这些路线或触发器。
NPC 使用公开地图、当前交通灯/停止标志和场景内当前实体作简单守规控制；
不是经验证的专家驾驶策略。placement.offset_m 沿车道左法向为正，单位米。
"""
import math


SCRIPTED_REQUIRED_FIELDS = frozenset(('name', 'seed', 'spawn_index', 'route_length_m',
    'initial_speed_mps', 'ego_blueprint', 'actors', 'route_requirements'))
SCRIPTED_DEFAULTS = {'route_anchors': [], 'weather': 'ClearNoon',
                     'navigation_step_yaw_limit_deg': 30.}
SCRIPTED_EXPERIMENT_FIELDS = SCRIPTED_REQUIRED_FIELDS | frozenset(SCRIPTED_DEFAULTS)
SCRIPTED_VERSION = 'carla-scripted-traffic-1.0'


def _finite(value, name, lower=0., upper=None):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < lower
            or upper is not None and value > upper):
        raise ValueError(name+'必须为有效范围内的有限数值')


def validate_scripted_experiment(settings):
    if not isinstance(settings, dict) or set(settings)-SCRIPTED_EXPERIMENT_FIELDS:
        raise ValueError('脚本交通配置包含未实现字段')
    missing = SCRIPTED_REQUIRED_FIELDS-set(settings)
    if missing:
        raise ValueError('脚本交通配置缺少字段：'+str(sorted(missing)))
    if settings['name'] != 'scripted_traffic':
        raise ValueError('脚本交通模板名称错误')
    if type(settings['seed']) is not int or settings['seed'] < 0:
        raise ValueError('seed必须为非负整数')
    index = settings['spawn_index']
    if index is not None and (type(index) is not int or index < 0):
        raise ValueError('spawn_index必须为null或非负整数')
    _finite(settings['route_length_m'], 'route_length_m', 100., 3000.)
    _finite(settings['initial_speed_mps'], 'initial_speed_mps', 0., 35.)
    _finite(settings.get('navigation_step_yaw_limit_deg',30.), 'navigation_step_yaw_limit_deg', 1., 45.)
    if not isinstance(settings['ego_blueprint'], str) or not settings['ego_blueprint'].startswith('vehicle.'):
        raise ValueError('ego_blueprint必须为车辆蓝图')
    if not isinstance(settings.get('weather','ClearNoon'), str):
        raise ValueError('weather必须为CARLA天气预设名称')
    requirements = settings['route_requirements']
    fields = {'kind','min_junctions','min_same_direction_lanes','search_limit'}
    if not isinstance(requirements,dict) or set(requirements)-fields:
        raise ValueError('route_requirements包含未实现字段')
    if requirements.get('kind','any') not in ('any','junction','merge','diverge'):
        raise ValueError('道路类型必须为any/junction/merge/diverge')
    for key, default, low in (('min_junctions',0,0),('min_same_direction_lanes',1,1),('search_limit',5000,1)):
        value = requirements.get(key,default)
        if type(value) is not int or value < low:
            raise ValueError('route_requirements.'+key+'必须为有效整数')
    anchors = settings.get('route_anchors',[])
    if not isinstance(anchors,list):raise ValueError('route_anchors必须为列表')
    for anchor in anchors:
        if not isinstance(anchor,dict) or set(anchor)!={'road_id','lane_id','s_m'}:
            raise ValueError('路线锚点必须为road_id/lane_id/s_m')
        if type(anchor['road_id']) is not int or type(anchor['lane_id']) is not int or anchor['lane_id']==0:
            raise ValueError('路线锚点必须指定有效道路及非零车道编号')
        _finite(anchor['s_m'],'路线锚点s_m')
    try:from .carla_extended_routes import validate_route_requirements
    except ImportError:from carla_extended_routes import validate_route_requirements
    validate_route_requirements(settings)
    actors = settings['actors']
    if not isinstance(actors,list) or not 0 <= len(actors) <= 24:
        raise ValueError('actors必须为至多24个演员的列表')
    roles = set()
    for row in actors:
        required = {'role','type','blueprint','placement','behavior','speed_mps'}
        allowed = required | {'trigger','duration_s','stop_duration_s'}
        if not isinstance(row,dict) or not required <= set(row) or set(row)-allowed:
            raise ValueError('演员字段缺失或包含未实现字段')
        role = row['role']
        if (not isinstance(role,str) or not role or role in roles or role=='ego'
                or not all(c.isascii() and (c.isalnum() or c=='_') for c in role)):
            raise ValueError('演员role必须唯一且只包含ASCII字母、数字和下划线')
        roles.add(role)
        typ, behavior = row['type'], row['behavior']
        behaviors = {'vehicle':('follow','parked','cut_in','stop_resume'),
                     'pedestrian':('cross',),'prop':('parked',)}
        prefixes = {'vehicle':'vehicle.','pedestrian':'walker.pedestrian.','prop':'static.prop.'}
        if typ not in behaviors or behavior not in behaviors[typ]:
            raise ValueError('演员类型与行为不相容：'+role)
        if not isinstance(row['blueprint'],str) or not row['blueprint'].startswith(prefixes[typ]):
            raise ValueError('演员蓝图与类型不相容：'+role)
        place = row['placement']
        if not isinstance(place,dict):raise ValueError('placement必须为对象')
        anchor=place.get('anchor')
        if anchor is not None:
            if set(place)-{'anchor','offset_m'} or typ!='vehicle' or behavior!='follow':
                raise ValueError('公开支路锚点仅用于follow车辆，与s_m/lane互斥')
            if not isinstance(anchor,dict) or set(anchor)!={'road_id','lane_id','s_m'}:
                raise ValueError('支路锚点须指定road_id/lane_id/s_m')
            if type(anchor['road_id']) is not int or type(anchor['lane_id']) is not int or anchor['lane_id']==0:
                raise ValueError('支路锚点编号非法')
            _finite(anchor['s_m'],'支路锚点s_m')
        else:
            if not {'s_m','lane'}<=set(place) or set(place)-{'s_m','lane','offset_m'}:
                raise ValueError('placement仅允许s_m/lane/offset_m，或公开支路anchor/offset_m')
            _finite(place['s_m'],'演员沿路线距离',8.,settings['route_length_m']-30.)
        _finite(place.get('offset_m',0.),'演员左正横移',-8.,8.)
        lanes = ('sidewalk_left','sidewalk_right') if typ=='pedestrian' else ('current','left','right')
        if anchor is None and (place['lane'] not in lanes or (behavior=='cut_in' and place['lane']=='current')):
            raise ValueError('演员车道与行为不相容：'+role)
        if typ=='pedestrian' and place.get('offset_m',0.) != 0:
            raise ValueError('行人必须从真实人行道中心出发，不能偏移到道路中')
        _finite(row['speed_mps'],'演员速度',0.,3. if typ=='pedestrian' else 35.)
        if behavior=='parked' and row['speed_mps']!=0:
            raise ValueError('parked演员速度必须为0')
        if behavior in ('cut_in','cross','stop_resume') and row['speed_mps']<=0:
            raise ValueError('交互演员必须有正行驶/步行速度')
        trigger = row.get('trigger',{})
        if not isinstance(trigger,dict) or set(trigger)-{'time_s','ego_distance_m'}:
            raise ValueError('trigger仅允许time_s/ego_distance_m')
        _finite(trigger.get('time_s',0.),'触发最早时间',0.,120.)
        if trigger.get('ego_distance_m') is not None:
            _finite(trigger['ego_distance_m'],'触发距离',1.,150.)
        _finite(row.get('duration_s',4.),'duration_s',1.,15.)
        _finite(row.get('stop_duration_s',4.),'stop_duration_s',1.,60.)
    return 'scripted_traffic'


def _distance(a,b):
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)


def _angle(a):return (a+math.pi)%(2*math.pi)-math.pi


def _yaw(wp):return math.radians(wp.transform.rotation.yaw)


def _at(route, distance):
    arc=0.
    for a,b in zip(route,route[1:]):
        arc+=_distance(a.transform.location,b.transform.location)
        if arc>=distance:return b
    raise ValueError('演员摆放超出实际路线长度')


def _adjacent(wp,side):
    other=wp.get_left_lane() if side=='left' else wp.get_right_lane()
    if (other is None or str(other.lane_type)!='Driving'
            or abs(_angle(_yaw(other)-_yaw(wp)))>.35
            or abs(other.transform.location.z-wp.transform.location.z)>1.5):
        raise ValueError('缺少同层同向相邻行驶车道：'+side)
    return other


def _sidewalk(wp,side):
    """相对起点的物理左/右搜索，跨过对向车道后不能反转搜索方向。"""
    start_yaw=_yaw(wp)
    seen=set()
    for _ in range(16):
        key=(wp.road_id,wp.section_id,wp.lane_id)
        if key in seen:break
        seen.add(key)
        if str(wp.lane_type)=='Sidewalk':return wp
        same=math.cos(_yaw(wp)-start_yaw)>=0
        left=(side=='left')==same
        wp=wp.get_left_lane() if left else wp.get_right_lane()
        if wp is None:break
    raise ValueError('横穿位置缺少真实人行道：'+side)


def _crossable(wp,side,carla):
    other=_adjacent(wp,side)
    out=carla.LaneChange.Left if side=='left' else carla.LaneChange.Right
    back=carla.LaneChange.Right if side=='left' else carla.LaneChange.Left
    if not (int(wp.lane_change)&int(out)) or not (int(other.lane_change)&int(back)):
        raise ValueError('切入区间存在不可跨越边界')
    if wp.is_junction or other.is_junction:
        raise ValueError('切入区间不能穿越路口')
    return other


def placement_plan(route,row,carla,road_map=None):
    place=row['placement']
    if 'anchor' in place:
        if road_map is None:raise ValueError('支路演员缺少公开地图，不能推断摆放位置')
        anchor=place['anchor']
        wp=road_map.get_waypoint_xodr(anchor['road_id'],anchor['lane_id'],anchor['s_m'])
        if wp is None or str(wp.lane_type)!='Driving':raise ValueError('支路锚点不是合法行驶车道')
        actor_route=_extend(wp,160.)
        # 同层且横向相交才称为交叉来车；邻道平行跟驰不能冒充该交互。
        junction_hits=[]
        cells={}
        for p in route:
            loc=p.transform.location;key=(int(math.floor(loc.x/4.)),int(math.floor(loc.y/4.)))
            cells.setdefault(key,[]).append(p)
        for p in actor_route:
            loc=p.transform.location;kx,ky=int(math.floor(loc.x/4.)),int(math.floor(loc.y/4.))
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for q in cells.get((kx+dx,ky+dy),[]):
                        if (p.is_junction and q.is_junction and _distance(loc,q.transform.location)<3.
                                and abs(loc.z-q.transform.location.z)<1.5
                                and abs(math.cos(_yaw(p)-_yaw(q)))<.7):
                            junction_hits.append((p,q))
        if not junction_hits:raise ValueError('支路公开路线未与自车导航在同层路口横向相交')
        if abs(place.get('offset_m',0.))>float(wp.lane_width)*.5:
            raise ValueError('支路演员横移中心超出车道')
        hit=junction_hits[0][0].transform.location
        return {'waypoint':wp,'base':wp,'end':None,'corridor_end_m':None,'actor_route':actor_route,
            'crossing_carla_xyz':[float(hit.x),float(hit.y),float(hit.z)]}
    base=_at(route,float(place['s_m']))
    lane=place['lane']; end=None
    if row['type']=='pedestrian':
        side=lane.split('_')[1]
        wp=_sidewalk(base,side)
        end=_sidewalk(base,'right' if side=='left' else 'left')
        if abs(wp.transform.location.z-end.transform.location.z)>1.5:
            raise ValueError('横穿两端人行道高度不连续')
        if not 2. <= _distance(wp.transform.location,end.transform.location) <= 35.:
            raise ValueError('行人横穿距离不在2至35米范围')
    else:
        wp=base if lane=='current' else _adjacent(base,lane)
        if abs(place.get('offset_m',0.))>float(wp.lane_width)*.5:
            raise ValueError('横移中心超出所选车道范围')
    corridor_end=None
    if row['behavior']=='cut_in':
        # 触发时间越晚，预核查的合法切入区间越长。等待距离触发时在走廊末前停下。
        corridor_end=(float(place['s_m'])+float(row['speed_mps'])*(
            float(row.get('trigger',{}).get('time_s',0.))+float(row.get('duration_s',4.))+2.)+12.)
        for s in range(int(place['s_m']),int(math.ceil(corridor_end))+1,2):
            _crossable(_at(route,s),lane,carla)
        wp=_crossable(base,lane,carla)
    return {'waypoint':wp,'base':base,'end':end,'corridor_end_m':corridor_end}


def validate_route_actors(route,settings,carla,road_map=None):
    """选址阶段只检查公开几何；碰撞生成失败仍由服务器报告，不凭空挪演员。"""
    for row in settings['actors']:
        try:placement_plan(route,row,carla,road_map)
        except ValueError as exc:return 'actor_placement:'+row['role']+':'+str(exc)
    return None


def select_scripted_route(road_map,settings,carla,audit_all=False):
    """只读检查与真实构造共用道路及演员摆放验收，不生成对象。"""
    validate_scripted_experiment(settings)
    try:from .carla_extended_routes import select_extended_route,validate_route_requirements
    except ImportError:from carla_extended_routes import select_extended_route,validate_route_requirements
    validate_route_requirements(settings)
    return select_extended_route(road_map,settings,carla,audit_all=audit_all,
        route_filter=lambda route:validate_route_actors(route,settings,carla,road_map))


def _extend(wp,length):
    route=[wp];arc=0.;seen=set()
    for _ in range(int(length/2)+100):
        current=route[-1]
        options=sorted(current.next(2.),key=lambda p:(abs(_angle(_yaw(p)-_yaw(current))),
            p.road_id,p.section_id,p.lane_id,p.s))
        options=[p for p in options if str(p.lane_type)=='Driving' and
            .1<_distance(p.transform.location,current.transform.location)<3.5]
        if not options:break
        nxt=options[0];key=(nxt.road_id,nxt.section_id,nxt.lane_id,round(nxt.s,1))
        if key in seen:break
        seen.add(key);arc+=_distance(nxt.transform.location,current.transform.location)
        route.append(nxt)
        if arc>=length:break
    return route


class ScriptedExperiment:
    def __init__(self,world,client,settings,carla):
        self.name=validate_scripted_experiment(settings)
        self.world,self.settings,self.carla=world,dict(settings),carla
        self.actors,self.events,self.controllers=[],[],[]
        self.ego=None;self.route_waypoints=[];self._closed=False
        self._weather=None;self._rows=[];self._stops=[]
        try:self._setup()
        except BaseException:
            self.close();raise

    def _spawn(self,blueprint_name,wp,role,speed=0.,offset=0.):
        bp=self.world.get_blueprint_library().find(blueprint_name)
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name','hero' if role=='ego' else 'ep_driver_'+role)
        if bp.has_attribute('is_invincible'):bp.set_attribute('is_invincible','false')
        t=wp.transform;r=t.get_right_vector()
        loc=self.carla.Location(x=t.location.x-offset*r.x,y=t.location.y-offset*r.y,z=t.location.z+.3)
        actor=self.world.try_spawn_actor(bp,self.carla.Transform(loc,t.rotation))
        if actor is None:raise RuntimeError('脚本演员生成失败：'+role+'；不替换演员或自动移动位置')
        self.actors.append(actor)
        if speed>0:
            forward=t.get_forward_vector()
            actor.set_target_velocity(self.carla.Vector3D(x=speed*forward.x,y=speed*forward.y,z=speed*forward.z))
        return actor

    def _setup(self):
        self.road_map=self.world.get_map()
        selected,self.selection_report=select_scripted_route(self.road_map,self.settings,self.carla)
        if selected is None:
            try:from .carla_scenarios import RouteSelectionError
            except ImportError:from carla_scenarios import RouteSelectionError
            raise RouteSelectionError(self.selection_report)
        idx,wp,self.forward_route,_=selected
        self.selected_spawn_index=idx
        preset=getattr(self.carla.WeatherParameters,self.settings.get('weather','ClearNoon'),None)
        if preset is None:raise ValueError('CARLA不存在指定天气预设')
        self._weather=self.world.get_weather();self.world.set_weather(preset)
        self.ego=self._spawn(self.settings['ego_blueprint'],wp,'ego',self.settings['initial_speed_mps'])
        behind=[];previous=wp
        for _ in range(30):
            options=[p for p in previous.previous(2.) if str(p.lane_type)=='Driving'
                and not p.is_junction and abs(_angle(_yaw(p)-_yaw(previous)))<.35]
            if not options:break
            previous=min(options,key=lambda p:(abs(_angle(_yaw(p)-_yaw(previous))),p.road_id,p.lane_id,p.s))
            behind.append(previous)
        self.route_waypoints=list(reversed(behind))+self.forward_route
        self._arcs=[0.]
        for a,b in zip(self.forward_route,self.forward_route[1:]):
            self._arcs.append(self._arcs[-1]+_distance(a.transform.location,b.transform.location))
        for source in self.settings['actors']:
            plan=placement_plan(self.forward_route,source,self.carla,self.road_map)
            speed=source['speed_mps'] if source['type']=='vehicle' else 0.
            actor=self._spawn(source['blueprint'],plan['waypoint'],source['role'],speed,
                              source['placement'].get('offset_m',0.))
            route=plan.get('actor_route') or (self.forward_route if source['placement']['lane']=='current' or source['behavior']=='cut_in'
                   else _extend(plan['waypoint'],float(self.settings['route_length_m'])))
            initial_index=min(range(len(route)),key=lambda i:_distance(actor.get_location(),route[i].transform.location))
            self._rows.append({'source':source,'actor':actor,'plan':plan,'route':route,'triggered_at':None,
                'stop_started_at':None,'finished':False,'index':initial_index,'stop_signs':{},'wheelbase':None})
            if source['type']=='vehicle' and source['behavior']=='parked':
                actor.apply_control(self.carla.VehicleControl(brake=1.,hand_brake=True))
            elif source['type']=='pedestrian':actor.apply_control(self.carla.WalkerControl(speed=0.))
            self.events.append({'event':'actor_created','role':source['role'],'actor_id':str(actor.id),
                'blueprint':source['blueprint'],'behavior':source['behavior'],'placement':source['placement'],
                'crossing_carla_xyz':plan.get('crossing_carla_xyz')})
        # 独立世界中只记录原交通灯状态；不冻结灯，也不把演员脚本作为隐藏意图输入规划器。
        self.events.append({'event':'created','version':SCRIPTED_VERSION,'scenario':self.name,
            'seed':self.settings['seed'],'spawn_index':idx,'weather':self.settings.get('weather','ClearNoon'),
            'route_length_m':self._arcs[-1],'traffic_lights':[
                {'id':str(a.id),'state':str(a.get_state()),'elapsed_s':float(a.get_elapsed_time())}
                for a in self.world.get_actors().filter('traffic.traffic_light*')],
            'traffic_light_phase_policy':'保留服务器灯周期；seed不保证初始相位相同',
            'npc_controller':'公开路线跟踪+当前实体间距+信号+停止标志；未验证为专家驾驶'})
        for sign in self.world.get_actors().filter('traffic.stop*'):
            if hasattr(sign,'get_stop_waypoints'):
                for swp in sign.get_stop_waypoints():self._stops.append((str(sign.id),swp))
            else:
                center=sign.get_transform().transform(sign.trigger_volume.location)
                swp=self.road_map.get_waypoint(center)
                if swp is not None:self._stops.append((str(sign.id),swp))

    def _trigger(self,row,elapsed):
        if row['triggered_at'] is not None:return
        source=row['source'];trigger=source.get('trigger',{})
        if source['behavior'] not in ('cross','cut_in','stop_resume'):return
        distance=_distance(self.ego.get_location(),row['actor'].get_location())
        if elapsed>=trigger.get('time_s',0.) and (trigger.get('ego_distance_m') is None
                or distance<=trigger['ego_distance_m']):
            row['triggered_at']=elapsed
            self.events.append({'event':'interaction_triggered','time':elapsed,'role':source['role'],
                                'behavior':source['behavior'],'distance_m':distance})

    @staticmethod
    def _speed(actor):
        v=actor.get_velocity();return math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)

    def _safe_speed(self,row,target,elapsed):
        actor=row['actor'];loc=actor.get_location();t=actor.get_transform()
        f=t.get_forward_vector();right=t.get_right_vector();box=actor.bounding_box
        # 这里只使用场景内当前实际位置。不给核心规划器额外读取权限。
        for other in self.actors:
            if other.id==actor.id or not other.is_alive:continue
            p=other.get_location();dx,dy=p.x-loc.x,p.y-loc.y
            ahead=dx*f.x+dy*f.y;lateral=abs(dx*right.x+dy*right.y)
            if abs(p.z-loc.z)>2. or ahead<=0:continue
            obox=other.bounding_box
            if lateral>abs(box.extent.y)+max(abs(obox.extent.x),abs(obox.extent.y))+.4:continue
            gap=ahead-abs(box.extent.x)-max(abs(obox.extent.x),abs(obox.extent.y))
            ov=other.get_velocity();lead_speed=max(0.,ov.x*f.x+ov.y*f.y)
            target=min(target,max(0.,lead_speed+(gap-3.)/1.5),math.sqrt(4.*max(0.,gap-2.)))
        if actor.is_at_traffic_light():
            state=str(actor.get_traffic_light_state()).split('.')[-1]
            if state in ('Red','Yellow'):target=0.
        current=self.road_map.get_waypoint(loc)
        for sign_id,wp in self._stops:
            if current is None or (current.road_id,current.lane_id)!=(wp.road_id,wp.lane_id):continue
            p=wp.transform.location;distance=(p.x-loc.x)*f.x+(p.y-loc.y)*f.y
            if not -1.<=distance<=20.:continue
            seen=row['stop_signs'].get(sign_id)
            if seen is not None and elapsed-seen>=2.:continue
            target=min(target,math.sqrt(4.*max(0.,distance-3.)))
            if distance<=4. and self._speed(actor)<.15:
                row['stop_signs'].setdefault(sign_id,elapsed)
        return target

    def _follow(self,row,elapsed):
        actor=row['actor'];source=row['source'];route=row['route'];loc=actor.get_location()
        if len(route)<2:raise RuntimeError('NPC公开行驶路线不足：'+source['role'])
        # 只在局部窗口推进索引，避免回环路线上跳到空间接近的远端。
        lo=max(0,row['index']-2);hi=min(len(route),row['index']+100)
        nearest=min(range(lo,hi),key=lambda i:_distance(loc,route[i].transform.location))
        row['index']=max(row['index'],nearest)
        speed=self._speed(actor);target_index=min(nearest+max(3,int(max(6.,speed*.8)/2)),len(route)-1)
        target=route[target_index].transform.location
        tx,ty=target.x,target.y
        request=float(source['speed_mps'])
        if source['behavior']=='cut_in':
            fraction=0. if row['triggered_at'] is None else min(1.,(elapsed-row['triggered_at'])/source.get('duration_s',4.))
            blend=fraction*fraction*(3.-2.*fraction)
            if fraction<1.:
                side=source['placement']['lane']
                _crossable(route[nearest],side,self.carla)
                target_side=_crossable(route[target_index],side,self.carla).transform.location
                tx,ty=(1.-blend)*target_side.x+blend*tx,(1.-blend)*target_side.y+blend*ty
                if row['triggered_at'] is None:
                    end=row['plan']['corridor_end_m']-source['speed_mps']*source.get('duration_s',4.)-10.
                    request=min(request,math.sqrt(4.*max(0.,end-self._arcs[nearest])))
            elif not row['finished']:
                row['finished']=True
                self.events.append({'event':'cut_in_command_completed','time':elapsed,'role':source['role'],
                    'note':'几何过渡指令结束，实际横向到达仍需从轨迹核查'})
        if source['behavior']=='stop_resume' and row['triggered_at'] is not None and not row['finished']:
            if speed<.15 and row['stop_started_at'] is None:row['stop_started_at']=elapsed
            if row['stop_started_at'] is None or elapsed-row['stop_started_at']<source.get('stop_duration_s',4.):
                actor.apply_control(self.carla.VehicleControl(brake=1.))
                return
            row['finished']=True
            self.events.append({'event':'stop_resume_released','time':elapsed,'role':source['role']})
        remaining=sum(_distance(a.transform.location,b.transform.location) for a,b in zip(route[nearest:],route[nearest+1:]))
        request=min(request,math.sqrt(4.*max(0.,remaining-3.)))
        request=self._safe_speed(row,request,elapsed)
        alpha=_angle(math.atan2(ty-loc.y,tx-loc.x)-math.radians(actor.get_transform().rotation.yaw))
        if row['wheelbase'] is None:
            try:from .carla_bridge import wheel_geometry
            except ImportError:from carla_bridge import wheel_geometry
            row['wheelbase']=wheel_geometry(actor.get_physics_control().wheels,
                actor.get_transform().get_inverse_matrix(),actor.bounding_box.location.x)[0]
        steering=math.atan2(2.*row['wheelbase']*math.sin(alpha),max(1.,math.hypot(tx-loc.x,ty-loc.y)))
        steering=max(-.5,min(.5,steering)) if remaining>3. else 0.
        # 急弯提前降低NPC速度；实际自车依然完全由主MPC控制。
        if abs(steering)>.05:request=min(request,math.sqrt(2.5*row['wheelbase']/abs(math.tan(steering))))
        actor.apply_ackermann_control(self.carla.VehicleAckermannControl(steer=steering,speed=request))

    def before_tick(self,elapsed_s,dt):
        if self._closed:raise RuntimeError('场景已关闭')
        _finite(elapsed_s,'仿真时间');_finite(dt,'仿真步长',1e-9)
        for row in self._rows:
            actor=row['actor'];source=row['source']
            if not actor.is_alive:raise RuntimeError('脚本演员意外失效：'+source['role'])
            self._trigger(row,elapsed_s)
            if source['type']=='pedestrian':
                endpoint=row['plan']['end'].transform.location;loc=actor.get_location()
                dx,dy=endpoint.x-loc.x,endpoint.y-loc.y;distance=math.hypot(dx,dy)
                if distance<.6 and row['triggered_at'] is not None and not row['finished']:
                    row['finished']=True
                    self.events.append({'event':'pedestrian_crossing_completed','time':elapsed_s,'role':source['role']})
                speed=source['speed_mps'] if row['triggered_at'] is not None and not row['finished'] else 0.
                direction=self.carla.Vector3D(x=dx/max(distance,1e-9),y=dy/max(distance,1e-9),z=0.)
                actor.apply_control(self.carla.WalkerControl(direction=direction,speed=speed))
            elif source['type']=='vehicle' and source['behavior']!='parked':self._follow(row,elapsed_s)

    def close(self):
        if self._closed:return
        self._closed=True;failures=[]
        for actor in reversed(self.actors):
            try:
                if actor.is_alive and actor.destroy() is False:failures.append('destroy返回False')
            except Exception as exc:failures.append(type(exc).__name__)
        if self._weather is not None:
            try:self.world.set_weather(self._weather)
            except Exception as exc:failures.append('weather_restore:'+type(exc).__name__)
        if failures:self.events.append({'event':'cleanup_error','errors':failures})
