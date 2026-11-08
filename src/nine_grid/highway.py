"""highway-env 1.12.1 的连续控制、公共地图和可见状态适配。"""
from __future__ import annotations

import copy
import logging
import math
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from shapely.geometry import Point, Polygon,LineString
from shapely.ops import unary_union

from .contracts import Actor, Lane, SceneSnapshot
from .geometry import (actor_shape, clean_polyline, polyline_lengths, project_polyline,
                       sample_polyline, rectangle_corners, wrap_angle)
from .configuration import highway_settings


def _lane_id(index):
    return "|".join(map(str, index))


class _ContinuousRewards:
    """禁用与研究无关的离散动作奖励，避免 ndarray 成员比较异常。"""
    def _reward(self, action):
        return 0.

    def _rewards(self, action):
        return {}

    def _info(self, obs, action=None):
        return {"crashed": bool(self.vehicle.crashed)}

    def render(self):
        # 包括环境自动绘制的物理子步，live与录像始终经过同一原生图层。
        if self.viewer is None:
            from .video import PlannedTrajectoryViewer
            self.viewer = PlannedTrajectoryViewer(self)
        return super().render()

    def step(self,action):
        from highway_env.envs.common.abstract import AbstractEnv
        result=AbstractEnv.step(self,action)
        if hasattr(self,"_clear_vehicles"):self._clear_vehicles()
        # 禁止调用原 IntersectionEnv.step 的每控制步随机生成，连空生成的随机数消耗也隔离。
        return result


class HighwayAdapter:
    def __init__(self, config: dict, seed: int, episode_id: str, live=False):
        self.env = None
        self._video_recorder = None
        self._traffic_class_name = None
        try:
            self._initialize(config, seed, episode_id, live)
        except BaseException:
            try:
                self.close()
            except Exception:
                logging.getLogger(__name__).exception('初始化失败后的仿真清理异常；保留原初始化错误')
            raise

    def _initialize(self, config: dict, seed: int, episode_id: str, live=False):
        from highway_env.envs.highway_env import HighwayEnv
        from highway_env.envs.merge_env import ConnectedLaneMergeEnv
        from highway_env.envs.roundabout_env import ConnectedLaneRoundaboutEnv
        from highway_env.envs.intersection_env import ConnectedLaneIntersectionEnv
        from highway_env.vehicle.behavior import IDMVehicle

        self.config = copy.deepcopy(config)
        angular_step=float(self.config.get("perception",{}).get("visibility_angular_step_deg",.25))
        angles=np.arange(-np.pi,np.pi,np.deg2rad(angular_step))
        self._visibility_directions=np.stack((np.cos(angles),np.sin(angles)),axis=1)
        self.scenario = config.get("scenario", "highway")
        if isinstance(self.scenario, dict):
            self.scenario = self.scenario.get("name", "highway")
        bases = {"highway": HighwayEnv, "merge": ConnectedLaneMergeEnv,
                 "roundabout": ConnectedLaneRoundaboutEnv, "intersection": ConnectedLaneIntersectionEnv}
        if self.scenario not in bases:
            raise ValueError(f"未知场景：{self.scenario}")
        self.settings = highway_settings(config,self.scenario)
        self.episode_id = episode_id
        self.dt = 1 / config.get("clock", {}).get("planner_hz", 10)
        self.history_dt=1/float(config.get("clock",{}).get("history_hz",2))
        self.history_seconds=float(config.get("clock",{}).get("history_seconds",3))
        self.duration = float(config.get("duration", config.get("clock", {}).get("episode_seconds", 30)))
        self.physics_hz = int(config.get("clock", {}).get("physics_hz", 50))
        if abs(self.physics_hz * self.dt - round(self.physics_hz * self.dt)) > 1e-9:
            raise ValueError("物理频率必须是规划频率的整数倍")
        mpc = config.get("mpc", {})
        self.acc_range = [mpc.get("acceleration_min_mps2", -6), mpc.get("acceleration_max_mps2", 3)]
        self.steer_max = mpc.get("steering_abs_max_rad", .6)
        self.desired_speed = float(self.settings.get("desired_speed_mps", {"highway":25,"merge":15,"roundabout":8,"intersection":10}[self.scenario]))
        count = int(config.get("vehicles", self.settings.get("vehicles_count", {"highway":12,"merge":4,"roundabout":4,"intersection":8}[self.scenario])))
        if count<0:raise ValueError("初始交通车辆数量不能为负数")
        # 路口原实现每次 policy step 生成交通；这里改为每仿真秒统一一次。
        self.spawn_probability = float(self.settings.get("spawn_probability", .6 if self.scenario == "intersection" and count else 0.))
        # 原生路口会修改交通车辆类属性；每个适配器独立派生类防止跨场景污染。
        self._traffic_class_name=f"ResearchTraffic_{id(self):x}"
        globals()[self._traffic_class_name]=type(self._traffic_class_name,(IDMVehicle,),{})
        env_cfg = {**self.settings.get('native',{}),
                   "simulation_frequency": self.physics_hz, "policy_frequency": round(1/self.dt),
                   "other_vehicles_type":f"{__name__}.{self._traffic_class_name}",
                   "duration": self.duration, "vehicles_count": count, "initial_vehicle_count": count,
                   "spawn_probability": 0., "lanes_count": int(self.settings.get("lanes_count",4)),
                   "destination": self.settings.get("destination", "o1"),
                   "observation": {"type":"Kinematics","vehicles_count":1,"observe_intentions":False},
                   "action": {"type":"ContinuousAction","dynamical":False,
                              "acceleration_range":self.acc_range,"steering_range":[-self.steer_max,self.steer_max],
                              "speed_range":[0.,mpc.get("maximum_speed_mps",33.3333333)]}}
        cls = type(f"Research{bases[self.scenario].__name__}", (_ContinuousRewards, bases[self.scenario]), {})
        self.env = cls(config=env_cfg, render_mode="human" if live else "rgb_array")
        self.env.reset(seed=int(seed))
        self.live = live
        ego = self.env.vehicle
        ego.speed = float(self.settings.get("initial_speed_mps", self.desired_speed))
        initial_longitudinal=self.settings.get("initial_longitudinal_m",110. if self.scenario=="roundabout" else None)
        if initial_longitudinal is not None:
            ego.position=ego.lane.position(float(initial_longitudinal),0)
            ego.heading=ego.lane.heading_at(float(initial_longitudinal))
            ego.on_state_update()
        ego.MIN_SPEED, ego.MAX_SPEED = 0., mpc.get("maximum_speed_mps",33.3333333)
        others = [v for v in self.env.road.vehicles if v is not ego]
        if self.scenario in ("merge","roundabout") and count>len(others):
            raise ValueError(f"{self.scenario} 原生模板最多支持 {len(others)} 辆初始交通车辆，收到 {count}；不会静默减少数量")
        # 固定车辆场景的原生构造不服从 vehicles_count；明确裁剪而非假空场。
        self.env.road.vehicles = [ego] + others[:count]
        self._ids = {id(ego): "ego"}
        self._identity_references = {id(ego):ego}
        self._next_id = 0
        self._history = {}
        self._step = 0
        self._next_spawn = 1.
        self._previous_route_s = None
        self._ego_lane_id=_lane_id(ego.lane_index)
        self._pending_lane=None
        self._pending_lane_since=None
        self._reference_revision=0
        self._route_progress_offset=0.
        self._right_handed=False
        self._previous_control = np.zeros(2)
        self._cached_snapshot = None
        self._lane_indexes = {}
        self._geometry_precision=float(config.get('map_geometry',{}).get('coordinate_precision_m',1e-9))
        self._arc_knots = self._shared_arc_knots()
        self.lanes, self.drivable = self._map()
        self.route_lane_ids, self.route, self.route_diagnostics = self._navigation()
        self._goal_s = self._goal_arc()
        # 原生 highway 屏幕坐标 y 向下；平台契约使用右手世界坐标 Y=-y。
        for lane in self.lanes:
            lane.center[:,1]*=-1;lane.polygon[:,1]*=-1
            lane.left,lane.right=lane.right,lane.left
            for boundary in lane.boundaries:
                boundary["points"][:,1]*=-1
                if "original_points" in boundary:boundary["original_points"]=boundary["original_points"]*np.array([1.,-1.])
        self.route[:,1]*=-1
        from shapely import transform
        self.drivable=transform(self.drivable,lambda xy:xy*np.array([1.,-1.]))
        # 本轮道路与容差固定，执行审计复用相同的几何结果。
        self._audit_drivable=self.drivable.buffer(float(self.config.get('mpc',{}).get('body_constraint_tolerance_m',.001)))
        self._right_handed=True
        self.physical_lane_centers={}
        physical_polygons=[]
        for key,index in self._lane_indexes.items():
            lane=self.env.road.network.get_lane(index)
            ss=self._lane_samples(lane,0.,lane.length)
            self.physical_lane_centers[key]=self._lane_points(lane,ss,0.)*np.array([1.,-1.])
            left=self._lane_points(lane,ss,.5)
            right=self._lane_points(lane,ss,-.5)
            physical_polygons.append(Polygon(np.vstack((left,right[::-1]))*np.array([1.,-1.])).buffer(0))
        # 原生道路真实端点供显示；规划的 on_lane 端部容差不能变成绘图突出。
        self.bev_drivable=self._road_union(physical_polygons)
        self._video_recorder=None
        self._next_history = 0.

    def _shared_arc_knots(self):
        """同圆心车道共用绝对角度网格，防止共边的内接折线产生假缝。"""
        groups={}
        for destinations in self.env.road.network.graph.values():
            for lanes in destinations.values():
                for lane in lanes:
                    if type(lane).__name__!="CircularLane":continue
                    key=tuple(np.round(lane.center,12))
                    endpoints=[lane.start_phase+lane.direction*s/lane.radius
                               for s in (-5.,0.,lane.length,lane.length+5.)]
                    groups.setdefault(key,[]).extend(endpoints)
        result={}
        step=float(self.config.get('map_geometry',{}).get('circular_angle_step_rad',.02))
        for key,endpoints in groups.items():
            low,high=min(endpoints),max(endpoints)
            grid=np.arange(math.floor(low/step),math.ceil(high/step)+1)*step
            result[key]=np.unique(np.concatenate((grid,endpoints)))
        return result

    def _lane_samples(self,lane,start,end):
        if type(lane).__name__=="CircularLane":
            first=lane.start_phase+lane.direction*start/lane.radius
            last=lane.start_phase+lane.direction*end/lane.radius
            knots=self._arc_knots[tuple(np.round(lane.center,12))]
            angles=knots[(knots>=min(first,last)-1e-12)&(knots<=max(first,last)+1e-12)]
            coordinates=(angles-lane.start_phase)*lane.radius/lane.direction
            return np.sort(np.clip(coordinates,start,end))
        return np.linspace(start,end,2 if type(lane).__name__=="StraightLane"
                           else max(2,math.ceil(end-start)+1))

    def _lane_points(self,lane,samples,width_fraction):
        if type(lane).__name__=="CircularLane":
            angles=lane.start_phase+lane.direction*samples/lane.radius
            knots=self._arc_knots[tuple(np.round(lane.center,12))]
            indexes=np.searchsorted(knots,angles).clip(0,len(knots)-1)
            previous=np.maximum(indexes-1,0)
            indexes=np.where(np.abs(knots[previous]-angles)<np.abs(knots[indexes]-angles),previous,indexes)
            # 直接使用公共角度，避免 s→角度往返的浮点误差再次裂开共边。
            angles=knots[indexes]
            radii=lane.radius-width_fraction*lane.width*lane.direction
            return lane.center+radii*np.stack((np.cos(angles),np.sin(angles)),axis=1)
        return np.asarray([lane.position(s,width_fraction*lane.width_at(s)) for s in samples])

    def _road_union(self,polygons):
        from shapely import set_precision
        # 纳米级统一坐标精度消除共边浮点往返裂缝，非道路膨胀或碰撞容差。
        return unary_union([set_precision(p,self._geometry_precision) for p in polygons])

    def _map(self):
        lanes = []
        network = self.env.road.network
        for a, destinations in network.graph.items():
            for b, road_lanes in destinations.items():
                for i, lane in enumerate(road_lanes):
                    index = (a,b,i); key = _lane_id(index)
                    self._lane_indexes[key] = index
                    # highway 的 on_lane 明确定义前后各 5m 端部，不能只裁 [0,length]。
                    ss = self._lane_samples(lane,-5.,lane.length+5.)
                    center_s=self._lane_samples(lane,0.,lane.length)
                    center = self._lane_points(lane,center_s,0.)
                    left = self._lane_points(lane,ss,.5)
                    right = self._lane_points(lane,ss,-.5)
                    polygon = np.vstack((left, right[::-1]))
                    successors = [_lane_id((b,c,j)) for c, next_lanes in network.graph.get(b,{}).items()
                                  for j in range(len(next_lanes))]
                    item = Lane(key,center,polygon,float(lane.width_at(0)),float(lane.speed_limit or self.desired_speed),
                                successors=successors,priority=int(lane.priority))
                    for j, other in enumerate(road_lanes):
                        if abs(i-j) != 1:
                            continue
                        delta = other.position(0,0)-lane.position(0,0)
                        normal = np.array([-np.sin(lane.heading_at(0)),np.cos(lane.heading_at(0))])
                        if delta @ normal > 0: item.left = _lane_id((a,b,j))
                        else: item.right = _lane_id((a,b,j))
                    for side,line_type in enumerate(lane.line_types):
                        physical_s=self._lane_samples(lane,0.,lane.length)
                        points=self._lane_points(lane,physical_s,(-.5 if side==0 else .5))
                        item.boundaries.append({"id":f"{key}:boundary:{side}","points":points,
                                                "crossable":int(line_type) in (0,1),"line_type":int(line_type),
                                                "kind":"lane_boundary"})
                    lanes.append(item)
        # 只有实际共边且行驶方向相反，才标注道路中心分隔线。
        for item in lanes:
            for boundary in item.boundaries:
                line=LineString(boundary['points'])
                middle=line.interpolate(.5,normalized=True)
                own_s=LineString(item.center).project(middle)
                own_center=LineString(item.center)
                own_direction=np.asarray(own_center.interpolate(min(own_s+.5,own_center.length)).coords)[0]-np.asarray(own_center.interpolate(max(0.,own_s-.5)).coords)[0]
                for other in lanes:
                    if other.id==item.id:continue
                    other_center=LineString(other.center)
                    other_s=other_center.project(middle)
                    other_direction=np.asarray(other_center.interpolate(min(other_s+.5,other_center.length)).coords)[0]-np.asarray(other_center.interpolate(max(0.,other_s-.5)).coords)[0]
                    if np.dot(own_direction,other_direction)>=-.8*np.linalg.norm(own_direction)*np.linalg.norm(other_direction):continue
                    if any(line.intersection(LineString(edge['points']).buffer(.02)).length>max(1.,min(line.length,LineString(edge['points']).length)*.5) for edge in other.boundaries):
                        boundary['kind']='road_center';break
        polygons = [Polygon(lane.polygon).buffer(0) for lane in lanes]
        # 部分原生场景将环岛整圈画成连续线，即使拓扑允许接入。
        # 仅在共享分支节点且方向相交的合法连接口裁剪，不清空整条重叠边线。
        degrees={node:sum(1 for a,dest in network.graph.items() for b in dest if a==node or b==node)
                 for node in {node for a,dest in network.graph.items() for b in dest for node in (a,b)}}
        for item,polygon in zip(lanes,polygons):
            a,b,_=self._lane_indexes[item.id];cuts=[]
            native=network.get_lane(self._lane_indexes[item.id])
            for other,other_polygon in zip(lanes,polygons):
                c,d,_=self._lane_indexes[other.id]
                if (a,b)==(c,d):continue
                common={a,b}&{c,d}
                if not any(degrees[node]>2 for node in common):continue
                other_native=network.get_lane(self._lane_indexes[other.id])
                for node in common:
                    own_s=0 if node==a else native.length
                    other_s=0 if node==c else other_native.length
                    if abs(np.sin(native.heading_at(own_s)-other_native.heading_at(other_s)))<.2:continue
                    center=(native.position(own_s,0)+other_native.position(other_s,0))/2
                    # 明确限制连接口半径，保留连接区域以外的原生实线。
                    aperture=Point(center).buffer(float(self.settings.get("connection_boundary_clip_radius_m",12.)))
                    cuts.append(other_polygon.intersection(aperture))
            if not cuts:continue
            cut=unary_union(cuts)
            revised=[]
            for boundary in item.boundaries:
                if boundary["crossable"]:revised.append(boundary);continue
                original=boundary["points"]
                geometry=LineString(original).difference(cut)
                fragments=[geometry] if geometry.geom_type=="LineString" else list(getattr(geometry,"geoms",[]))
                for number,fragment in enumerate(fragments):
                    if fragment.length<.05:continue
                    revised.append({**boundary,"id":f"{boundary['id']}:part:{number}","points":np.asarray(fragment.coords),
                                    "original_boundary_id":boundary["id"],"original_points":original,
                                    "adaptation":"public_connected_junction_aperture"})
            item.boundaries=revised
        return lanes,self._road_union(polygons)

    def _navigation(self,start_override=None):
        network = self.env.road.network
        start = self.env.vehicle.lane_index if start_override is None else start_override
        destination = self.settings.get("destination", {"highway":"1","merge":"d","roundabout":"nxr","intersection":"o1"}[self.scenario])
        nodes = network.shortest_path(start[1], destination)
        if not nodes and start[1] != destination:
            raise ValueError(f"自车导航目的地不可达：{destination}")
        indexes = [start]
        for a,b in zip(nodes,nodes[1:]):
            previous = network.get_lane(indexes[-1])
            candidates = network.graph[a][b]
            if len(network.graph[indexes[-1][0]][indexes[-1][1]]) == len(candidates):
                index = min(indexes[-1][2],len(candidates)-1)
            else:
                endpoint = previous.position(previous.length,0)
                index = min(range(len(candidates)), key=lambda j:np.linalg.norm(candidates[j].position(0,0)-endpoint))
            indexes.append((a,b,index))
        # 此路线仅属于自车；不访问周车路线或目标车道。
        self.env.vehicle.route = list(indexes)
        pieces = []
        for index in indexes:
            lane = network.get_lane(index)
            ss = np.linspace(0,lane.length,2 if type(lane).__name__=="StraightLane" else max(2,math.ceil(lane.length/.5)+1))
            pieces.append(np.array([lane.position(s,0) for s in ss]))
        route = pieces[0]
        connections = []
        for piece in pieces[1:]:
            gap = float(np.linalg.norm(route[-1]-piece[0]))
            h0 = np.arctan2(*(route[-1]-route[-2])[::-1]); h1=np.arctan2(*(piece[1]-piece[0])[::-1])
            if gap > .2 or abs(wrap_angle(h1-h0)) > .1:
                route,piece,detail = self._connect(route,piece)
                connections.append(detail)
            route = np.vstack((route,piece))
        route = clean_polyline(route)
        return [_lane_id(i) for i in indexes],route,{"connections":connections,"destination":destination}

    def _connect(self, before, after):
        """只在实际道路内搜索导航连接，不改变道路几何或规划硬约束。"""
        best = None
        if self._right_handed:
            from shapely import transform
            road_test=transform(self.drivable,lambda xy:xy*np.array([1.,-1.])).buffer(1e-6)
        else:road_test=self.drivable.buffer(1e-6)
        before_arc=polyline_lengths(before);after_arc=polyline_lengths(after)
        length=float(self.env.vehicle.LENGTH);width=float(self.env.vehicle.WIDTH)
        for trim in (4.,6.,8.,10.,12.,16.,20.):
            s0=max(.5,before_arc[-1]-trim);s1=min(trim,after_arc[-1]-.5)
            p0,p1=sample_polyline(before,s0),sample_polyline(after,s1)
            t0=sample_polyline(before,s0+.25)-p0;t0/=np.linalg.norm(t0)
            t1=p1-sample_polyline(after,s1-.25);t1/=np.linalg.norm(t1)
            distance=np.linalg.norm(p1-p0)
            for scale in (.75,1.,1.25,1.5):
                curve=CubicHermiteSpline([0.,1.],np.stack((p0,p1)),np.stack((t0,t1))*distance*scale)
                ts=np.linspace(0,1,max(30,math.ceil(distance/.2)))
                chord=(p1-p0)/distance;normal=np.array([-chord[1],chord[0]])
                # 连接处的车道中心线不一定是整车可通过的质心路径。
                # 零端值、零端导数的横向形变保持接点与切向连续。
                for shift in (0.,-.5,.5,-1.,1.,-1.5,1.5,-2.,2.):
                    points=curve(ts)+shift*np.sin(np.pi*ts)[:,None]**2*normal
                    first=curve(ts,1)+shift*np.pi*np.sin(2*np.pi*ts)[:,None]*normal
                    second=curve(ts,2)+shift*2*np.pi**2*np.cos(2*np.pi*ts)[:,None]*normal
                    curvature=np.abs(first[:,0]*second[:,1]-first[:,1]*second[:,0])/np.maximum(np.linalg.norm(first,axis=1)**3,1e-9)
                    tangents=np.unwrap(np.arctan2(first[:,1],first[:,0]))
                    previous_tangent=p0-sample_polyline(before,max(0,s0-.5))
                    initial_curvature=float(wrap_angle(np.arctan2(t0[1],t0[0])-np.arctan2(previous_tangent[1],previous_tangent[0])))/.375
                    headings=np.empty(len(points));headings[0]=tangents[0]-np.arcsin(np.clip(length*initial_curvature/2,-.9,.9))
                    # 路径切向与质心车体朝向不同。以弧长积分同一自行车模型。
                    for j,ds in enumerate(np.linalg.norm(np.diff(points,axis=0),axis=1)):
                        theta=(tangents[j]+tangents[j+1])/2
                        midpoint=headings[j]+ds/length*np.sin(tangents[j]-headings[j])
                        headings[j+1]=headings[j]+ds*2/length*np.sin(theta-midpoint)
                    steering=np.arctan(2*np.tan(tangents-headings))
                    outside=sum(not road_test.covers(Polygon(rectangle_corners(p,h,length,width))) for p,h in zip(points,headings))
                    excess=max(0.,float(np.abs(steering).max())-self.steer_max)
                    score=outside*10000+excess*1000+float(curvature.max())+scale*.001+abs(shift)*.0001
                    if best is None or score<best[0]:best=(score,s0,s1,points,outside,float(curvature.max()),float(np.abs(steering).max()))
        _,s0,s1,points,outside,curvature,max_steering=best
        return np.vstack((before[before_arc<s0],points)),after[after_arc>s1],{"outside_samples":outside,"maximum_curvature":curvature,"maximum_steering":max_steering,"method":"road_checked_cubic_hermite"}

    def _goal_arc(self):
        if self.scenario == "highway": return float("inf")
        network=self.env.road.network
        final=network.get_lane(self._lane_indexes[self.route_lane_ids[-1]])
        distance=float(self.settings.get("exit_distance_m",{"merge":60.,"roundabout":30.,"intersection":25.}[self.scenario]))
        target=final.position(min(distance,final.length-5),0)
        return project_polyline(target,self.route)[0]

    def _identity(self,obj):
        key=id(obj)
        if key not in self._ids:
            self._next_id+=1;self._ids[key]=f"object_{self._next_id:05d}"
            self._identity_references[key]=obj
        return self._ids[key]

    def _stable_ego_lane(self,ego,timestamp):
        """仅相邻并行车道采用半车身及持续时间门槛；正常路段接续即时更新。"""
        candidate=ego.lane_id
        if candidate==self._ego_lane_id:
            self._pending_lane=None;self._pending_lane_since=None
            return
        current=self._lane_indexes.get(self._ego_lane_id)
        proposed=self._lane_indexes.get(candidate)
        adjacent=current is not None and proposed is not None and current[:2]==proposed[:2] and abs(current[2]-proposed[2])==1
        if not adjacent:
            self._ego_lane_id=candidate;self._pending_lane=None;self._pending_lane_since=None
            return
        lane=next(l for l in self.lanes if l.id==candidate)
        polygon=Polygon(lane.polygon);body=actor_shape(ego)
        minimum=float(self.config.get("coordinates",{}).get("lane_switch_overlap_fraction",.5))
        if not polygon.covers(Point(ego.position)) or body.intersection(polygon).area/body.area<=minimum:
            self._pending_lane=None;self._pending_lane_since=None
            return
        if self._pending_lane!=candidate:
            self._pending_lane=candidate;self._pending_lane_since=timestamp
        hold=float(self.config.get("coordinates",{}).get("lane_switch_hold_s",.3))
        if timestamp-float(self._pending_lane_since)+1e-8<hold:
            return
        old_progress=project_polyline(ego.position,self.route,self._previous_route_s,ego.heading)[0]+self._route_progress_offset
        self._ego_lane_id=candidate;self._pending_lane=None;self._pending_lane_since=None
        self.route_lane_ids,native_route,self.route_diagnostics=self._navigation(proposed)
        self.route=native_route
        self._goal_s=self._goal_arc()
        self.route[:,1]*=-1
        new_progress=project_polyline(ego.position,self.route,heading=ego.heading)[0]
        self._route_progress_offset=old_progress-new_progress
        self._reference_revision+=1;self._previous_route_s=None

    def _actor(self,obj):
        from highway_env.vehicle.objects import Obstacle
        kind=getattr(obj,"research_kind", "static_obstacle" if isinstance(obj,Obstacle) else "vehicle")
        speed=float(getattr(obj,"speed",0.));heading=float(obj.heading)
        control=getattr(obj,"action",{})
        steer=float(control.get("steering",0.));acc=float(control.get("acceleration",0.))
        beta=np.arctan(.5*np.tan(steer));motion=heading+beta
        velocity=speed*np.array([np.cos(motion),np.sin(motion)])
        omega=2*speed/max(float(obj.LENGTH),.1)*np.sin(beta)
        acceleration=acc*np.array([np.cos(motion),np.sin(motion)])+speed*omega*np.array([-np.sin(motion),np.cos(motion)])
        lane_id=_lane_id(obj.lane_index) if getattr(obj,"lane_index",None) else None
        return Actor(self._identity(obj),kind,np.asarray(obj.position,float)*np.array([1.,-1.]),velocity*np.array([1.,-1.]),acceleration*np.array([1.,-1.]),
                     -heading,float(obj.LENGTH),float(obj.WIDTH),lane_id=lane_id,
                     radius=float(obj.WIDTH)/2 if kind=="pedestrian" else None)

    def _visibility(self,ego,actors):
        radius=float(self.config.get("perception",{}).get("radius_m",120.))
        directions=self._visibility_directions
        in_range=[a for a in actors if np.linalg.norm(a.position-ego.position)<=radius]
        distance=np.full(len(directions),radius)
        shapes=[actor_shape(a) for a in in_range]
        if in_range:
            # 计算每条可见射线与公开物体外轮廓的最近交点。
            rings=[np.array(shape.exterior.coords) for shape in shapes]
            starts=np.vstack([r[:-1] for r in rings])-ego.position
            segments=np.vstack([np.diff(r,axis=0) for r in rings])
            cross=directions[:,None,0]*segments[None,:,1]-directions[:,None,1]*segments[None,:,0]
            safe=np.where(np.abs(cross)<1e-12,np.nan,cross)
            t=(starts[:,0]*segments[:,1]-starts[:,1]*segments[:,0])[None,:]/safe
            u=(starts[None,:,0]*directions[:,None,1]-starts[None,:,1]*directions[:,None,0])/safe
            valid=(t>=0)&(u>=0)&(u<=1)&np.isfinite(t)
            distance=np.minimum(distance,np.min(np.where(valid,t,np.inf),axis=1))
        polygon=Polygon(ego.position+directions*distance[:,None]).buffer(0)
        visible_boundary=polygon.buffer(.015)
        visible=[a for a,shape in zip(in_range,shapes) if shape.boundary.intersects(visible_boundary)]
        return visible,polygon

    def snapshot(self):
        if self._cached_snapshot is not None:return self._cached_snapshot
        ego=self._actor(self.env.vehicle)
        timestamp=self._step*self.dt
        self._stable_ego_lane(ego,timestamp)
        from highway_env.vehicle.objects import Obstacle
        radius=float(self.config.get("perception",{}).get("radius_m",120.))
        objects=[v for v in [*self.env.road.vehicles,*self.env.road.objects] if v is not self.env.vehicle
                 and np.linalg.norm(v.position-self.env.vehicle.position)<=radius]
        # 可见性阶段只访问几何；通过判定以后才提取速度、加速度等状态。
        raw=[Actor(self._identity(v),getattr(v,"research_kind","static_obstacle" if isinstance(v,Obstacle) else "vehicle"),
                   np.asarray(v.position)*np.array([1.,-1.]),np.zeros(2),np.zeros(2),-float(v.heading),float(v.LENGTH),float(v.WIDTH)) for v in objects]
        visible_geometry,visibility=self._visibility(ego,raw)
        visible_ids={a.id for a in visible_geometry}
        actors=[self._actor(v) for v in objects if self._identity(v) in visible_ids]
        if timestamp+1e-8>=self._next_history:
            for actor in [ego]+actors:
                hist=self._history.setdefault(actor.id,[])
                hist.append({"time":timestamp,"position":actor.position.tolist(),"velocity":actor.velocity.tolist(),
                             "acceleration":actor.acceleration.tolist(),"heading":actor.heading,"observed":True})
            self._next_history=round(timestamp+self.history_dt,8)
        for identity in list(self._history):
            self._history[identity]=[h for h in self._history[identity] if timestamp-h["time"]<=self.history_seconds+1e-8]
            if not self._history[identity]:del self._history[identity]
        for actor in [ego]+actors:actor.history=copy.deepcopy(self._history.get(actor.id,[]))
        route_s=project_polyline(ego.position,self.route,self._previous_route_s,ego.heading)[0]
        self._previous_route_s=route_s
        rules={"desired_speed":self.desired_speed,"must_wait":False,"wait_reason":None,"stop_s":None,
               "speed_limit":float(self.env.vehicle.lane.speed_limit or self.desired_speed),
               "crosswalks":[],"traffic_lights":[],"map_available":{"crosswalk":True,"traffic_light":False,"stop_line":False},
               "road_priority_available":self.scenario=="intersection","goal_s":self._goal_s}
        diagnostics={"route_s":route_s,"previous_control":self._previous_control.tolist(),
                     "lane_map_complete": True,
                     "reference_revision":self._reference_revision,
                     "stable_navigation_lane_id":self._ego_lane_id,"navigation_progress_m":route_s+self._route_progress_offset,
                     "reference_anchor_s":route_s,"has_occlusion":len(actors)<len(raw),
                     "visibility_polygon":np.array(visibility.exterior.coords).tolist(),
                     "historical_only_objects":{key:value for key,value in self._history.items() if key not in {a.id for a in [ego]+actors}},
                     "navigation":self.route_diagnostics,"acceleration_source":"kinematic_dynamics_current_hold", "physics_hz":self.physics_hz}
        self._cached_snapshot=SceneSnapshot(f"{self.episode_id}:{self._step:06d}",timestamp,ego,actors,self.lanes,
                                           self.route,self.route_lane_ids,self.drivable,self.scenario,rules=rules,
                                           assets={"physical_lane_centers":self.physical_lane_centers,
                                                   "bev_drivable":self.bev_drivable},diagnostics=diagnostics)
        return self._cached_snapshot

    def step(self,control):
        control=np.asarray(control,float)
        if control.shape!=(2,) or not np.isfinite(control).all():raise ValueError("控制必须是有限的[a,delta]")
        if not self.acc_range[0]-1e-8<=control[0]<=self.acc_range[1]+1e-8 or abs(control[1])>self.steer_max+1e-8:
            raise ValueError("物理控制超出配置边界")
        action=np.array([2*(control[0]-self.acc_range[0])/(self.acc_range[1]-self.acc_range[0])-1,-control[1]/self.steer_max])
        self.env.step(action)
        self._previous_control=control.copy();self._step+=1;self._cached_snapshot=None
        if self.scenario=="intersection" and self._step*self.dt+1e-8>=self._next_spawn:
            self.env._spawn_vehicle(spawn_probability=self.spawn_probability)
            self._next_spawn+=1.
        ego=self._actor(self.env.vehicle)
        progress,lateral,_=project_polyline(ego.position,self.route,self._previous_route_s,ego.heading)
        collision=bool(self.env.vehicle.crashed)
        offroad=not self._audit_drivable.covers(actor_shape(ego))
        arrival=bool(progress>=self._goal_s and abs(lateral)<2.5 and not offroad and not collision)
        timeout=self._step*self.dt>=self.duration-1e-8
        return {"terminated":collision or arrival or timeout,"collision":collision,"arrival":arrival,"offroad":offroad,
                "timeout":timeout,"time":self._step*self.dt,"progress":progress+self._route_progress_offset,"reason":"collision" if collision else "arrival" if arrival else "timeout" if timeout else "running"}

    def set_planned_trajectory(self,states):
        """保存世界坐标规划状态的副本；空或非法输入清除旧计划，不改变物理状态。"""
        self.env._planned_trajectory=None
        try:
            values=np.asarray(states,dtype=float)
        except (TypeError,ValueError,OverflowError):
            return False
        if values.ndim!=2 or values.shape[0]<2 or values.shape[1] not in (2,4) or not np.isfinite(values).all():
            return False
        self.env._planned_trajectory=values[:,:2].copy()
        return True

    def render(self):
        image=self.env.render()
        # human 模式原生 pygame 显示后返回 None，录像仍取同一显示缓冲。
        if image is None and self.env.viewer is not None:
            image=self.env.viewer.get_image()
        return image

    def start_recording(self,path):
        """使用原生渲染缓冲，接收每个物理子步而非每个规划步的画面。"""
        from .video import NativeVideoRecorder
        if self._video_recorder is not None:raise RuntimeError('本轮录像已启动')
        self._video_recorder=NativeVideoRecorder(self,path,self.physics_hz)
        self.env.set_record_video_wrapper(self._video_recorder)
        # 原生默认将录像按2倍仿真速度播放；研究录像保持1倍时间。
        self.env.update_metadata(video_real_time_ratio=1)
        self._video_recorder.frames_per_sec=self.physics_hz
        self._video_recorder._capture_frame()
        return self._video_recorder

    def close(self):
        try:
            if self._video_recorder is not None:
                recorder = self._video_recorder
                self._video_recorder = None
                recorder.close()
        finally:
            try:
                if self.env is not None:
                    environment = self.env
                    self.env = None
                    environment.close()
            finally:
                if self._traffic_class_name:
                    globals().pop(self._traffic_class_name, None)
                    self._traffic_class_name = None
