"""导航之外的局部几何参考；候选筛选不替代 MPC 的物理验收。"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter
import numpy as np
from shapely import covers, intersects, polygons, unary_union
from shapely.geometry import LineString, Point, Polygon

from .geometry import (actor_shape, clean_polyline, polyline_lengths,
                       project_polyline, sample_polyline, wrap_angle)
from .quintic_reference import quintic_offsets, initial_lateral_state, spatial_samples


DEFAULTS = {
    "lookahead_m": 140., "sample_m": .5, "blend_min_m": 30.,
    "blend_time_s": 3., "minimum_gain_mps": 2.,
    "front_gap_m": 8., "front_headway_s": 1.2,
    "rear_gap_m": 8., "rear_closing_time_s": 1.5,
    "prediction_check_s": 3., "prediction_min_gap_m": 4.,
    "complete_lateral_m": .35, "complete_heading_rad": .1,
    "complete_lateral_speed_mps": .25, "complete_hold_s": .3,
    "dwell_s": 3., "abort_lateral_m": .25,
    "candidate_period_s": .5,
    "geometry_margin_m": .001,
    "allow_right_overtake": False,
}


@dataclass
class LocalReferenceResult:
    world_path: np.ndarray
    revision: int
    diagnostics: dict


class LocalReferenceManager:
    """一个几何候选进入同一个 MPC；车道变换过程跨模型帧保持。"""

    def __init__(self, config):
        self.p = {**DEFAULTS, **config.get("local_reference", {})}
        self.state = {"phase": "keep", "revision": 0, "source_lane_id": None,
                      "target_lane_id": None, "path": None, "selected_source": "navigation",
                      "started_at": None, "complete_since": None,
                      "last_completed_at": None, "last_cancelled_at": None,
                      "last_candidate_time": None,
                      "candidate_diagnostics": [], "keep_path": None,
                      "keep_lane_id": None}

    def export_state(self):
        """供原帧重放与求解失败回滚使用；不包含平台对象。"""
        return deepcopy(self.state)

    def import_state(self, state):
        self.state = deepcopy(state)

    def reset_intent(self):
        """语义目标切换/释放时移除旧参考、车道锁、滞回及初值偏好。"""
        revision = int(self.state["revision"])+1
        self.__init__({"local_reference": self.p})
        self.state["revision"] = revision

    @staticmethod
    def _segment(lane_id):
        parts = str(lane_id).rsplit("|", 1)
        return parts[0] if len(parts) == 2 else str(lane_id)

    def _lane(self, scene, lanes):
        # 已完成换道时优先使用实际几何所在车道，不等待平台稳定标签更新。
        def distance(lane):
            if lane is None:return None
            s, d, heading = project_polyline(scene.ego.position, lane.center,
                                             heading=scene.ego.heading)
            if abs(float(wrap_angle(heading-scene.ego.heading))) > .7:
                return None
            if Polygon(lane.polygon).buffer(.01).covers(Point(scene.ego.position)):
                return abs(d)
            return None
        # 与原优先级等价的快速路径：有效的公开关联已能决定身份时，
        # 不再为每0.1秒控制扫描全图所有重叠车道。
        known=(scene.ego.lane_id,scene.diagnostics.get('stable_navigation_lane_id'))
        for key in known:
            lane=lanes.get(key)
            if distance(lane) is not None:
                # 路口多条连接车道覆盖同一点时，平台最近lane标签可能落到
                # 直行分支。只有非平行邻道重叠才按公开已知导航消歧，不能把
                # 正常完成的平行换道拉回旧车道。
                if key not in scene.route_lane_ids:
                    parallel=any(candidate.id in scene.route_lane_ids and
                        (candidate.id in (lane.left,lane.right) or lane.id in (candidate.left,candidate.right))
                        for candidate in lanes.values())
                    if not parallel:
                        route_choices=[(distance(candidate),candidate) for candidate in lanes.values()
                                       if candidate.id in scene.route_lane_ids and distance(candidate) is not None]
                        if route_choices:return min(route_choices,key=lambda item:(item[0],item[1].id))[1]
                return lane
        candidates=[]
        for lane in lanes.values():
            if lane.id in known:continue
            value=distance(lane)
            if value is not None:candidates.append((value,lane))
        if candidates:
            # 路口的不同分支可能几何重叠；公开的当前车道关联优先于横距微差。
            # 已完成换道的车体不再被旧车道覆盖时，自然使用实际目标车道。
            on_route=[item for item in candidates if item[1].id in scene.route_lane_ids]
            if on_route:return min(on_route,key=lambda item:item[0])[1]
            return min(candidates, key=lambda item: item[0])[1]
        return lanes.get(scene.ego.lane_id)

    def _continuation(self, scene, lane, lanes):
        """只沿导航声明的有序道路分支，不能从 successors 任挑一个出口。"""
        anchor=scene.diagnostics.get('reference_anchor_s',scene.diagnostics.get('route_s'))
        nav_s=project_polyline(scene.ego.position,scene.route,anchor,scene.ego.heading)[0]
        nav_point=sample_polyline(scene.route,nav_s)
        nav_heading=project_polyline(nav_point,scene.route,nav_s,scene.ego.heading)[2]
        lane_heading=project_polyline(scene.ego.position,lane.center,heading=scene.ego.heading)[2]
        if (lane.id in scene.route_lane_ids and
                Polygon(lane.polygon).buffer(.01).covers(Point(nav_point)) and
                abs(float(wrap_angle(nav_heading-lane_heading)))<.7):
            # 平台导航已处理路口/环岛入口出口的几何连接。重新拼原始车道中心
            # 会丢掉连接曲线，甚至因端点间隙提前截断，使车停在入口或出口前。
            # 邻道不在导航序列时仍使用下方拓扑核验，不能强行拉回旧车道。
            arc=polyline_lengths(scene.route)
            start=max(0.,nav_s-20.);end=min(float(arc[-1]),nav_s+self.p['lookahead_m']+20.)
            return clean_polyline(np.vstack((sample_polyline(scene.route,start),
                scene.route[(arc>start)&(arc<end)],sample_polyline(scene.route,end))))
        path = np.asarray(lane.center, float).copy()
        ordered = scene.route_lane_ids
        own_segment = self._segment(lane.id)
        matches=[i for i,key in enumerate(ordered) if self._segment(key)==own_segment]
        index=(matches[-1] if scene.rules.get('goal_remaining_m',float('inf'))<90. else matches[0]) if matches else None
        if index is None:
            return path
        current = lane
        for key in ordered[index+1:]:
            intended = lanes.get(key)
            if intended is None:
                break
            segment = self._segment(key)
            allowed = [lanes[k] for k in current.successors if k in lanes
                       and self._segment(k) == segment]
            if not allowed:
                break
            end_heading = project_polyline(current.center[-1], current.center)[2]
            def cost(candidate):
                start_heading = project_polyline(candidate.center[0], candidate.center)[2]
                return (np.linalg.norm(current.center[-1]-candidate.center[0])
                        + 5*abs(float(wrap_angle(start_heading-end_heading))))
            chosen = min(allowed, key=cost)
            gap = float(np.linalg.norm(current.center[-1]-chosen.center[0]))
            if gap > max(3., .8*max(current.width, chosen.width)):
                break
            path = np.vstack((path, chosen.center))
            current = chosen
        return clean_polyline(path)

    def _navigation_compatible(self, scene, source, target, lanes):
        if self._segment(source.id) != self._segment(target.id):
            return False
        ordered = scene.route_lane_ids
        index = next((i for i, key in enumerate(ordered)
                      if self._segment(key) == self._segment(source.id)), None)
        if index is None:
            return False
        if index == len(ordered)-1:
            return True
        required = self._segment(ordered[index+1])
        choices = [lanes[key] for key in target.successors
                   if key in lanes and self._segment(key) == required]
        if not choices:
            return False
        # 不跨接横向断开的车道，也不允许转到另一个导航道路分支。
        return any(np.linalg.norm(target.center[-1]-lane.center[0])
                   <= max(3., .8*max(target.width, lane.width)) for lane in choices)

    @staticmethod
    def _shared_crossable(source, target, position):
        """按实际共边几何找权限，不依赖边界数组顺序。"""
        source_s = project_polyline(position, source.center)[0]
        a = sample_polyline(source.center, source_s)
        b = sample_polyline(target.center, project_polyline(a, target.center)[0])
        middle = Point((a+b)/2)
        for lane in (source, target):
            nearby = [edge for edge in lane.boundaries
                      if len(edge.get("points", [])) >= 2
                      and LineString(edge["points"]).distance(middle) <= .3]
            if not nearby or not all(edge.get("crossable", False) for edge in nearby):
                return False
        return True

    def _legal(self, scene, source, target, lanes):
        destination_lane = self._destination_route_lane(scene, source)
        if destination_lane == source.id and target.id != source.id:
            return '终点接近阶段保留任务目标车道'
        if target.id not in (source.left, source.right):
            return "不是同向相邻车道"
        if not self._shared_crossable(source, target, scene.ego.position):
            return "共享边界不可跨越或权限缺失"
        if not self._navigation_compatible(scene, source, target, lanes):
            return "目标车道与有序导航分支不兼容"
        speed = np.linalg.norm(scene.ego.velocity)
        distance = polyline_lengths(target.center)[-1]-project_polyline(scene.ego.position, target.center)[0]
        if distance < max(self.p["blend_min_m"], speed*self.p["blend_time_s"])+8.:
            return "目标车道剩余长度不足"
        return None

    def _navigation_change_needed(self, scene, source, target, lanes):
        """当前车道无法继续既定导航时，优先选择可接入的合法邻道。"""
        destination_lane = self._destination_route_lane(scene, source)
        if destination_lane is not None:
            return source.id != destination_lane and target.id == destination_lane
        return (not self._navigation_compatible(scene,source,source,lanes)
                and self._navigation_compatible(scene,source,target,lanes))

    def _destination_route_lane(self, scene, source):
        """终点前提前回到原任务路线的车道；仍必须通过合法性/车隙验收。"""
        remaining = scene.rules.get('goal_remaining_m')
        if remaining is None or remaining > max(140.,8.*np.linalg.norm(scene.ego.velocity)+60.):
            return None
        matches=[key for key in scene.route_lane_ids if self._segment(key)==self._segment(source.id)]
        # 回环路线起终点可同 road/section，终点接近阶段取最终声明车道。
        return matches[-1] if matches else None

    def _right_overtake_reason(self, scene, source, target, lanes, traffic=None):
        """限制绕过慢前车的右侧超车，不把普通右变道或归右一并禁止。"""
        if (target.id != source.right or self.p['allow_right_overtake']
                or self._navigation_change_needed(scene, source, target, lanes)):
            return None
        traffic = traffic if traffic is not None else self._traffic(scene, None, source, check_future=False)
        front = traffic['front']
        actor = next((obj for obj in scene.actors if front and obj.id == front['id']), None)
        desired = float(scene.rules.get('desired_speed', np.linalg.norm(scene.ego.velocity)))
        if (actor is not None and actor.kind in ('vehicle', 'bicycle')
                and front['speed_mps'] + self.p['minimum_gain_mps'] < desired
                and traffic['estimated_speed_mps'] < desired-1e-9):
            return '禁止为绕过本车道慢前车主动从右侧超车；此右换道非导航必需'
        return None

    def _traffic(self, scene, predictions, lane, check_future=True):
        """可见当前净距及 EKF 未来占道检查，仅决定候选是否值得求解。"""
        ego_s, _, heading = project_polyline(scene.ego.position, lane.center)
        tangent = np.array([np.cos(heading), np.sin(heading)])
        ego_speed = max(0., float(scene.ego.velocity @ tangent))
        desired = float(scene.rules.get("desired_speed", ego_speed))
        front = None; rear = None; rejection = None; predicted_gap = float("inf")
        lane_polygon = Polygon(lane.polygon)
        for actor in scene.actors:
            if not actor.visible:
                continue
            if not actor_shape(actor).intersects(lane_polygon):
                continue
            other_s = project_polyline(actor.position, lane.center, previous_s=ego_s,
                                      window=self.p["lookahead_m"]+self.p["rear_gap_m"])[0]
            projected_half = (abs(np.cos(actor.heading-heading))*actor.length
                              + abs(np.sin(actor.heading-heading))*actor.width)/2
            ego_half = (abs(np.cos(scene.ego.heading-heading))*scene.ego.length
                        + abs(np.sin(scene.ego.heading-heading))*scene.ego.width)/2
            gap = abs(other_s-ego_s)-projected_half-ego_half
            other_speed = max(0., float(actor.velocity @ tangent))
            item = {"id": actor.id, "gap_m": float(gap), "speed_mps": other_speed}
            if other_s >= ego_s:
                if front is None or gap < front["gap_m"]:
                    front = item
            else:
                item["closing_ttc_s"] = (gap/(other_speed-ego_speed)
                                         if other_speed > ego_speed+.01 else None)
                if rear is None or gap < rear["gap_m"]:
                    rear = item
        if front and front["gap_m"] < self.p["front_gap_m"]+self.p["front_headway_s"]*ego_speed:
            rejection = "目标车道前方净距不足"
        if rear and rear["gap_m"] < self.p["rear_gap_m"]+self.p["rear_closing_time_s"]*max(0., rear["speed_mps"]-ego_speed):
            rejection = "目标车道后方净距或追近时间不足"
        for actor in scene.actors:
            if not check_future or rejection is not None:
                break
            if not actor.visible or actor.id not in predictions.trajectories:
                continue
            trajectory = predictions.trajectories[actor.id]
            times = np.asarray(predictions.times[:len(trajectory)])
            mask = (times>.2) & (times<=self.p["prediction_check_s"])
            rows = np.asarray(trajectory)[mask]
            if not len(rows):
                continue
            other_s, lateral, tangent_heading = project_polyline(
                rows[:, :2], lane.center, previous_s=ego_s,
                window=self.p["lookahead_m"]+self.p["rear_gap_m"])
            angle = rows[:, 2]-tangent_heading
            if actor.kind == "pedestrian":
                half_long = np.full(len(rows), actor.radius or actor.width/2)
                half_lateral = half_long
            else:
                half_long = (abs(np.cos(angle))*actor.length+abs(np.sin(angle))*actor.width)/2
                half_lateral = (abs(np.cos(angle))*actor.width+abs(np.sin(angle))*actor.length)/2
            # 批量投影的包围投影只做保守筛选；真正车体交叠仍由同一MPC逐时验收。
            occupied = abs(lateral)<=lane.width/2+half_lateral
            signed_gap = other_s-(ego_s+ego_speed*times[mask])
            gaps = abs(signed_gap)-half_long-scene.ego.length/2
            if not occupied.any():
                continue
            predicted_gap = min(predicted_gap, float(np.min(gaps[occupied])))
            predicted_speed = np.maximum(0., rows[:, 3]*np.cos(angle))
            required = np.where(signed_gap>=0,
                                self.p["front_gap_m"]+self.p["front_headway_s"]*ego_speed,
                                self.p["rear_gap_m"]+self.p["rear_closing_time_s"]*np.maximum(0., predicted_speed-ego_speed))
            required = np.maximum(required, self.p["prediction_min_gap_m"])
            if np.any(occupied & (gaps<required)):
                rejection = rejection or "EKF预测目标车道间隙不足"
        if front:
            free_speed = min(desired, front["speed_mps"]+max(0., front["gap_m"]-self.p["front_gap_m"])/4.)
        else:
            free_speed = desired
        return {"front": front, "rear": rear, "estimated_speed_mps": float(free_speed),
                "predicted_min_gap_m": None if not np.isfinite(predicted_gap) else predicted_gap,
                "rejection": rejection}

    def _path(self, scene, source, target, lanes):
        """五次分段横向插值，终端沿目标中心线延伸，不指定到达时刻。"""
        reference = self._continuation(scene, source, lanes)
        if target is source:
            # 普通循道沿已连接的导航中心线。按瞬时转角再生成长距离五次
            # “归中”会把入弯时的曲率差放大成外鼓路径，甚至指向出口实线。
            # 实际初始偏差由 MPC 的恢复包络处理，jerk/转角速率保持控制连续。
            # 主动换道以及模型局部引导点仍使用各自的五次参考。
            return reference
        ego_s, d0, base_heading = project_polyline(scene.ego.position, reference,
                                                  heading=scene.ego.heading)
        total = polyline_lengths(reference)[-1]
        speed = float(np.linalg.norm(scene.ego.velocity))
        blend = max(self.p["blend_min_m"], speed*self.p["blend_time_s"])
        end = min(total, ego_s+self.p["lookahead_m"])
        ss = spatial_samples(max(0.,ego_s-15.),end,self.p['sample_m'],
                             anchors=(ego_s,ego_s+blend))
        base = sample_polyline(reference, ss)
        headings = project_polyline(base, reference, previous_s=ego_s,
                                     heading=base_heading,
                                     window=self.p["lookahead_m"]+20.)[2]
        normals = np.c_[-np.sin(headings), np.cos(headings)]
        if target is source:
            target_d = np.zeros(len(base))
        else:
            target_reference = self._continuation(scene, target, lanes)
            target_ego_s = project_polyline(scene.ego.position, target_reference)[0]
            target_s = project_polyline(base, target_reference, previous_s=target_ego_s,
                                       window=self.p["lookahead_m"]+20.)[0]
            target_points = sample_polyline(target_reference, target_s)
            target_d = np.sum((target_points-base)*normals, axis=1)
        relative = ss-ego_s
        terminal = blend
        terminal_d = float(np.interp(terminal, relative, target_d))
        start=initial_lateral_state(scene,reference,ego_s,d0,base_heading)
        first=np.gradient(target_d,ss);second=np.gradient(first,ss)
        finish=(terminal_d,float(np.interp(ego_s+terminal,ss,first)),
                float(np.interp(ego_s+terminal,ss,second)))
        offsets=quintic_offsets(relative,terminal,start,finish)
        # 身后点只支持参考投影，不能影响未来的边界状态与车体检查。
        before=relative<0
        offsets[before]=start[0]+relative[before]*start[1]+.5*relative[before]**2*start[2]
        # 车道宽度或弯道半径变化时，终端继续精确目标车道，而非固定横向偏移。
        after = relative >= terminal
        offsets[after] = target_d[after]
        return clean_polyline(base+normals*offsets[:, None])

    def _road_ok(self, scene, path):
        if len(path)<2:
            return False
        headings = np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0]))
        # 批量使用完整矩形；只扫描局部包围盒内的实线，避免身体×全地图循环。
        indices = np.arange(0, len(path), max(1, int(2./self.p["sample_m"])))
        positions = path[indices]; angles = headings[indices]
        c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
        local = np.array([[scene.ego.length/2, scene.ego.width/2],
                          [scene.ego.length/2, -scene.ego.width/2],
                          [-scene.ego.length/2, -scene.ego.width/2],
                          [-scene.ego.length/2, scene.ego.width/2]])
        corners = np.stack((positions[:, 0, None]+c*local[:, 0]-s*local[:, 1],
                            positions[:, 1, None]+s*local[:, 0]+c*local[:, 1]), axis=-1)
        bodies = polygons(corners)
        if not np.all(covers(scene.drivable.buffer(self.p["geometry_margin_m"]), bodies)):
            return False
        lo = np.min(corners, axis=(0, 1)); hi = np.max(corners, axis=(0, 1))
        lines = []
        for lane in scene.lanes:
            for boundary in lane.boundaries:
                if (boundary.get("crossable", False)
                        or boundary.get("kind") in ("virtual_connection", "virtual_lane_boundary")):
                    continue
                points = np.asarray(boundary.get("points", []), float)
                if len(points)<2 or np.any(np.max(points, axis=0)<lo) or np.any(np.min(points, axis=0)>hi):
                    continue
                lines.append(LineString(points))
        if lines and np.any(intersects(unary_union(lines), bodies)):
            return False
        return True

    def select(self, scene, predictions):
        started = perf_counter()
        lanes = {lane.id: lane for lane in scene.lanes}
        current = self._lane(scene, lanes) if lanes else None
        candidates = []
        event = None
        if current is None:
            path = clean_polyline(scene.route).copy()
            return LocalReferenceResult(path, int(self.state["revision"]), {
                "phase": "keep", "maneuver_active": False, "source_lane_id": None,
                "target_lane_id": None, "selected_source": "navigation",
                "candidates": [],
                "selection_wall_s": perf_counter()-started})
        phase = self.state["phase"]
        if phase in ("initiate", "hold"):
            target = lanes.get(self.state["target_lane_id"])
            source = lanes.get(self.state["source_lane_id"])
            if target is not None and source is not None:
                _, d, tangent = project_polyline(scene.ego.position, target.center)
                lateral_speed = abs(float(scene.ego.velocity @ np.array([-np.sin(tangent), np.cos(tangent)])))
                complete = (abs(d)<=self.p["complete_lateral_m"]
                            and abs(float(wrap_angle(scene.ego.heading-tangent)))<=self.p["complete_heading_rad"]
                            and lateral_speed<=self.p["complete_lateral_speed_mps"])
                if complete:
                    if self.state["complete_since"] is None:
                        self.state["complete_since"] = float(scene.time)
                    if scene.time-self.state["complete_since"]>=self.p["complete_hold_s"]-1e-8:
                        self.state["phase"] = "complete"
                        self.state["last_completed_at"] = float(scene.time)
                        event = "complete"
                else:
                    self.state["complete_since"] = None
                _, source_d, _ = project_polyline(scene.ego.position, source.center)
                traffic = self._traffic(scene, predictions, target)
                # 仅在尚未离开原车道中心附近时取消；已跨出的车体不能突然被拉回。
                if (not complete and abs(source_d)<=self.p["abort_lateral_m"]
                        and traffic["rejection"] is not None):
                    self.state["phase"] = "cancel"
                    self.state["target_lane_id"] = source.id
                    self.state["path"] = None
                    self.state["last_cancelled_at"] = float(scene.time)
                    self.state["revision"] = int(self.state["revision"])+1
                    event = "cancel"
                    candidates.append({"lane_id": target.id, **traffic})
                else:
                    if self.state["phase"] != "complete":
                        self.state["phase"] = "hold"
                    path = (np.asarray(self.state["path"], float)
                            if self.state["path"] is not None
                            else self._path(scene, source, target, lanes))
                    self.state["path"] = path.tolist()
                    if self.state["phase"] == "complete":
                        path = self._path(scene, target, target, lanes)
                        self.state["path"] = path.tolist()
                    return self._result(path, candidates, started, event)
            else:
                self.state["phase"] = "cancel"
                self.state["path"] = None
                self.state["last_cancelled_at"] = float(scene.time)
                self.state["revision"] = int(self.state["revision"])+1
                event = "cancel_map_changed"
        # 基础换道完成后采用实际目标车道，不固定返回原车道。
        cached_keep = self.state.get("keep_path")
        completed_target = (self.state["phase"] == "complete"
                            and self.state["target_lane_id"] == current.id)
        same_navigation_road=any(self._segment(current.id)==self._segment(key) for key in scene.route_lane_ids)
        if not same_navigation_road and not completed_target:
            # 路口车道关联可能暂指向重叠的另一分支。无主动换道承诺时，
            # 始终保留已知导航连接，不能把该关联自动变成局部转向意图。
            path = np.asarray(scene.route, float).copy()
        elif cached_keep is not None and self.state.get("keep_lane_id") == current.id:
            path = np.asarray(cached_keep, float)
            along = project_polyline(scene.ego.position, path, heading=scene.ego.heading)[0]
            if polyline_lengths(path)[-1]-along<60.:
                path = self._path(scene, current, current, lanes)
        else:
            path = self._path(scene, current, current, lanes)
        self.state["keep_path"] = path.tolist()
        self.state["keep_lane_id"] = current.id
        selected = current
        chosen_source = "navigation"
        can_change = (self.state["last_completed_at"] is None
                      or scene.time-self.state["last_completed_at"]>=self.p["dwell_s"])
        can_change = can_change and (self.state.get("last_cancelled_at") is None
                                    or scene.time-self.state["last_cancelled_at"]>=self.p["dwell_s"])
        destination_lane=self._destination_route_lane(scene,current)
        if destination_lane is not None and destination_lane != current.id:
            # 必需归道不受“无收益换道后的驻留”阻挡；车隙/标线/完整车体复核仍照常。
            can_change=True
        candidate_due = (self.state.get("last_candidate_time") is None
                         or scene.time-self.state["last_candidate_time"]>=self.p["candidate_period_s"]-1e-8)
        if can_change and candidate_due:
            self.state["last_candidate_time"] = float(scene.time)
            own_traffic = self._traffic(scene, predictions, current, check_future=False)
            candidates.append({"lane_id": current.id, "source": "keep", **own_traffic})
            best_gain = self.p["minimum_gain_mps"]
            selected_priority = 0
            for key in (current.left, current.right):
                if key not in lanes:
                    continue
                lane = lanes[key]
                reason = self._legal(scene, current, lane, lanes)
                navigation_required=self._navigation_change_needed(scene,current,lane,lanes)
                reason = reason or self._right_overtake_reason(scene, current, lane, lanes, own_traffic)
                traffic = (self._traffic(scene, predictions, lane) if reason is None else
                           {"front": None, "rear": None, "estimated_speed_mps": 0.,
                            "predicted_min_gap_m": None, "rejection": reason})
                gain = traffic["estimated_speed_mps"]-own_traffic["estimated_speed_mps"]
                # 无收益的返回不应被“超车收益”门槛卡住。安全等效时允许归右，
                # 无须先有左变道历史；但不能覆盖导航必需或更好的合法超车候选。
                keep_right = key == current.right and gain >= -1e-9
                priority = (3 if navigation_required else 2 if gain > self.p['minimum_gain_mps']+1e-9
                            else 1 if keep_right else 0)
                item = {"lane_id": key, "source": "base", **traffic, "gain_mps": float(gain),
                        "navigation_required":navigation_required, "keep_right": keep_right}
                item["rejection"] = reason or traffic["rejection"]
                proposed = None
                preferred = priority > selected_priority or (priority == selected_priority == 2 and gain > best_gain+1e-9)
                if item["rejection"] is None and preferred:
                    proposed = self._path(scene, current, lane, lanes)
                    if not self._road_ok(scene, proposed):
                        item["rejection"] = "完整车体粗检超出合法走廊或跨越实线"
                if item["rejection"] is None and preferred and proposed is not None:
                    selected = lane; path = proposed; best_gain = gain
                    selected_priority = priority
                    chosen_source = ("navigation_lane_change" if navigation_required else
                                     "base_keep_right" if priority == 1 else "base_lane_change")
                elif item["rejection"] is None and priority == 0:
                    item["rejection"] = "换道收益不足"
                candidates.append(item)
            self.state["candidate_diagnostics"] = deepcopy(candidates)
        elif can_change:
            candidates = deepcopy(self.state.get("candidate_diagnostics", []))
        if selected is not current:
            self.state.update(phase="initiate", source_lane_id=current.id,
                              target_lane_id=selected.id, path=path.tolist(),
                              selected_source=chosen_source, started_at=float(scene.time),
                              complete_since=None, revision=int(self.state["revision"])+1)
            event = "initiate"
        else:
            if self.state["phase"] not in ("complete", "cancel"):
                self.state["phase"] = "keep"
            self.state.update(source_lane_id=current.id, target_lane_id=current.id,
                              selected_source=chosen_source)
            self.state["path"] = None
        return self._result(path, candidates, started, event)

    def _result(self, path, candidates, started, event):
        return LocalReferenceResult(np.asarray(path, float), int(self.state["revision"]), {
            "phase": self.state["phase"],
            "maneuver_active": self.state["phase"] in ("initiate", "hold"),
            "source_lane_id": self.state["source_lane_id"],
            "target_lane_id": self.state["target_lane_id"],
            "selected_source": self.state["selected_source"],
            "reference_geometry": ("quintic_lane_change" if self.state["phase"] in ("initiate", "hold")
                                   else "connected_lane_center_with_mpc_offset_recovery"),
            "candidates": candidates,
            "candidate_evaluated_at": self.state.get("last_candidate_time"),
            "event": event, "selection_wall_s": perf_counter()-started})
