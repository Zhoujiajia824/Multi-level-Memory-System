"""八向角色的公开车道拓扑；不读取对象目标车道、隐藏路线或未来。

地图面可能在路口重叠，不能把“车体碰到地图面”直接当成所属车道。
先确定连续车道链，再用当前中心线和车体跨界证据消歧；原地图不缩放。
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

ASSIGNMENT_PROTOCOL = "lane-topology-role-assignment-1.1"
# 开发默认值；仅用于地图连接/数值消歧，不是交通安全距离。
PARAMETERS = {"connection_gap_m": 2., "connection_angle_rad": math.pi / 3,
              "geometry_tolerance_m": 1e-5,
              "owner_heading_weight_m_per_rad": 1., "owner_motion_min_speed_mps": .5,
              "branch_sample_step_m": 2., "max_chain_segments": 128}


def _wrap(x):
    return (float(x) + math.pi) % (2 * math.pi) - math.pi


def project(points, point):
    """返回弧长、脚点、切向角和欧氏距离；不把端点外的距离丢成横向零。"""
    points = _clean(points)
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if len(lengths) == 0 or np.any(lengths <= 1e-8):
        raise ValueError("车道中心线包含零长段或不足两点")
    u = np.clip(np.sum((np.asarray(point)-points[:-1])*delta, axis=1)/lengths**2, 0., 1.)
    feet = points[:-1] + u[:, None]*delta
    distances = np.linalg.norm(feet-np.asarray(point), axis=1)
    j = int(np.argmin(distances))
    arc = np.r_[0., np.cumsum(lengths)]
    return float(arc[j]+u[j]*lengths[j]), feet[j], math.atan2(*delta[j][::-1]), float(distances[j])


def _clean(points):
    points = np.asarray(points, float)
    return points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]


def _join(first, second):
    """公开几何的短端部重叠只保留一次，不把中心线倒走计入弧长。"""
    first, second = _clean(first), _clean(second)
    s, foot, _, distance = project(second, first[-1])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(second, axis=0), axis=1))]
    if distance <= PARAMETERS['connection_gap_m'] and 1e-8 < s < arc[-1]-1e-8:
        second = np.vstack((foot, second[arc > s+1e-8]))
    return _clean(np.vstack((first, second)))


def _connection(source, target):
    """前后继必须端部相接且切向相容；不接受反向行驶的另一条车道。"""
    source, target = _clean(source), _clean(target)
    s, _, target_h, distance = project(target, source[-1])
    target_length = np.linalg.norm(np.diff(target, axis=0), axis=1).sum()
    # 仅允许后继开头重叠；不能把另一车道中部任意接到当前末端。
    if s > min(10., target_length*.5)+1e-6:
        return None
    source_h = math.atan2(*(source[-1]-source[-2])[::-1])
    angle = abs(_wrap(target_h-source_h))
    if distance > PARAMETERS['connection_gap_m'] or angle > PARAMETERS['connection_angle_rad']:
        return None
    return distance, angle


def trace_lane(scene, origin_id, *, back_m=30., front_m=70., center_role=True):
    """为一个角色延伸唯一公开链；导航lane序列优先，完整导航几何消歧。"""
    lanes = {lane.id: lane for lane in scene.lanes}
    if origin_id not in lanes:
        return None
    origin = lanes[origin_id]
    points = _clean(origin.center)
    if len(points) < 2:
        return None
    used, decisions = [origin.id], []
    route = _clean(scene.route)
    route_line = LineString(route)
    _, source_foot, _, _ = project(points, scene.ego.position)
    _, route_foot, route_h, _ = project(route, source_foot)
    expected_d = float(np.dot(source_foot-route_foot, [-math.sin(route_h), math.cos(route_h)]))
    route_order = {key: i for i, key in enumerate(scene.route_lane_ids)}

    def choose(source, candidates, forward):
        ranked = []
        for lane in candidates:
            geometry = _clean(lane.center)
            connection = _connection(source.center, geometry) if forward else _connection(geometry, source.center)
            if connection is None:
                continue
            line = LineString(geometry)
            distances = np.linspace(0., line.length, max(3, int(line.length/PARAMETERS['branch_sample_step_m'])+1))
            # 整段而非第3个顶点：在共同起段之后仍能辨别直行/转向分支。
            cost = []
            for s in distances:
                p = np.asarray(line.interpolate(float(s)).coords[0])
                _, foot, h, distance = project(route, p)
                side = float(np.dot(p-foot, [-math.sin(h), math.cos(h)]))
                _, _, lane_h, _ = project(geometry, p)
                cost.append(abs(side-expected_d)+distance-abs(side)+2*abs(_wrap(lane_h-h)))
            explicit = 1
            if center_role and lane.id in route_order and source.id in route_order:
                if (route_order[lane.id]-route_order[source.id]) * (1 if forward else -1) > 0:
                    explicit = 0
            ranked.append((explicit, float(np.mean(cost))+connection[0]+connection[1], lane.id, lane))
        if not ranked:
            return None
        ranked.sort(key=lambda value: value[:3])
        best = ranked[0]
        decisions.append({"from": source.id, "to": best[2], "direction": "forward" if forward else "backward",
                          "source": "navigation_lane_ids" if best[0] == 0 else "public_topology_and_route_geometry",
                          "alternatives": len(ranked)})
        return best[3]

    before = origin
    for _ in range(PARAMETERS['max_chain_segments']):
        if project(points, scene.ego.position)[0] >= back_m:
            break
        candidates = [lane for lane in lanes.values() if before.id in lane.successors and lane.id not in used]
        previous = choose(before, candidates, False)
        if previous is None:
            break
        used.insert(0, previous.id)
        points = _join(previous.center, points)
        before = previous
    after = origin
    for _ in range(PARAMETERS['max_chain_segments']):
        s = project(points, scene.ego.position)[0]
        if np.linalg.norm(np.diff(points, axis=0), axis=1).sum()-s >= front_m:
            break
        candidates = [lanes[key] for key in after.successors if key in lanes and key not in used]
        following = choose(after, candidates, True)
        if following is None:
            break
        used.append(following.id)
        points = _join(points, following.center)
        after = following
    shapes = [Polygon(lanes[key].polygon) for key in used if len(lanes[key].polygon) >= 3]
    return {"points": points, "lane_ids": used, "shape": unary_union(shapes) if shapes else None,
            "ego_s": project(points, scene.ego.position)[0], "decisions": decisions}


def lane_owner(actor, lane_cache):
    """重叠地图面按中心线距离和当前运动方向消歧，不单独过滤逆行对象。

    只有地图面有多个归属时方向才参与排序；唯一物理车道中的逆行车保留。
    角度项量纲用米/弧度配置，不能把地图宽大交叉面当成几何精确车道。
    """
    point = Point(actor.position)
    choices = []
    for key, (lane, polygon, line) in lane_cache.items():
        if not polygon.covers(point):
            continue
        _, _, heading, distance = project(lane.center, actor.position)
        angle = abs(_wrap(actor.heading-heading))
        velocity = np.asarray(actor.velocity, float)
        if np.all(np.isfinite(velocity)) and np.linalg.norm(velocity) >= PARAMETERS['owner_motion_min_speed_mps']:
            motion_heading = math.atan2(*velocity[::-1])
            angle = .5*(angle+abs(_wrap(motion_heading-heading)))
        choices.append((distance, angle, key))
    if not choices:
        return None
    if len(choices) == 1:
        return choices[0][2]
    def score(row):
        return row[0]+PARAMETERS['owner_heading_weight_m_per_rad']*row[1]
    # 源lane_id仅用于同分，不得压过中心线/运动证据。
    return min(choices, key=lambda row: (score(row), row[2] != actor.lane_id, row[0], row[2]))[2]


def forward_reachable(lanes, start, target):
    """正常前进可跨多个短连接段；仅沿公开且几何连续的有向后继搜索。"""
    visited, pending = set(), [start]
    for _ in range(PARAMETERS['max_chain_segments']):
        if not pending:
            break
        key = pending.pop(0)
        if key in visited or key not in lanes:
            continue
        visited.add(key)
        if key == target:
            return True
        source = lanes[key]
        pending.extend(candidate for candidate in sorted(source.successors)
                       if candidate in lanes and candidate not in visited
                       and _connection(source.center, lanes[candidate].center) is not None)
    return False


def belongs_to_trace(actor, body, owner_id, trace, lane_cache, area_tolerance):
    """车身实际侵入准入车道的物理宽度带即占用，不以来源车道方向否决。

    owner_id只说明主车道归属；横穿、对向和转弯来车仍可能占用当前行驶链。
    地图多边形须与真实中心线半宽带相交，避免宽路口包围面造成假占用。
    范围、邻道权限和每区代表选择由调用方处理，本函数不扩大准入车道链。
    """
    if trace is None or trace['shape'] is None:
        return False
    for key in trace['lane_ids']:
        if key not in lane_cache:
            continue
        lane, polygon, line = lane_cache[key]
        width = lane.width
        if (isinstance(width, (bool, np.bool_))
                or not isinstance(width, (int, float, np.integer, np.floating))
                or not math.isfinite(float(width)) or width <= 0):
            continue
        intersection = body.intersection(polygon)
        if intersection.area < area_tolerance:
            continue
        if intersection.intersection(line.buffer(float(width)/2, cap_style=2)).area >= area_tolerance:
            return True
    # 仅消费trace已有的验证连接；不在此生成或猜测新的道路连接。
    return any(body.intersection(link['shape']).area >= area_tolerance
               for link in trace.get('navigation_connections', []))
