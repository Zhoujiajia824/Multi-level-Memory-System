"""长路线场景的公开地图选址；兼容 CARLA 0.9.15 / Python 3.7。

不生成演员、不推进时钟、不读取任何周车导航。锚点首项是起点，后续项是
自车必须经过的公开导航路点。离线 OpenDRIVE 没有 spawn_points 时使用明示的
20m 地图采样起点；审计记录两者区别，不把离线选址冒充真实物理运行。
"""
import math
import random


EXTENDED_ROUTE_VERSION = 'carla-scripted-route-1.0'
ROUTE_SELECTION_VERSION = EXTENDED_ROUTE_VERSION
ROUTE_DEFAULTS = {'kind': 'any', 'min_junctions': 0,
                  'min_same_direction_lanes': 1, 'search_limit': 5000}


def validate_route_requirements(settings):
    supplied = settings.get('route_requirements', {})
    if not isinstance(supplied, dict) or set(supplied)-set(ROUTE_DEFAULTS):
        raise ValueError('route_requirements含未实现字段')
    req = dict(ROUTE_DEFAULTS, **supplied)
    if req['kind'] not in ('any', 'junction', 'merge', 'diverge'):
        raise ValueError('路线kind只支持any/junction/merge/diverge')
    for key, low, high in (('min_junctions', 0, 30), ('min_same_direction_lanes', 1, 8),
                           ('search_limit', 100, 50000)):
        if type(req[key]) is not int or not low <= req[key] <= high:
            raise ValueError('路线参数{}须为[{},{}]内整数'.format(key, low, high))
    anchors = settings.get('route_anchors', [])
    if not isinstance(anchors, list) or len(anchors) > 20:
        raise ValueError('route_anchors须为最多20个地图锚点列表')
    for item in anchors:
        if (not isinstance(item, dict) or set(item) != {'road_id','lane_id','s_m'}
                or type(item['road_id']) is not int or type(item['lane_id']) is not int
                or item['lane_id'] == 0 or type(item['s_m']) not in (int, float)
                or not math.isfinite(item['s_m']) or item['s_m'] < 0):
            raise ValueError('导航锚点必须提供有效road_id/lane_id/s_m')
    if anchors and settings.get('spawn_index') is not None:
        raise ValueError('route_anchors起点与spawn_index不得同时指定')
    return req


def distance(a, b):
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)


def wrap(value):
    return (value+math.pi) % (2*math.pi)-math.pi


def waypoint_key(wp):
    return (int(wp.road_id), int(wp.section_id), int(wp.lane_id), round(float(wp.s), 2))


def road_key(wp):
    return (int(wp.road_id), int(wp.section_id), int(wp.lane_id))


def same_direction_lane_count(wp):
    """同层同向 Driving 车道数；与能否跨线分开，不把对向车道计入。"""
    seen = {road_key(wp)}
    yaw = wp.transform.rotation.yaw
    z = wp.transform.location.z
    for side in ('left', 'right'):
        other = wp
        for _ in range(12):
            other = other.get_left_lane() if side == 'left' else other.get_right_lane()
            if (other is None or str(other.lane_type) != 'Driving'
                    or abs(wrap(math.radians(other.transform.rotation.yaw-yaw))) > .4
                    or abs(other.transform.location.z-z) > 1.5
                    or road_key(other) in seen):
                break
            seen.add(road_key(other))
    return len(seen)


def at_distance(route, target):
    if not route or not math.isfinite(target) or target < 0:
        raise ValueError('路线不能为空且距离须为有限非负数')
    if target == 0:
        return route[0]
    arc = 0.
    for a,b in zip(route,route[1:]):
        arc += distance(a.transform.location,b.transform.location)
        if arc >= target:
            return b
    raise ValueError('场景摆放距离超出完整路线')


def _public_branches(wp, kind):
    candidates = wp.previous(8.) if kind == 'merge' else wp.next(8.)
    return sorted({road_key(p) for p in candidates
        if str(p.lane_type) == 'Driving'
        and abs(wrap(math.radians(p.transform.rotation.yaw-wp.transform.rotation.yaw))) < math.pi/4
        and abs(p.transform.location.z-wp.transform.location.z) < 2.})


def route_evidence(route):
    arcs = [0.]
    for a,b in zip(route,route[1:]):
        arcs.append(arcs[-1]+distance(a.transform.location,b.transform.location))
    junctions, events, seen_junctions = [], [], set()
    counts = [same_direction_lane_count(p) for p in route]
    for i,wp in enumerate(route):
        if wp.is_junction:
            jid = int(wp.junction_id)
            if jid not in seen_junctions:
                seen_junctions.add(jid)
                junctions.append({'junction_id':jid, 'route_s_m':arcs[i]})
        for kind in ('merge', 'diverge'):
            # 只报告真实选择路径上的单车道/多车道过渡与相容分支证据。
            lo, hi = max(0,i-40), min(len(route),i+41)
            before = [counts[j] for j in range(lo,i+1) if not route[j].is_junction]
            after = [counts[j] for j in range(i,hi) if not route[j].is_junction]
            # 路口内部每条连接都可能是单车道，不能拿它伪造匝道。
            transition = bool(before and after) and ((min(before)==1 and max(after)>=2)
                if kind=='merge' else (max(before)>=2 and min(after)==1))
            if transition:
                branches = _public_branches(wp,kind)
                if len(branches)>=2 and not any(e['kind']==kind and abs(e['route_s_m']-arcs[i])<20 for e in events):
                    events.append({'kind':kind, 'route_s_m':arcs[i], 'public_branches':[list(x) for x in branches],
                        'lane_count_before':min(before) if kind=='merge' else max(before),
                        'lane_count_after':max(after) if kind=='merge' else min(after)})
    return {'length_m':arcs[-1], 'junctions':junctions,'lane_transitions':events,
            'maximum_same_direction_lanes':max(counts),
            'height_range_m':[min(p.transform.location.z for p in route),max(p.transform.location.z for p in route)]}


def _evidence_failure(evidence, req):
    if len(evidence['junctions']) < max(req['min_junctions'], int(req['kind']=='junction')):
        return 'required_junction_not_reached'
    if evidence['maximum_same_direction_lanes'] < req['min_same_direction_lanes']:
        return 'required_parallel_lanes_not_reached'
    if req['kind'] in ('merge','diverge') and not any(e['kind']==req['kind'] for e in evidence['lane_transitions']):
        return 'required_'+req['kind']+'_transition_not_reached'
    return None


def _anchor_hit(wp, anchor):
    return (wp.road_id == anchor['road_id'] and wp.lane_id == anchor['lane_id']
            and abs(wp.s-anchor['s_m']) <= 2.05)


def _trace(start, length, req, anchors, max_yaw, route_filter):
    """2m采样有界DFS；锚点偏好仅排序，仍只走地图公开后继。"""
    points, arcs, choices, hits = [start], [0.], [None], [0]
    visited = {waypoint_key(start)}
    expanded, longest, failure = 0, 0., 'navigation_dead_end'
    while points:
        if arcs[-1] >= length:
            evidence = route_evidence(points)
            failure = ('required_anchor_not_reached' if hits[-1] != len(anchors)
                       else _evidence_failure(evidence,req))
            if failure is None and route_filter:
                failure = route_filter(points)
            if failure is None:
                return list(points), 'accepted', arcs[-1], evidence
            # 不枚举同一起点下指数数量的完整路线来凑场景属性。先按明确锚点/
            # 最小局部转角生成一条完整路线；属性或演员不符则检查下一个起点。
            return None, failure, arcs[-1], None
        if choices[-1] is None:
            if expanded >= req['search_limit']:
                return None, 'navigation_search_limit:'+failure, longest, None
            expanded += 1
            current = points[-1]
            # 不用未来路点跨越道路，锚点只提高合法搜索分支的优先级。
            next_anchor = anchors[hits[-1]] if hits[-1] < len(anchors) else None
            choices[-1] = sorted(current.next(2.), key=lambda p:(
                0 if next_anchor and p.road_id == next_anchor['road_id'] and p.lane_id == next_anchor['lane_id'] else 1,
                abs(wrap(math.radians(p.transform.rotation.yaw-current.transform.rotation.yaw))),waypoint_key(p)))
        if not choices[-1]:
            visited.remove(waypoint_key(points.pop()))
            arcs.pop();choices.pop();hits.pop()
            continue
        nxt = choices[-1].pop(0)
        old = points[-1]
        dist = distance(old.transform.location,nxt.transform.location)
        dz = abs(nxt.transform.location.z-old.transform.location.z)
        key = waypoint_key(nxt)
        if (str(nxt.lane_type) != 'Driving' or not .1 <= dist <= 3.2 or dz > 1.
                or abs(wrap(math.radians(nxt.transform.rotation.yaw-old.transform.rotation.yaw))) > math.radians(max_yaw)
                or key in visited):
            continue
        # 离散的s可能偏移，几何邻近重复检查每10m一次，避免回环凑长度。
        if any(distance(nxt.transform.location,p.transform.location)<.75 for p in points[:-5:5]):
            continue
        count = hits[-1]
        if count < len(anchors) and _anchor_hit(nxt,anchors[count]):
            count += 1
        points.append(nxt);arcs.append(arcs[-1]+dist);choices.append(None);hits.append(count)
        visited.add(key)
        longest = max(longest,arcs[-1])
    return None,failure,longest,None


def select_extended_route(road_map, settings, carla, audit_all=False, route_filter=None):
    """返回与旧模板相同的(selected,report)，兼容桥的选址诊断接口。"""
    req = validate_route_requirements(settings)
    length = float(settings.get('route_length_m',600.))
    anchors = settings.get('route_anchors',[])
    source = 'carla_spawn_points'
    if anchors:
        wp = road_map.get_waypoint_xodr(anchors[0]['road_id'],anchors[0]['lane_id'],anchors[0]['s_m'])
        if wp is None:
            raise ValueError('路线起点锚点不在当前地图公开车道上')
        candidates, source = [(-1,wp)], 'explicit_opendrive_anchor'
    else:
        poses = road_map.get_spawn_points()
        candidates = [(i,road_map.get_waypoint(p.location)) for i,p in enumerate(poses)]
        if not candidates:
            source = 'opendrive_20m_waypoints'
            candidates = list(enumerate(sorted(road_map.generate_waypoints(20.),key=waypoint_key)))
        index = settings.get('spawn_index')
        if index is not None:
            if not 0 <= index < len(candidates):raise ValueError('spawn_index超出当前地图生成点范围')
            candidates=[candidates[index]]
        else:
            random.Random(settings.get('seed',1)).shuffle(candidates)
    report={'version':EXTENDED_ROUTE_VERSION,'success':False,'map':str(getattr(road_map,'name','unknown')),
        'seed':settings.get('seed',1),'template':'scripted_traffic','route_length_m':length,
        'interaction_length_m':0.,'total_spawn_points':len(candidates),'selected_spawn_index':None,
        'rejection_counts':{},'candidates':[],'audit_all':bool(audit_all),'start_source':source,
        'requirements':req,'scope':'只读公开地图连通性与演员摆放检查；不是动态驾驶通过证明'}
    selected = None
    for index,wp in candidates:
        route,evidence,prefix,reason=None,None,0.,'start_not_driving'
        if wp is not None and str(wp.lane_type)=='Driving':
            if wp.is_junction:
                reason='start_in_junction'
            else:
                route,reason,prefix,evidence=_trace(wp,length,req,anchors[1:],
                    float(settings.get('navigation_step_yaw_limit_deg',30.)),route_filter)
        row={'spawn_index':index,'reason':reason,'valid_prefix_m':prefix}
        if wp is not None:
            row['start_waypoint']={'road_id':int(wp.road_id),'lane_id':int(wp.lane_id),'s_m':float(wp.s)}
        report['candidates'].append(row)
        if route is not None:
            if selected is None:
                lead=at_distance(route,float(settings.get('lead_distance_m',35.)))
                selected=(index,wp,route,lead)
                report.update(success=True,selected_spawn_index=index,route_evidence=evidence)
                report['selected_route']=[{'x_m':float(p.transform.location.x),'y_m':float(p.transform.location.y),
                    'z_m':float(p.transform.location.z),'yaw_deg':float(p.transform.rotation.yaw),
                    'junction':bool(p.is_junction),'road_id':int(p.road_id),'section_id':int(p.section_id),
                    'lane_id':int(p.lane_id),'s_m':float(p.s)} for p in route]
            if not audit_all:break
        else:
            report['rejection_counts'][reason]=report['rejection_counts'].get(reason,0)+1
    report['checked_spawn_points']=len(report['candidates'])
    report['longest_valid_prefix_m']=max([r['valid_prefix_m'] for r in report['candidates']] or [0.])
    return selected,report
