"""统一的平面几何、坐标变换和可追溯 Frenet 投影。"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Point, Polygon


class PolygonalGeometryError(ValueError):
    """Invalid source area cannot be silently repaired into driving permission."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def polygonal_area(geometry, *, context='geometry', require_nonempty=True):
    """Return the exact union of every valid two-dimensional component.

    Overlay can return a GeometryCollection containing road area and isolated
    touching edges/points. Those lower-dimensional pieces are not drivable
    surface. Polygon shells, holes and disconnected areas are all retained;
    no hull, buffer, snapping, area threshold or invalid-ring repair is used.
    Physical lane/marking boundaries remain a separate, unchanged contract.
    """
    evidence = dict(version='exact-polygonal-area-1.0', context=context,
        source_type=getattr(geometry, 'geom_type', None), polygon_components=0,
        interior_rings=0, empty_components=0, lower_dimensional_components={},
        lower_dimensional_length_m=0., policy='all_polygon_components_union;no_buffer_hull_snap_or_ring_repair')
    def reject(reason):
        evidence['rejection'] = reason
        raise PolygonalGeometryError(context+': '+reason, evidence)
    if geometry is None or not hasattr(geometry, 'geom_type'):
        reject('missing geometry')
    coordinates=shapely.get_coordinates(geometry)
    if not np.isfinite(coordinates).all():
        reject('nonfinite coordinates')
    parts=[]
    def visit(item):
        if item.is_empty:
            evidence['empty_components'] += 1
        elif item.geom_type == 'Polygon':
            if not item.is_valid:
                reject('invalid source polygon: '+str(shapely.is_valid_reason(item)))
            parts.append(item)
            evidence['polygon_components'] += 1
            evidence['interior_rings'] += len(item.interiors)
        elif item.geom_type in ('MultiPolygon', 'GeometryCollection'):
            for child in item.geoms:
                visit(child)
        elif item.geom_type in ('LineString', 'LinearRing', 'MultiLineString', 'Point', 'MultiPoint'):
            counts=evidence['lower_dimensional_components']
            counts[item.geom_type]=counts.get(item.geom_type,0)+1
            evidence['lower_dimensional_length_m'] += float(item.length)
        else:
            reject('unsupported component type: '+item.geom_type)
    visit(geometry)
    result=shapely.union_all(parts) if len(parts)>1 else parts[0] if parts else Polygon()
    if result.is_empty:
        if require_nonempty:
            reject('no two-dimensional area')
    elif result.geom_type not in ('Polygon','MultiPolygon') or not result.is_valid:
        reject('polygon component union is not a valid polygonal area')
    evidence.update(result_type=result.geom_type,area_m2=float(result.area),
                    lower_dimensions_are_driving_surface=False,physical_boundaries_unchanged=True)
    return result,evidence


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _pose(position, heading):
    if hasattr(position, "position"):
        return np.asarray(position.position), float(position.heading)
    if heading is None:
        raise ValueError("坐标变换缺少朝向")
    return np.asarray(position), float(heading)


def world_to_ego(points, position, heading=None):
    """输出 x 向前、y 向左；支持单点与任意批量点。"""
    position, heading = _pose(position, heading)
    rotation = np.array([[np.cos(heading), -np.sin(heading)],
                         [np.sin(heading), np.cos(heading)]])
    return (np.asarray(points) - position) @ rotation


def ego_to_world(points, position, heading=None):
    position, heading = _pose(position, heading)
    rotation = np.array([[np.cos(heading), -np.sin(heading)],
                         [np.sin(heading), np.cos(heading)]])
    return np.asarray(points) @ rotation.T + position


def rectangle_corners(position, heading, length, width):
    local = np.array([[length/2, width/2], [length/2, -width/2],
                      [-length/2, -width/2], [-length/2, width/2]])
    return ego_to_world(local, position, heading)


def actor_shape(actor, position=None, heading=None):
    """地图覆盖和绘制用几何；圆形实体的净距与相交使用解析函数。"""
    position = actor.position if position is None else np.asarray(position)
    heading = actor.heading if heading is None else heading
    if actor.kind == "pedestrian":
        return Point(position).buffer(actor.radius or actor.width/2, quad_segs=16)
    return Polygon(rectangle_corners(position, heading, actor.length, actor.width))


def polyline_lengths(reference):
    reference = np.asarray(reference, float)
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(reference, axis=0), axis=1))]


def clean_polyline(reference):
    reference = np.asarray(reference, float)
    if len(reference) < 2:
        raise ValueError("参考线至少需要两个不同的点")
    result = reference[np.r_[True, np.linalg.norm(np.diff(reference, axis=0), axis=1) > 1e-7]]
    if len(result) < 2:
        raise ValueError("参考线长度为零")
    return result


def project_polyline(points, reference, previous_s=None, heading=None, window=40.):
    """投影到共同弧长参考线；连续投影可限制搜索避免环道跳段。"""
    reference = clean_polyline(reference)
    points = np.asarray(points, float)
    single = points.ndim == 1
    points = np.atleast_2d(points)
    segment = np.diff(reference, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    starts=reference[:-1]
    if previous_s is not None:
        # 先裁剪再广播，避免整条长道路分配点数×线段数的大矩阵。
        allowed=(cumulative[:-1]<=float(previous_s)+window)&(cumulative[1:]>=float(previous_s)-window)
        if allowed.any():
            starts=starts[allowed];segment=segment[allowed];lengths=lengths[allowed];cumulative=cumulative[:-1][allowed]
    t = np.clip(np.einsum("nsi,si->ns", points[:, None]-starts, segment)
                / lengths**2, 0., 1.)
    closest = starts + t[..., None] * segment
    residual = points[:, None] - closest
    cost = np.sum(residual**2, axis=-1)
    tangent = np.arctan2(segment[:, 1], segment[:, 0])
    if heading is not None:
        cost += .25 * wrap_angle(tangent-np.asarray(heading)[..., None])**2
    index = np.argmin(cost, axis=1)
    row = np.arange(len(points))
    s = cumulative[index] + t[row, index] * lengths[index]
    unit = segment[index] / lengths[index, None]
    d = unit[:, 0] * residual[row, index, 1] - unit[:, 1] * residual[row, index, 0]
    h = tangent[index]
    return (float(s[0]), float(d[0]), float(h[0])) if single else (s, d, h)


def sample_polyline(reference, s):
    reference = clean_polyline(reference)
    arc = polyline_lengths(reference)
    s = np.asarray(s)
    return np.stack([np.interp(s, arc, reference[:, dim]) for dim in range(2)], axis=-1)


def rectangle_separation(corners_a, corners_b):
    """分离轴最大间隙；正值分离、负值穿透，不把零净距误当安全。"""
    a, b = np.asarray(corners_a), np.asarray(corners_b)
    edges = np.vstack((np.diff(a[:3], axis=0), np.diff(b[:3], axis=0)))
    axes = np.stack([-edges[:, 1], edges[:, 0]], axis=-1)
    axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
    ap, bp = a @ axes.T, b @ axes.T
    return float(np.max(np.maximum(bp.min(axis=0)-ap.max(axis=0),
                                   ap.min(axis=0)-bp.max(axis=0))))


def signed_body_distance(actor_a, actor_b, position_a=None, heading_a=None,
                         position_b=None, heading_b=None):
    """圆形使用解析净距；矩形之间分离时取欧氏距离、相交时取分离轴值。"""
    pa=actor_a.position if position_a is None else np.asarray(position_a)
    pb=actor_b.position if position_b is None else np.asarray(position_b)
    ha=actor_a.heading if heading_a is None else heading_a
    hb=actor_b.heading if heading_b is None else heading_b
    circle_a=actor_a.kind=="pedestrian";circle_b=actor_b.kind=="pedestrian"
    if circle_a and circle_b:
        return float(np.linalg.norm(pa-pb)-(actor_a.radius or actor_a.width/2)-(actor_b.radius or actor_b.width/2))
    if circle_a:
        return float(circle_rectangle_distance(pa,actor_a.radius or actor_a.width/2,pb,hb,actor_b.length,actor_b.width))
    if circle_b:
        return float(circle_rectangle_distance(pb,actor_b.radius or actor_b.width/2,pa,ha,actor_a.length,actor_a.width))
    a = actor_shape(actor_a, position_a, heading_a)
    b = actor_shape(actor_b, position_b, heading_b)
    if not a.intersects(b):
        return float(a.distance(b))
    return rectangle_separation(np.array(a.exterior.coords)[:4], np.array(b.exterior.coords)[:4])


def circle_rectangle_distance(circle_position,radius,rectangle_position,heading,length,width):
    """圆—有向矩形解析有符号净距；支持同形批量，接触为零、重叠为负。"""
    relative=np.asarray(circle_position)-np.asarray(rectangle_position)
    c,s=np.cos(heading),np.sin(heading)
    local=np.stack((relative[...,0]*c+relative[...,1]*s,
                    -relative[...,0]*s+relative[...,1]*c),axis=-1)
    q=np.abs(local)-np.array([length/2,width/2])
    return np.linalg.norm(np.maximum(q,0.),axis=-1)+np.minimum(np.max(q,axis=-1),0.)-radius
