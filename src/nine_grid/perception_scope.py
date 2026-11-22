"""本帧模型/EKF候选：车道范围对象与实际BEV内动态对象的并集。

八向代表属于检索表示，不是感知白名单。BEV准入不授予自车跨线或驶入
对向车道的权限，也不把范围外静态环境实体重新解释为道路障碍。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np
from shapely.geometry import Point, Polygon

from .geometry import actor_shape, ego_to_world, world_to_ego
from .object_scope import select_scope_objects


VERSION = 'lane-plus-bev-dynamic-perception-1.0'
DYNAMIC_KINDS = ('vehicle', 'bicycle', 'pedestrian')
# 与既有BEV绘制逐像素一致。配置文件的其它bev取值此前没有改变实际图像。
BEV_VIEWPORT = dict(front_m=70., rear_m=30., left_m=25., right_m=25.,
                    width_px=500, height_px=1000, meters_per_pixel=.1,
                    ego_pixel=[250., 700.])


def bev_viewport_settings():
    """实际渲染窗口；返回独立副本，调用者不能修改公共图像坐标契约。"""
    return deepcopy(BEV_VIEWPORT)


def bev_pixels(points, ego):
    """当前BEV的米到像素变换：自车朝上，x前y左。"""
    local = np.atleast_2d(world_to_ego(points, ego))
    setting = BEV_VIEWPORT
    scale = 1./setting['meters_per_pixel']
    return np.column_stack((setting['ego_pixel'][0]-local[:, 1]*scale,
                            setting['ego_pixel'][1]-local[:, 0]*scale))


def bev_viewport_polygon(ego):
    setting = BEV_VIEWPORT
    local = np.array([[-setting['rear_m'], -setting['right_m']],
                      [setting['front_m'], -setting['right_m']],
                      [setting['front_m'], setting['left_m']],
                      [-setting['rear_m'], setting['left_m']]])
    return Polygon(ego_to_world(local, ego))


def _selected_signature(scene, actors):
    """验证已筛选冻结帧再次使用时，状态和车道几何没有被原地改变。"""
    def entity(actor):
        return dict(id=actor.id, kind=actor.kind, position=actor.position,
                    velocity=actor.velocity, acceleration=actor.acceleration,
                    heading=actor.heading, length=actor.length, width=actor.width,
                    radius=actor.radius, visible=bool(actor.visible), lane_id=actor.lane_id,
                    history=actor.history)

    def numeric(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError('感知范围身份签名含不支持值：'+type(value).__name__)

    value = dict(frame_id=scene.frame_id, time=float(scene.time), coordinates=scene.coordinates,
                 ego=entity(scene.ego), actors=[entity(actor) for actor in sorted(actors, key=lambda a: a.id)],
                 lane_map_complete=scene.diagnostics.get('lane_map_complete'),
                 lanes=[dict(id=lane.id, center=lane.center, polygon=lane.polygon, width=lane.width,
                             successors=lane.successors, left=lane.left, right=lane.right,
                             boundaries=lane.boundaries)
                        for lane in sorted(scene.lanes, key=lambda a: a.id)],
                 route=scene.route, route_lane_ids=scene.route_lane_ids)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                     allow_nan=False, default=numeric).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def select_perception_objects(scene, config=None):
    """选择全部当前候选，既不预测未来，也不恢复已被上游删除的真值。"""
    previous = scene.diagnostics.get('perception_scope')
    attention_options=(config or {}).get('planning_objects')
    if attention_options is None:
        attention_options=previous.get('attention_policy',{}) if isinstance(previous,dict) else {}
    attention_policy=dict(include_attention=attention_options.get('include_attention',True),
                          max_attention_objects=attention_options.get('max_attention_objects',3))
    identities = sorted(actor.id for actor in scene.actors)
    if scene.diagnostics.get('planning_scope') is not None:
        raise ValueError('规划子集不能冒充完整感知候选，须从完整冻结帧重新构建')
    if isinstance(previous, dict):
        if previous.get('version') != VERSION:
            raise ValueError('已筛选感知场景的范围协议已改变，须从完整冻结帧重新构建')
        if (previous.get('frame_id') != scene.frame_id
                or previous.get('time') != float(scene.time)):
            raise ValueError('已筛选感知场景的原帧身份或时间不一致，须从完整冻结帧重新构建')
        missing = set(previous.get('selected_ids', []))-set(identities)
        if missing:
            raise ValueError('已筛选感知场景缺少原帧候选，须从完整冻结帧重新构建：'+','.join(sorted(missing)))
        if identities == previous.get('selected_ids'):
            if previous.get('selected_input_signature') != _selected_signature(scene, scene.actors):
                raise ValueError('已筛选感知场景的当前对象或车道几何发生变化，须从完整冻结帧重新构建')
            # 配置显式改变时不能从子集扩大范围。
            if config is not None:
                _, requested = select_scope_objects(scene, config)
                if requested['settings'] != previous['settings']['lane']:
                    raise ValueError('已筛选感知场景不能用新配置重解释车道范围，须从完整冻结帧重新构建')
                if attention_policy != previous.get('attention_policy',attention_policy):
                    raise ValueError('已筛选感知场景不能用新配置重解释关注扩展，须从完整冻结帧重新构建')
            return list(scene.actors), deepcopy(previous)

    if config is None and isinstance(previous, dict) and 'object_scope' not in scene.diagnostics:
        config = {'planning_objects': previous['settings']['lane']}
    lane_actors, lane_scope = select_scope_objects(scene, config)
    lane_ids = {actor.id for actor in lane_actors}
    viewport = bev_viewport_polygon(scene.ego)
    dynamic_ids = set()
    for actor in scene.actors:
        if actor.id == scene.ego.id or not actor.visible or actor.kind not in DYNAMIC_KINDS:
            continue
        # 行人绘图使用24段圆周，保持边缘部分可见准入与实际BEV相同。
        body = (Point(actor.position).buffer(actor.radius or actor.width/2, quad_segs=24)
                if actor.kind == 'pedestrian' else actor_shape(actor))
        if body.intersects(viewport):
            dynamic_ids.add(actor.id)
    selected_ids = lane_ids | dynamic_ids
    selected = sorted((actor for actor in scene.actors if actor.id in selected_ids), key=lambda actor: actor.id)
    excluded = {row['id']: row['reason'] for row in lane_scope['excluded'] if row['id'] not in selected_ids}
    for actor in scene.actors:
        if (actor.id not in selected_ids and actor.id != scene.ego.id
                and actor.visible and actor.kind in DYNAMIC_KINDS):
            excluded[actor.id] = 'outside_lane_scope_and_current_bev_viewport'
    objects = {}
    for actor in selected:
        row = deepcopy(lane_scope['objects'].get(actor.id, {}))
        row['sources'] = ([name for name, members in (('lane_scope', lane_ids), ('bev_dynamic', dynamic_ids))
                           if actor.id in members])
        row.setdefault('roles', [])
        row.setdefault('memberships', {})
        objects[actor.id] = row
    diagnostic = dict(version=VERSION, frame_id=scene.frame_id, time=float(scene.time),
        coordinates=scene.coordinates, settings=dict(lane=deepcopy(lane_scope['settings']), bev=bev_viewport_settings()),
        lane_scope=lane_scope, lane_scope_ids=sorted(lane_ids), bev_dynamic_ids=sorted(dynamic_ids),
        attention_policy=attention_policy,
        outside_lane_ids=sorted(dynamic_ids-lane_ids), selected_ids=sorted(selected_ids),
        selected_count=len(selected_ids), input_count=len(scene.actors), objects=objects,
        excluded=[dict(id=key, reason=excluded[key]) for key in sorted(excluded)],
        selection_rule='all_lane_scope_objects_union_current_visible_bev_dynamic_body_intersection',
        limitation='本帧已观测候选；BEV内不判断遮挡，不代表自车可驶入其车道；不读取隐藏未来')
    diagnostic['selected_input_signature'] = _selected_signature(scene, selected)
    return selected, diagnostic


def perception_scene(scene, config=None):
    """返回独立候选列表，完整审计快照和对象状态不被原地改写。"""
    selected, diagnostic = select_perception_objects(scene, config)
    diagnostics = {**scene.diagnostics, 'object_scope': deepcopy(diagnostic['lane_scope']),
                   'perception_scope': deepcopy(diagnostic)}
    # 新的候选比常规车道范围更宽，必须重新冻结同一帧路权输入；不能沿用旧窄集合。
    diagnostics.pop('road_rule_observations', None)
    return replace(scene, actors=selected, diagnostics=diagnostics), diagnostic
