"""CARLA SDK 子进程到平台无关场景契约；主环境不导入 carla。"""
from __future__ import annotations

import base64
import binascii
import copy
import json
import math
from pathlib import Path
import queue
import subprocess
import threading
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union

from .contracts import Actor, Lane, SceneSnapshot, plain
from .geometry import actor_shape, clean_polyline, ego_to_world, polyline_lengths, project_polyline, sample_polyline


class BridgeProcess:
    """有限等待的 JSON-lines 请求；EOF 允许 SDK 桥执行 finally 清理。"""
    def __init__(self, python, output_dir, timeout_s=30., bridge_script=None):
        self.timeout = float(timeout_s)
        self.queue = queue.Queue()
        self.counter = 0
        self.closed = False
        self.stderr = open(Path(output_dir) / 'carla_bridge.log', 'a', encoding='utf-8')
        script = Path(bridge_script) if bridge_script else Path(__file__).with_name('carla_bridge.py')
        try:
            self.process = subprocess.Popen([str(python), '-u', str(script)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.stderr, text=True, encoding='utf-8', bufsize=1)
        except BaseException:
            self.stderr.close()
            raise
        def reader():
            try:
                for line in self.process.stdout:
                    self.queue.put(line)
            finally:
                self.queue.put(None)
        self.reader = threading.Thread(target=reader, name='carla-sdk-output', daemon=True)
        self.reader.start()

    def call(self, method, **params):
        if self.closed:
            raise RuntimeError('CARLA 桥已关闭')
        self.counter += 1
        identity = self.counter
        try:
            self.process.stdin.write(json.dumps({'id': identity, 'method': method, 'params': params},
                                               ensure_ascii=False, allow_nan=False) + '\n')
            self.process.stdin.flush()
            response = self.queue.get(timeout=self.timeout)
        except (OSError, queue.Empty) as exc:
            raise RuntimeError('CARLA 桥通信失败或超时；详见 carla_bridge.log') from exc
        if response is None:
            raise RuntimeError('CARLA SDK 桥提前退出；请检查 SDK 环境与 carla_bridge.log')
        try:
            result = json.loads(response)
        except ValueError as exc:
            raise RuntimeError('CARLA 桥输出不是有效 JSON') from exc
        if result.get('id') != identity:
            raise RuntimeError('CARLA 桥响应序号不一致，拒绝混用迟到结果')
        if 'error' in result:
            raise RuntimeError('CARLA 适配失败：' + result['error'])
        return result['result']

    def close(self):
        if self.closed:
            return
        self.closed = True
        forced = False
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=self.timeout + 2.)
            except subprocess.TimeoutExpired:
                forced = True
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5.)
        finally:
            self.stderr.close()
            if self.process.stdout is not None:
                self.process.stdout.close()
        if forced or self.process.returncode != 0:
            raise RuntimeError('CARLA SDK桥未正常收尾（退出码{}）；请检查carla_bridge.log及独立世界残留资源'.format(self.process.returncode))


def decode_depth(path):
    """兼容读取已有原始深度资产；新运行不再写入或依赖深度图片。"""
    with Image.open(path) as source:
        rgb = np.asarray(source.convert('RGB'), dtype=np.float64)
    return (rgb[:, :, 0] + 256 * rgb[:, :, 1] + 65536 * rgb[:, :, 2]) * (1000. / 16777215.)


def camera_depth(camera):
    """同帧临时BGRA深度解码；缺证据/损坏不能当作空区，也不静默回退旧文件。"""
    if 'depth_transport' not in camera:
        if 'depth_path' not in camera:
            raise ValueError('相机缺少临时深度观测证据')
        depth = decode_depth(camera['depth_path'])
        if depth.shape != (camera['height'], camera['width']):
            raise ValueError('已有深度资产尺寸与相机不一致')
        return depth
    payload = camera['depth_transport']
    if not isinstance(payload, dict) or payload.get('encoding') != 'zlib-base64-bgra8-1.0':
        raise ValueError('临时深度编码无效')
    width, height = camera.get('width'), camera.get('height')
    if any(type(v) is not int or v <= 0 for v in (width,height)):
        raise ValueError('临时深度相机尺寸无效')
    if (payload.get('width') != width or payload.get('height') != height or
            type(payload.get('frame')) is not int or payload['frame'] != camera.get('frame')):
        raise ValueError('临时深度尺寸或帧号不一致')
    timestamp, source_time = payload.get('timestamp'), camera.get('timestamp')
    if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
            for v in (timestamp,source_time)) or abs(timestamp-source_time) > 1e-6):
        raise ValueError('临时深度时间戳不一致')
    encoded = payload.get('data')
    expected = width*height*4
    if not isinstance(encoded,str) or len(encoded) > 4*((expected*2+128+2)//3):
        raise ValueError('临时深度载荷长度无效')
    try:
        compressed = base64.b64decode(encoded, validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, expected+1)
    except (ValueError,binascii.Error,zlib.error) as exc:
        raise ValueError('临时深度载荷损坏') from exc
    if (len(raw) != expected or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data):
        raise ValueError('临时深度字节数或压缩边界无效')
    bgra = np.frombuffer(raw,dtype=np.uint8).reshape(height,width,4).astype(np.float64)
    return (bgra[:,:,2]+256*bgra[:,:,1]+65536*bgra[:,:,0])*(1000./16777215.)


def retained_camera_metadata(camera):
    """只保留RGB及标定/观测来源；临时深度数据和旧深度路径不进入落盘资产。"""
    keys = ('view','rgb_path','width','height','fov_deg','frame','timestamp',
            'world_to_camera_ue','camera_to_world_ue','depth_frame','depth_timestamp','depth_encoding')
    result = {key:copy.deepcopy(camera[key]) for key in keys if key in camera}
    result['depth_retention'] = 'not_saved_by_current_run'
    return result


def observed_geometry(ego, cameras, cfg):
    """多个高度和格角均有相机深度支持才确认空区，保留盲区孔洞。"""
    resolution = float(cfg.get('observation_resolution_m', 1.0))
    if not np.isfinite(resolution) or resolution <= .02 or resolution > 2:
        raise ValueError('观测掩码网格边长必须位于 (0,2] 米')
    # 检索画布固定；画布之外当前可见实体仍留在规划集合。
    xs, ys = np.arange(-30., 70., resolution), np.arange(-25., 25., resolution)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    centers = np.stack((xx.ravel(), yy.ravel()), axis=1)
    all_known = np.ones(len(centers), dtype=bool)
    heights = cfg.get('observation_heights_m', [.2, .8])
    tolerance = float(cfg.get('depth_empty_tolerance_m', .05))
    if not isinstance(heights, list) or not heights or not all(np.isfinite(h) and h >= 0 for h in heights):
        raise ValueError('观测高度必须是非空的有限非负米值列表')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('深度空区容差必须为有限非负米值')
    world_origin_z = float(ego['origin_z'])
    prepared = [(c, camera_depth(c)) for c in cameras]
    for offset in ([.01, .01], [resolution - .01, .01], [.01, resolution - .01], [resolution - .01, resolution - .01]):
        world = ego_to_world(centers + offset, np.asarray(ego['position']), float(ego['heading']))
        for height in heights:
            points = np.c_[world[:, 0], -world[:, 1], np.full(len(world), world_origin_z + height), np.ones(len(world))]
            visible = np.zeros(len(points), dtype=bool)
            for camera, depth in prepared:
                transformed = points @ np.asarray(camera['world_to_camera_ue'], float).T
                forward = transformed[:, 0]
                focal = camera['width'] / (2 * np.tan(np.deg2rad(camera['fov_deg']) / 2))
                safe_forward = np.maximum(forward, .001)
                u = camera['width'] / 2 + focal * transformed[:, 1] / safe_forward
                v = camera['height'] / 2 - focal * transformed[:, 2] / safe_forward
                valid = (forward > .05) & (u >= 1) & (u < camera['width'] - 1) & (v >= 1) & (v < camera['height'] - 1)
                indexes = np.flatnonzero(valid)
                measured = depth[v[indexes].astype(int), u[indexes].astype(int)]
                # 不把天际饱和/无有效深度当作确认空，深度容差仅消除编码误差。
                visible[indexes] |= (measured >= forward[indexes] + tolerance) & (measured < 999.)
            all_known &= visible
    known = all_known.reshape(len(xs), len(ys))
    parts = []
    for i, row in enumerate(known):
        starts = np.flatnonzero(np.diff(np.r_[False, row].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[row, False].astype(int)) == -1)
        for start, end in zip(starts, ends):
            corners = np.array([[xs[i], ys[start]], [xs[i] + resolution, ys[start]],
                                [xs[i] + resolution, ys[end] + resolution], [xs[i], ys[end] + resolution]])
            parts.append(Polygon(ego_to_world(corners, np.asarray(ego['position']), ego['heading'])))
    return unary_union(parts), {'method': 'same_frame_depth_multiple_height_cell_corners',
        'resolution_m': resolution, 'height_above_actor_origin_m': heights,
        'confirmed_fraction': float(known.mean()), 'depth_tolerance_m': tolerance,
        'limitation': '平面道路深度几何近似；保守盲区不补空，需真实传感器验证'}


class CameraVideoRecorder:
    def __init__(self, adapter, path):
        self.adapter, self.path = adapter, Path(path)
        self.writer = None
        self.frames = 0
        self.closed = False

    def _capture_frame(self):
        if self.closed:
            return
        import cv2
        image = self.adapter.render()
        if image is None:
            return
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'),
                self.adapter.planner_hz, (image.shape[1], image.shape[0]))
            if not self.writer.isOpened():
                raise RuntimeError('CARLA 原始相机录像编码器启动失败')
        self.writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.closed = True

    def metadata(self):
        return {'path': str(self.path), 'frames': self.frames, 'fps': self.adapter.planner_hz,
                'source': 'CARLA_synchronized_six_camera_mosaic',
                'encoded_duration_s': self.frames / self.adapter.planner_hz,
                'plan_overlay': False, 'note': '六相机真实画面；规划叠加保留在BEV诊断图，原始模型图像不被修改'}


class CarlaAdapter:
    def __init__(self, config, seed, episode_id, live=False, bridge_factory=BridgeProcess):
        self.config = copy.deepcopy(config)
        self.cfg = copy.deepcopy(config.get('carla', {}))
        self.seed, self.episode_id, self.live = seed, episode_id, live
        self.planner_hz = float(config.get('clock', {}).get('planner_hz', 10))
        self.dt = 1. / self.planner_hz
        self.cfg.update(physics_hz=int(config.get('clock', {}).get('physics_hz', 50)), planner_hz=int(self.planner_hz))
        for key in ('acceleration_min_mps2', 'acceleration_max_mps2', 'steering_abs_max_rad'):
            if key in config.get('mpc', {}):
                self.cfg[key] = config['mpc'][key]
        self.cfg['perception_radius_m'] = config.get('perception', {}).get('radius_m', 120.)
        self.bridge_factory = bridge_factory
        self.bridge = None
        self.output_dir = None
        self._snapshot = None
        self._raw = None
        self._history = {}
        self._next_history = 0.
        self._route_s = None
        self._previous_control = np.zeros(2)
        self._planned = None
        self._recorder = None
        self._mosaic = None
        self._step = 0
        self._confirmed_lane = None
        self._pending_lane = None
        self._pending_since = None
        self._reference_revision = 0
        self._progress_offset = 0.
        self.closed = False

    def set_output_directory(self, path):
        if self.bridge is not None:
            raise RuntimeError('CARLA 连接后不能更换资产目录')
        self.output_dir = Path(path).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _initialize(self):
        if self.output_dir is None:
            raise RuntimeError('CARLA 采集必须先绑定本轮输出目录')
        self.bridge = self.bridge_factory(self.cfg.get('python', 'python'), self.output_dir,
            timeout_s=float(self.cfg.get('timeout_s', 30)) * 2 + 10)
        try:
            raw = self.bridge.call('init', config=self.cfg, directory=str(self.output_dir), seed=int(self.seed))
            mapping = raw.pop('map')
            self.lanes = [Lane(**{**item, 'center': np.asarray(item['center'], float), 'polygon': np.asarray(item['polygon'], float),
                'boundaries': [{**b, 'points': np.asarray(b['points'], float)} for b in item.get('boundaries', [])]}) for item in mapping['lanes']]
            if not self.lanes:
                raise ValueError('CARLA 地图没有可用的同层车道')
            self.drivable = unary_union([Polygon(l.polygon).buffer(0) for l in self.lanes])
            self.route = clean_polyline(mapping['route'])
            self.route_lane_ids = mapping['route_lane_ids']
            self.crosswalks = mapping.get('crosswalks')
            self._navigation_route = self.route.copy()
            self._navigation_lane_ids = list(self.route_lane_ids)
            self._goal_s = float(polyline_lengths(self.route)[-1]) - float(self.cfg.get('arrival_margin_m', 3.))
            if not set(self.route_lane_ids).issubset({l.id for l in self.lanes}):
                raise ValueError('自车路线超出已提取的同层地图；不能补造缺失车道')
            (self.output_dir / 'carla_map.json').write_text(json.dumps(mapping, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            self._raw = raw
            self._confirmed_lane = raw['ego']['lane_id']
        except BaseException as original:
            try:
                self.close()
            except Exception as cleanup:
                original.add_note('另有清理异常：' + str(cleanup))
            raise

    def _convert_actor(self, value):
        fields = {key: value[key] for key in ('id', 'kind', 'heading', 'length', 'width', 'lane_id', 'visible', 'radius')}
        for key in ('position', 'velocity', 'acceleration'):
            fields[key] = np.asarray(value[key], float)
        for key in ('wheelbase_m', 'rear_axle_to_center_m'):
            if key in value:
                fields[key] = value[key]
        return Actor(**fields)

    def _lane_route(self, start_lane):
        """沿已知自车道路分支连接新车道，不修改导航目的地，不访问周车路线。"""
        lanes = {lane.id: lane for lane in self.lanes}
        sections = [tuple(identity.split('|')[:2]) for identity in self._navigation_lane_ids]
        section = tuple(start_lane.id.split('|')[:2])
        if section not in sections:
            raise ValueError('换道后的道路不属于已知导航分支')
        index = sections.index(section)
        chain = [start_lane]
        for target_section in sections[index + 1:]:
            if target_section == tuple(chain[-1].id.split('|')[:2]):
                continue
            options = [lanes[key] for key in chain[-1].successors if key in lanes
                       and tuple(key.split('|')[:2]) == target_section]
            if not options:
                raise ValueError('目标车道缺少通向已知导航下一路段的公开连接')
            chain.append(min(options, key=lambda lane: (np.linalg.norm(lane.center[0]-chain[-1].center[-1]), lane.id)))
        path = clean_polyline(np.vstack([lane.center for lane in chain]))
        start = max(0., project_polyline(self._navigation_route[0], path)[0]) if index == 0 else 0.
        end = project_polyline(self._navigation_route[-1], path)[0]
        if end-start < 2.:
            raise ValueError('目标车道导航延伸长度不足')
        arc = polyline_lengths(path)
        path = clean_polyline(np.vstack([sample_polyline(path, start), path[(arc>start)&(arc<end)], sample_polyline(path, end)]))
        if not self.drivable.buffer(.02).covers(LineString(path)):
            raise ValueError('新车道导航连接穿出公开可行驶区域，不能直线补造连接')
        return path, [lane.id for lane in chain]

    def _confirm_lane(self, ego, timestamp):
        """半车体进入邻道并持续0.3秒后切换坐标参考；保持导航进度连续。"""
        candidate = ego.lane_id
        if candidate is None or candidate == self._confirmed_lane:
            self._pending_lane = self._pending_since = None
            return
        lanes = {lane.id: lane for lane in self.lanes}
        old, new = lanes.get(self._confirmed_lane), lanes.get(candidate)
        if new is None:
            return
        if candidate in self.route_lane_ids and old is not None and candidate in old.successors:
            self._confirmed_lane = candidate
            self._pending_lane = self._pending_since = None
            return
        if old is None or candidate not in (old.left, old.right):
            return
        geometry = Polygon(new.polygon)
        body = actor_shape(ego)
        settings = self.config.get('coordinates', {})
        if not geometry.covers(Point(ego.position)) or body.intersection(geometry).area/body.area <= float(settings.get('lane_switch_overlap_fraction', .5)):
            self._pending_lane = self._pending_since = None
            return
        if self._pending_lane != candidate:
            self._pending_lane, self._pending_since = candidate, timestamp
        if timestamp-self._pending_since+1e-8 < float(settings.get('lane_switch_hold_s', .3)):
            return
        path, identities = self._lane_route(new)
        old_progress = project_polyline(ego.position, self.route, self._route_s, ego.heading)[0]+self._progress_offset
        new_progress = project_polyline(ego.position, path, heading=ego.heading)[0]
        self.route, self.route_lane_ids = path, identities
        self._progress_offset = old_progress-new_progress
        self._route_s = new_progress
        self._goal_s = float(polyline_lengths(path)[-1])-float(self.cfg.get('arrival_margin_m', 3.))
        self._confirmed_lane = candidate
        self._pending_lane = self._pending_since = None
        self._reference_revision += 1

    def snapshot(self):
        if self.closed:
            raise RuntimeError('CARLA 适配器已关闭')
        if self.bridge is None:
            self._initialize()
        if self._snapshot is not None:
            return self._snapshot
        raw = self._raw
        ego = self._convert_actor(raw['ego'])
        actors = [self._convert_actor(a) for a in raw['actors']]
        timestamp = float(raw['time'])
        self._confirm_lane(ego, timestamp)
        history_cfg = self.config.get('clock', {})
        history_window = float(history_cfg.get('history_seconds', 3.))
        if timestamp + 1e-8 >= self._next_history:
            for actor in [ego] + actors:
                self._history.setdefault(actor.id, []).append({'time': timestamp, 'position': actor.position.tolist(),
                    'velocity': actor.velocity.tolist(), 'acceleration': actor.acceleration.tolist(),
                    'heading': actor.heading, 'observed': True})
            self._next_history = timestamp + 1. / float(history_cfg.get('history_hz', 2))
        for identity in list(self._history):
            self._history[identity] = [p for p in self._history[identity] if timestamp - p['time'] <= history_window + 1e-8]
            if not self._history[identity]:
                del self._history[identity]
        for actor in [ego] + actors:
            actor.history = copy.deepcopy(self._history.get(actor.id, []))
        self._route_s = project_polyline(ego.position, self.route, self._route_s, ego.heading)[0]
        observed, observation_audit = observed_geometry(raw['ego'], raw['cameras'], self.cfg)
        observation_audit['depth_storage'] = 'runtime_only; retained_RGB_and_observation_mask'
        # 已确认部分可见对象按既定权限提供完整框；这些实体面积本身是已知占据，
        # 不能被“后方空区不可见”误删。框外遮挡仍为未知，不补隐藏对象。
        observed = unary_union([observed, actor_shape(ego)] + [actor_shape(a) for a in actors])
        observation_audit['visible_body_support'] = '允许部分可见对象的完整几何框；不填补框外遮挡'
        frame_id = '{}:{:06d}'.format(self.episode_id, self._step)
        folder = self.output_dir / 'camera_frames' / '{:08d}'.format(raw['frame'])
        mosaic = self._build_mosaic(raw['cameras'])
        mosaic_path = folder / 'multi_camera_mosaic.png'
        mosaic.save(mosaic_path)
        self._mosaic = np.asarray(mosaic)
        # RPC载荷仅在本帧内存使用；不换成另一种深度文件或写入JSON/记忆库。
        retained_cameras = [retained_camera_metadata(camera) for camera in raw['cameras']]
        manifest = {'frame_id': frame_id, 'time': timestamp, 'carla_frame': raw['frame'],
            'carla_timestamp': raw['timestamp'], 'cameras': retained_cameras,
            'calibration_version': 'carla-pinhole-rgb-depth-1.0', 'storage_version': 'carla-rgb-only-1.0'}
        manifest_path = folder / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        retained_raw = {**raw, 'cameras': retained_cameras, 'storage_version': 'carla-rgb-only-1.0',
            'observed_support_wkt': observed.wkt, 'observation_evidence': observation_audit}
        (folder / 'current_observed_state.json').write_text(json.dumps(retained_raw, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        stop_s, light_records = None, []
        for light in raw.get('traffic_lights', []):
            relevant = []
            for stop in light['stops']:
                if stop['lane_id'] not in self.route_lane_ids:
                    continue
                s, d, _ = project_polyline(np.asarray(stop['position']), self.route)
                if s >= self._route_s - .1 and abs(d) < 3.:
                    relevant.append(s)
            if not relevant:
                continue
            s = min(relevant)
            light_records.append({'id': light['id'], 'state': light['state'], 'stop_s': s})
            if light['state'] in ('red', 'yellow'):
                stop_s = s if stop_s is None else min(stop_s, s)
        desired = float(self.cfg.get('desired_speed_mps', 10.))
        rules = {'desired_speed': desired, 'speed_limit': float(raw['speed_limit_mps']),
            'must_wait': stop_s is not None, 'stop_s': stop_s,
            'wait_reason': '当前导航车道交通灯要求停车' if stop_s is not None else None,
            'traffic_lights': light_records, 'crosswalks': self.crosswalks or [],
            'map_available': {'crosswalk': self.crosswalks is not None, 'traffic_light': True, 'stop_line': True},
            'road_priority_available': False, 'goal_s': self._goal_s}
        evidence = raw.get('current_road_evidence')
        identities = {lane.id for lane in self.lanes}
        map_complete = bool(evidence is not None and evidence.get('lane_id') == ego.lane_id and
            ego.lane_id in identities and all(evidence.get(side) is None or evidence[side] in identities for side in ('left', 'right')))
        if map_complete:
            current_lane = next(lane for lane in self.lanes if lane.id == ego.lane_id)
            current_lane.left, current_lane.right = evidence['left'], evidence['right']
        diagnostics = {'route_s': self._route_s, 'reference_anchor_s': self._route_s,
            'navigation_progress_m': self._route_s+self._progress_offset, 'stable_navigation_lane_id': self._confirmed_lane,
            'previous_control': self._previous_control.tolist(), 'reference_revision': self._reference_revision,
            'lane_map_complete': map_complete, 'visibility_wkt': observed.wkt,
            'visibility_evidence': observation_audit, 'physics_hz': self.cfg['physics_hz'],
            'historical_only_objects': {k: v for k, v in self._history.items() if k not in {a.id for a in [ego] + actors}},
            'platform': 'carla', 'carla_frame': raw['frame'], 'carla_timestamp': raw['timestamp'],
            'actual_actuation': raw.get('actual_actuation'), 'control_response': raw.get('control_response'),
            'scenario_events': raw.get('events', []), 'acceleration_source': raw['ego'].get('state_sources'),
            'scenario_template': raw.get('scenario', 'custom'),
            'environment_obstacles': raw.get('environment_obstacles', {}),
            'weather': raw.get('weather', {}), 'weather_source': raw.get('weather_source', 'unknown'),
            'platform_limitations': ['同层平面道路；不推断未提供的路权', '深度几何可见性近似，未执行真实服务器验证']}
        self._snapshot = SceneSnapshot(frame_id, timestamp, ego, actors, self.lanes, self.route,
            self.route_lane_ids, self.drivable, 'carla', rules=rules,
            assets={'camera_images': [{'view': 'multi_camera_mosaic', 'path': str(mosaic_path.resolve()),
                                      'frame_id': frame_id, 'time': timestamp}],
                    'camera_manifest_json': str(manifest_path.resolve()),
                    'original_camera_images': [c['rgb_path'] for c in raw['cameras']],
                    'depth_images': [],
                    'current_state_json': str((folder / 'current_observed_state.json').resolve()),
                    'physical_lane_centers': {l.id: l.center for l in self.lanes}, 'bev_drivable': self.drivable},
            diagnostics=diagnostics)
        return self._snapshot

    @staticmethod
    def _build_mosaic(cameras):
        order = ('front_left', 'front', 'front_right', 'back_left', 'back', 'back_right')
        indexed = {c['view']: c for c in cameras}
        if len(cameras) != 6 or set(indexed) != set(order) or len({c['frame'] for c in cameras}) != 1:
            raise ValueError('必须提供六路同帧相机')
        width, height = cameras[0]['width'], cameras[0]['height']
        mosaic = Image.new('RGB', (3 * width, 2 * height))
        draw = ImageDraw.Draw(mosaic)
        for index, name in enumerate(order):
            current = Image.open(indexed[name]['rgb_path']).convert('RGB')
            if current.size != (width, height):
                raise ValueError('相机图像大小与标定不一致')
            x, y = index % 3 * width, index // 3 * height
            mosaic.paste(current, (x, y))
            draw.rectangle((x, y, x + 140, y + 20), fill='black')
            draw.text((x + 4, y + 3), name, fill='white')
        return mosaic

    def step(self, control):
        current = self.snapshot()
        values = np.asarray(control, float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError('CARLA 控制必须是有限的[a,delta]')
        raw = self.bridge.call('step', control=values.tolist(), plan=self._planned)
        expected = current.time + self.dt
        if abs(raw['time'] - expected) > 1e-5 or raw['frame'] <= self._raw['frame']:
            raise RuntimeError('CARLA 世界步进时间或帧号不符合固定步长')
        self._previous_control = values.copy()
        self._raw, self._snapshot = raw, None
        self._step += 1
        ego = self._convert_actor(raw['ego'])
        progress, lateral, _ = project_polyline(ego.position, self.route, self._route_s, ego.heading)
        offroad = not self.drivable.buffer(float(self.config.get('mpc', {}).get('body_constraint_tolerance_m', .001))).covers(actor_shape(ego))
        collision = bool(raw['collision'])
        arrival = progress >= self._goal_s and abs(lateral) < 2.5 and not collision and not offroad
        timeout = raw['time'] >= float(self.config.get('duration', self.config.get('clock', {}).get('episode_seconds', 30))) - 1e-8
        return {'terminated': collision or arrival or timeout, 'collision': collision, 'arrival': arrival,
            'offroad': offroad, 'timeout': timeout, 'time': raw['time'], 'progress': progress+self._progress_offset,
            'reason': 'collision' if collision else 'arrival' if arrival else 'timeout' if timeout else 'running',
            'carla_frame': raw['frame'], 'control_response': raw.get('control_response'), 'events': raw.get('events', [])}

    def set_planned_trajectory(self, states):
        try:
            values = np.asarray(states, float)
            self._planned = values[:, :2].tolist() if values.ndim == 2 and values.shape[1] >= 2 and np.isfinite(values).all() else None
        except (ValueError, TypeError):
            self._planned = None

    def start_recording(self, path):
        if self._recorder is not None:
            raise RuntimeError('CARLA 录像已经启动')
        self._recorder = CameraVideoRecorder(self, path)
        return self._recorder

    def render(self):
        self.snapshot()
        if self.live:
            import cv2
            cv2.imshow('CARLA 六视角实时画面', cv2.cvtColor(self._mosaic, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return self._mosaic.copy()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self._recorder is not None:
                self._recorder.close()
        finally:
            try:
                if self.bridge is not None:
                    self.bridge.close()
            finally:
                if self.live:
                    import cv2
                    try:
                        cv2.destroyWindow('CARLA 六视角实时画面')
                    except cv2.error:
                        pass
