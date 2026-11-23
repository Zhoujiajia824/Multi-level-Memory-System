"""CARLA 0.9.15 独立 SDK 桥；兼容 Python 3.7，仅依赖标准库及 carla。"""
import base64
import copy
import json
import hashlib
import math
import os
import queue
import signal
import sys
import threading
import time
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from .carla_execution import ControlTimeline, SignedAccelerationController, duration_tick_limit
except ImportError:
    from carla_execution import ControlTimeline, SignedAccelerationController, duration_tick_limit


class CameraMetadataError(RuntimeError):
    """A sensor header cannot establish the requested image/world identity."""

    def __init__(self, key, image, frame, timestamp, reason):
        raw_time = float(image.timestamp)
        self.details = {'camera': key, 'reason': reason,
            'sensor_frame': int(image.frame),
            'sensor_timestamp': raw_time if math.isfinite(raw_time) else None,
            'world_frame': int(frame), 'world_timestamp': float(timestamp)}
        super().__init__('RGB元数据冲突：{}；{}；图像帧{}、时间{}，世界帧{}、时间{}；'
            '拒绝改写帧号或用时间戳替代帧号'.format(
                key, reason, image.frame, image.timestamp, frame, timestamp))


def camera_matches_snapshot(key, image, frame, timestamp):
    """Only genuinely older headers can be drained as stale; never relabel pixels.

    CARLA 0.9.15 PixelReader captures the frame before asynchronous GPU readback,
    but constructs the stream timestamp/transform later. Equal timestamps alone
    cannot rescue an earlier frame (PixelReader.h and AsyncDataStreamImpl.h).
    """
    stamp = float(image.timestamp)
    if not math.isfinite(stamp):
        raise CameraMetadataError(key, image, frame, timestamp, 'nonfinite_timestamp')
    if image.frame < frame:
        if stamp >= timestamp-1e-5:
            raise CameraMetadataError(key, image, frame, timestamp, 'old_frame_with_current_or_future_time')
        return False
    if image.frame > frame:
        raise CameraMetadataError(key, image, frame, timestamp, 'future_frame')
    if abs(stamp-timestamp) > 1e-5:
        raise CameraMetadataError(key, image, frame, timestamp, 'frame_timestamp_disagreement')
    return True


def validate_camera_source_clock(clock):
    """文件型模型输入只能使用本轮归档源帧，禁止高频调用时混入旧相机图。"""
    frequency = clock.get('fast_hz', 1)
    if isinstance(frequency, bool) or frequency != 1:
        raise ValueError('CARLA相机按1Hz归档，clock.fast_hz当前也必须为1；不复用旧图调用模型')


def camera_storage_due(sim_time, camera_save_hz=1, physical_ticks=None, physics_hz=50):
    """50Hz捕获、10Hz输出中选择1Hz原始图片归档，不影响显示/录像时钟。"""
    if isinstance(camera_save_hz, bool) or camera_save_hz != 1:
        raise ValueError('camera_save_hz当前必须为1，与快模型原帧采样对齐')
    if isinstance(sim_time, bool) or not isinstance(sim_time, (int, float)) or not math.isfinite(sim_time) or sim_time < -1e-5:
        raise ValueError('相机落盘需要有限非负仿真时间')
    if physical_ticks is not None:
        if type(physical_ticks) is not int or physical_ticks < 0 or type(physics_hz) is not int or physics_hz <= 0:
            raise ValueError('相机落盘需要非负整数物理帧计数及正整数物理频率')
        return physical_ticks % physics_hz == 0
    # 兼容离线旧记录；在线桥和适配器使用冻结物理帧，不累计float32时间误差。
    return abs(sim_time-round(sim_time)) <= 1e-5


def lane_key(waypoint):
    return '{}|{}|{}'.format(waypoint.road_id, waypoint.section_id, waypoint.lane_id)


def native_lane_endpoints(road_map):
    """Query real lane-section ends, which uniform samples/topology can omit."""
    document = ET.fromstring(road_map.to_opendrive())
    points = []
    for road in document.findall('road'):
        road_id = int(road.get('id'))
        length = float(road.get('length'))
        sections = road.findall('lanes/laneSection')
        for index, section in enumerate(sections):
            start = float(section.get('s'))
            end = float(sections[index+1].get('s')) if index+1 < len(sections) else length
            if end <= start:
                continue
            for lane in section.findall('left/lane') + section.findall('right/lane'):
                if lane.get('type') != 'driving':
                    continue
                identity = int(lane.get('id'))
                for value, direction in ((start, 1.), (end, -1.)):
                    # Prefer the exact endpoint. At a section boundary CARLA may
                    # resolve the other section, or reject s==road.length.
                    candidates = [value] + [value + direction*min(epsilon, (end-start)/4.)
                                             for epsilon in (1e-6, 1e-4, .001)]
                    for candidate in candidates:
                        wp = road_map.get_waypoint_xodr(road_id, identity, candidate)
                        if (wp is not None and wp.section_id == index
                                and str(wp.lane_type) == 'Driving'):
                            points.append(wp)
                            break
                    else:
                        raise ValueError('公开地图缺少真实车道端点：{}|{}|{}'.format(road_id,index,identity))
    return points


def boundary_semantics(marking_type, waypoint_permission, marking_permission, is_junction):
    """当前位置的实物标线与路口虚拟连接边分离；不拿无标线当实线。

    双线的通行方向由CARLA当前waypoint/marking许可共同提供，不猜测名称顺序。
    未知地图属性保守保留，但明确标记为未知而非伪造实线。
    """
    name = str(marking_type).split('.')[-1].lower()
    virtual = bool(is_junction) and name in ('none', 'nonetype')
    known = name in ('none', 'nonetype', 'solid', 'solidsolid', 'solidbroken',
                     'brokensolid', 'broken', 'brokenbroken', 'bottsdots', 'grass', 'curb')
    physical = name not in ('none', 'nonetype')
    crossable = bool(waypoint_permission and marking_permission)
    if name in ('solid', 'solidsolid', 'curb', 'grass'):
        crossable = False
    if not known:
        crossable = False
    if virtual:
        crossable = True  # 仅移除虚拟连接边的实体惩罚；道路多边形仍独立约束。
    return {'crossable': crossable, 'kind': 'virtual_lane_boundary' if virtual else 'lane_boundary',
            'physical_marking': physical, 'map_evidence_known': known,
            'permission_source': 'junction_unmarked_virtual' if virtual else 'current_waypoint_and_marking'}


def map_identity(road_map):
    """地图名称和原始OpenDRIVE指纹仅供复现审计，不进入检索分数。"""
    document = road_map.to_opendrive()
    if not isinstance(document,str) or not document.strip():
        raise ValueError('公开地图OpenDRIVE缺失，无法核查同地点实验来源')
    return {'map_name':str(road_map.name),
            'opendrive_sha256':hashlib.sha256(document.encode('utf-8')).hexdigest()}


def inspect_scene(config, seed):
    """只读场景选址：不换地图、不设置同步、不创建或销毁演员、不推进时钟。"""
    import carla
    try:
        from .carla_experiment_routes import inspect_instance
    except ImportError:
        from carla_experiment_routes import inspect_instance
    client = carla.Client(config.get('host','127.0.0.1'),int(config.get('port',2000)))
    client.set_timeout(float(config.get('timeout_s',30.)))
    if (client.get_client_version().split('-')[0] != '0.9.15' or
            client.get_server_version().split('-')[0] != '0.9.15'):
        raise ValueError('选址检查仅支持CARLA 0.9.15')
    world = client.get_world()
    road_map = world.get_map()
    requested = str(config.get('map','Town05')).rstrip('/').split('/')[-1]
    if road_map.name.rstrip('/').split('/')[-1] != requested:
        raise ValueError('只读检查不切换地图：当前{}，场景要求{}'.format(road_map.name,requested))
    settings = dict(config.get('experiment',{})); settings['seed'] = int(seed)
    _, _, report = inspect_instance(road_map,settings,carla)
    report.update(map_identity(road_map))
    report['read_only'] = True
    return report


def xy(location):
    return [float(location.x), -float(location.y)]


def bounding_box_world_center(transform, box, carla):
    """CARLA Transform.transform会原地修改参数，必须复制框的actor局部中心。

    保留box.location供后续轴距参考点、可见性及重复状态读取使用；不能将其变成世界坐标。
    """
    local = box.location
    point = carla.Location(x=local.x, y=local.y, z=local.z)
    return transform.transform(point)


def wheel_geometry(wheels, inverse_transform, box_center_x):
    """CARLA 轮位为世界厘米；先转米再转车体局部，不能直接拿世界x作轴距。"""
    if len(wheels) != 4:
        raise ValueError('首批 CARLA 自车仅支持四轮汽车')
    local_x = []
    for wheel in wheels:
        position = wheel.position
        world = [position.x / 100., position.y / 100., position.z / 100., 1.]
        local_x.append(sum(inverse_transform[0][j] * world[j] for j in range(4)))
    front, rear = sum(local_x[:2]) / 2., sum(local_x[2:]) / 2.
    wheelbase, rear_distance = front - rear, box_center_x - rear
    if (not all(math.isfinite(v) for v in local_x+[box_center_x,wheelbase,rear_distance])
            or not wheelbase > 0 or not 0 < rear_distance < wheelbase):
        raise ValueError('无法从真实轮位提取有效轴距/参考点：'
            'front_x_m={}，rear_x_m={}，bbox_actor_local_x_m={}，wheelbase_m={}，'
            'rear_axle_to_center_m={}；轮位按世界厘米转换，框中心须为actor局部米；不使用车长代替'.format(
                front,rear,box_center_x,wheelbase,rear_distance))
    return wheelbase, rear_distance


class RouteElevationSupport:
    """按公开导航的最近平面路段插值高度，不将坡道固定在起点高度。

    只负责地图提取。导航车道带自身跨层重叠时，当前二维核心无法分层，明确拒绝。
    查询距离相同但层高不同也拒绝，不能按待查询车道的高度自行挑选另一层。
    """
    VERSION = 'carla-route-elevation-1.0'
    CELL_M = 20.

    def __init__(self, waypoints, tolerance_m, radius_m=120.):
        if (not math.isfinite(tolerance_m) or tolerance_m <= 0
                or not math.isfinite(radius_m) or radius_m <= 0):
            raise ValueError('导航高度筛选范围必须为有限正数')
        self.tolerance_m, self.radius_m = tolerance_m, radius_m
        self.segments, self.cells = [], {}
        arc = 0.
        for first, second in zip(waypoints, waypoints[1:]):
            a, b = first.transform.location, second.transform.location
            values = tuple(float(v) for v in (a.x,a.y,a.z,b.x,b.y,b.z))
            if not all(math.isfinite(v) for v in values):
                raise ValueError('导航高度参考包含非有限坐标')
            dx, dy = values[3]-values[0], values[4]-values[1]
            length = math.hypot(dx,dy)
            if length < 1e-6:
                if abs(values[5]-values[2]) > 1e-6:
                    raise ValueError('导航存在平面零长度但高度跳变的路段')
                continue
            width = max(float(first.lane_width),float(second.lane_width))
            if not math.isfinite(width) or width <= 0:
                raise ValueError('导航车道宽度必须有限且为正')
            segment = (values,dx,dy,length*length,width,arc)
            index = len(self.segments)
            self.segments.append(segment)
            arc += length
            for ix in range(math.floor(min(a.x,b.x)/self.CELL_M),math.floor(max(a.x,b.x)/self.CELL_M)+1):
                for iy in range(math.floor(min(a.y,b.y)/self.CELL_M),math.floor(max(a.y,b.y)/self.CELL_M)+1):
                    self.cells.setdefault((ix,iy),[]).append(index)
        if not self.segments:
            raise ValueError('导航高度参考至少需要两个不同平面位置')
        self._check_level_overlap()

    @staticmethod
    def _project(segment, x, y):
        values,dx,dy,norm2,_,_ = segment
        alpha = max(0.,min(1.,((x-values[0])*dx+(y-values[1])*dy)/norm2))
        px,py = values[0]+alpha*dx, values[1]+alpha*dy
        return math.hypot(x-px,y-py), values[2]+alpha*(values[5]-values[2]), alpha

    def _check_level_overlap(self):
        # 非相邻路段的最近点：交点或四个端点的正交投影。只核查导航车道带，
        # 不禁止普通同层路口，也不把相距很远的山坡误认成重叠道路。
        checked = set()
        for index, segment in enumerate(self.segments):
            a,dx,dy,_,width,arc = segment
            pad = width
            candidates = set()
            for ix in range(math.floor((min(a[0],a[3])-pad)/self.CELL_M),math.floor((max(a[0],a[3])+pad)/self.CELL_M)+1):
                for iy in range(math.floor((min(a[1],a[4])-pad)/self.CELL_M),math.floor((max(a[1],a[4])+pad)/self.CELL_M)+1):
                    candidates.update(self.cells.get((ix,iy),()))
            for other_index in candidates:
                if other_index <= index+1 or (index,other_index) in checked:
                    continue
                checked.add((index,other_index))
                other = self.segments[other_index]
                b,ex,ey,_,other_width,other_arc = other
                # 连续坡道在局部车道宽度内无需当成两层道路。
                if other_arc-arc < max(10.,width+other_width):
                    continue
                tests = []
                for x,y,z in ((a[0],a[1],a[2]),(a[3],a[4],a[5])):
                    distance,height,_ = self._project(other,x,y)
                    tests.append((distance,abs(z-height)))
                for x,y,z in ((b[0],b[1],b[2]),(b[3],b[4],b[5])):
                    distance,height,_ = self._project(segment,x,y)
                    tests.append((distance,abs(z-height)))
                cross = dx*ey-dy*ex
                if abs(cross)>1e-12:
                    rx,ry = b[0]-a[0],b[1]-a[1]
                    ta,tb = (rx*ey-ry*ex)/cross,(rx*dy-ry*dx)/cross
                    if 0<=ta<=1 and 0<=tb<=1:
                        tests.append((0.,abs(a[2]+ta*(a[5]-a[2])-b[2]-tb*(b[5]-b[2]))))
                distance,height_difference = min(tests)
                if distance < .5*(width+other_width) and height_difference>self.tolerance_m:
                    raise ValueError('导航自身存在跨层平面重叠，当前二维规划地图不支持；请缩短路线或更换公开路线锚点')

    def height(self, x, y):
        # 仅在120m地图走廊内查找；空间桶减少大地图上的全部逐段扫描。
        radius, cell = self.radius_m, self.CELL_M
        buckets=[]
        for ix in range(math.floor((x-radius)/cell),math.floor((x+radius)/cell)+1):
            for iy in range(math.floor((y-radius)/cell),math.floor((y+radius)/cell)+1):
                indices=self.cells.get((ix,iy))
                if indices:
                    lower=math.hypot(max(ix*cell-x,0.,x-(ix+1)*cell),max(iy*cell-y,0.,y-(iy+1)*cell))
                    if lower <= radius:buckets.append((lower,ix,iy,indices))
        best, height, visited = radius+1e-9, None, set()
        for lower,_,_,indices in sorted(buckets):
            if lower>best+1e-6:break
            for index in indices:
                if index in visited:continue
                visited.add(index)
                distance,value,_=self._project(self.segments[index],x,y)
                if distance<best-1e-6 or (height is None and distance<=radius):
                    best,height=distance,value
                elif abs(distance-best)<=1e-6 and height is not None and abs(value-height)>self.tolerance_m:
                    raise ValueError('地图点到导航的最近投影具有跨层歧义，不能静默选取另一层道路')
        return height if best <= radius and height is not None else None

    def contains(self, location):
        height=self.height(location.x,location.y)
        return height is not None and abs(location.z-height)<=self.tolerance_m


def closed_crosswalks(locations, base_z, tolerance, height_support=None):
    """官方地图以重复首点分隔人行横道闭合多边形；不把未闭合片段补齐。"""
    polygons, pending = [], []
    for location in locations:
        pending.append(location)
        if len(pending) >= 4 and math.sqrt((location.x-pending[0].x)**2+
                (location.y-pending[0].y)**2+(location.z-pending[0].z)**2) < 1e-4:
            if all((height_support.contains(point) if height_support is not None
                    else abs(point.z-base_z) <= tolerance) for point in pending):
                polygons.append([xy(point) for point in pending])
            pending = []
    if pending:
        raise ValueError('地图人行横道包含未闭合多边形，不能声称完整地图证据')
    return polygons


OBJECT_CATEGORY_VERSION = 'carla-road-participant-categories-1.0'
ENVIRONMENT_PARTICIPANT_KINDS = {
    'Car': 'vehicle', 'Truck': 'vehicle', 'Bus': 'vehicle', 'Train': 'vehicle',
    'Motorcycle': 'vehicle', 'Vehicles': 'vehicle', 'Bicycle': 'bicycle',
    'Pedestrians': 'pedestrian', 'Pedestrian': 'pedestrian',
}
# Exact SDK blueprint IDs from CARLA 0.9.15 catalogue_props/ (Road category).
# Road signs, permanent barriers/fences, furniture and arbitrary Static/Other
# meshes are not obstacles. No substring guessing from a map component name.
ROAD_OBSTACLE_BLUEPRINTS = frozenset('static.prop.' + name for name in (
    'streetbarrier', 'constructioncone', 'trafficcone01', 'trafficcone02',
    'ironplank', 'brokentile01', 'brokentile02', 'brokentile03', 'brokentile04',
    'dirtdebris01', 'dirtdebris02', 'dirtdebris03'))
BICYCLE_BLUEPRINTS = frozenset(('vehicle.bh.crossbike', 'vehicle.diamondback.century',
                              'vehicle.gazelle.omafiets'))


def environment_category(label, name):
    """Category evidence only; lane-chain/range admission happens downstream."""
    if label in ENVIRONMENT_PARTICIPANT_KINDS:
        return ENVIRONMENT_PARTICIPANT_KINDS[label], 'semantic_label:' + label
    if label in ('Static', 'Other') and name in ROAD_OBSTACLE_BLUEPRINTS:
        return 'static_obstacle', 'exact_road_obstacle_blueprint_name:' + name
    return None, ('unresolved_static_or_other' if label in ('Static', 'Other')
                  else 'not_a_road_participant_or_obstacle')


def actor_category(actor):
    """Do not turn arbitrary experiment props into road obstacles."""
    identity = actor.type_id
    if identity in BICYCLE_BLUEPRINTS:
        return 'bicycle', 'sdk_bicycle_blueprint:' + identity
    if identity.startswith('vehicle.'):
        return 'vehicle', 'sdk_vehicle_blueprint:' + identity
    if identity.startswith('walker.pedestrian.'):
        return 'pedestrian', 'sdk_pedestrian_blueprint:' + identity
    if identity in ROAD_OBSTACLE_BLUEPRINTS:
        return 'static_obstacle', 'exact_road_obstacle_blueprint:' + identity
    return None, 'unrecognized_road_obstacle_blueprint:' + identity


def environment_catalogue(world, carla):
    """只读取环境网格，不启停对象。0.9.15官方ObjectLabel.h区分Car/Truck等。

    get_environment_objects的框已经位于世界坐标，不能再乘EnvironmentObject.transform。
    在创建实验车辆之前读取，配合独立空世界准入避免把动态actor重复成静态物。
    只收有类别证据的交通对象和路面障碍候选；车道内与前后范围由公共筛选器判断。
    围栏、护栏、墙、杆和未识别Static/Other不因有包围框就变成障碍物。
    """
    names = tuple(ENVIRONMENT_PARTICIPANT_KINDS) + ('Static', 'Other')
    labels = {getattr(carla.CityObjectLabel, name): name for name in names
              if hasattr(carla.CityObjectLabel, name)}
    objects, seen, excluded, normalizations = [], {}, {}, []
    category_audit = []
    for item in sorted(world.get_environment_objects(carla.CityObjectLabel.Any), key=lambda obj: obj.id):
        label = labels.get(item.type, str(item.type))
        name = getattr(item, 'name', '')
        kind, evidence = environment_category(label, name)
        category_audit.append({'id': str(item.id), 'name': name, 'label': label,
                              'kind': kind, 'included': kind is not None, 'evidence': evidence})
        if kind is None:
            excluded[label] = excluded.get(label, 0) + 1
            continue
        box = item.bounding_box
        values = (box.location.x, box.location.y, box.location.z, box.extent.x, box.extent.y,
                  box.extent.z, box.rotation.yaw, box.rotation.pitch, box.rotation.roll)
        if not all(math.isfinite(value) for value in values) or any(value == 0 for value in values[3:6]):
            raise ValueError('环境障碍物框包含非有限值或零半尺寸：id={}，name={!r}，label={}，'
                'center={}，extent={}，rotation={}'.format(item.id, getattr(item, 'name', ''),
                    label, values[:3], values[3:6], values[6:]))
        if any(value < 0 for value in values[3:6]):
            # 0.9.15直接将半尺寸乘组件缩放，镜像网格会保留负号。
            # ±半尺寸枚举的八角点集合不变；不删除障碍物，不修改原始地图框。
            extent = tuple(abs(value) for value in values[3:6])
            positive_box = carla.BoundingBox(
                carla.Location(x=values[0], y=values[1], z=values[2]),
                carla.Vector3D(x=extent[0], y=extent[1], z=extent[2]))
            positive_box.rotation = carla.Rotation(yaw=values[6], pitch=values[7], roll=values[8])
            normalizations.append({'id': str(item.id), 'name': getattr(item, 'name', ''),
                'label': label, 'source_extent_m': list(values[3:6]),
                'canonical_extent_m': list(extent)})
            box = positive_box
            values = values[:3] + extent + values[6:]
        # 完全相同的环境几何可能有多个组件标签，仅合并精确几何重复，不近似删物体。
        signature = (kind,) + tuple(values)
        if signature in seen:
            seen[signature]['source_ids'].append(str(item.id))
            continue
        entry = {'id': 'environment_{}'.format(item.id), 'box': box, 'label': label,
                 'kind': kind, 'classification_evidence': evidence, 'source_ids': [str(item.id)]}
        seen[signature] = entry
        objects.append(entry)
    return objects, {'source': 'CARLA_0.9.15_get_environment_objects_world_bounding_boxes',
        'included_labels': sorted(set(entry['label'] for entry in objects)),
        'excluded_label_counts': excluded,
        'category_version': OBJECT_CATEGORY_VERSION, 'category_decisions': category_audit,
        'category_counts': {kind: sum(entry['kind'] == kind for entry in objects)
                            for kind in ('vehicle', 'bicycle', 'pedestrian', 'static_obstacle')},
        'road_obstacle_blueprints': sorted(ROAD_OBSTACLE_BLUEPRINTS),
        'catalogue_count': len(objects), 'loaded_before_experiment_actors': True,
        'geometry_version': 'carla-environment-box-1.1',
        'signed_extent_normalized_count': len(normalizations),
        'signed_extent_normalizations': normalizations,
        'scope': '类别准入候选，不代表车道范围准入；泛Static/Other及围栏护栏墙杆不入对象表，实体与RGB不修改'}


class CarlaSession:
    def __init__(self):
        self.world = None
        self.saved_settings = None
        self._settings_change_attempted = False
        self.world_ownership = None
        self.cleanup_report = None
        self.experiment = None
        self.sensors = []
        self.queues = {}
        self.frames = {}
        self.events = []
        self.collision = False
        self.t0 = None
        self.frame = None
        self.closed = False
        self.angular_history = {}
        self.cleanup_errors = []
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._ticker = None
        self._latest = None
        self._worker_error = None
        self._terminal_reason = None
        self._terminal_published = False
        self._paused = True
        self._sim_time = 0.
        self._latest_published_at = None
        self._started_at = None
        self._initialized_at = None
        self._history = {}
        self._history_sample_index = -1
        self._entities_cache = None
        self._last_applied_control = None
        self._last_command = None
        self._execution_controller = None
        self._last_control_response = None
        self._reference_transition = None
        self._applied_key = None
        self._pending_plan = None
        self._visualization = None
        self._camera_payload_cache = None
        self._camera_audit_lock = threading.Lock()
        self._camera_ready = threading.Event()
        self._camera_audit = {}
        self.camera_sync_failure = None
        self.world_reset = None
        self._initialization_warmup = None
        self._initial_speed_audit = None
        self._initial_source_frame = None
        self._initialization_ticks = []
        self._rgb_initialization = None
        self._rgb_initialization_deadline = None
        self._display_frames = []
        self._published_sources = {}
        self.evaluation_probe = None
        self._evaluation = None
        self._timing = dict((name, 0.) for name in
            ('tick_s', 'sensor_wait_s', 'snapshot_state_s', 'image_s', 'pacing_s', 'scenario_s'))
        self._runtime_metrics = {'physical_ticks': 0, 'wall_pacing_wait_s': 0.,
                                 'wall_lag_max_s': 0., 'wall_deadline_misses': 0}

    def initialize(self, config, directory, seed):
        if config.get('exclusive_world') is not True:
            raise ValueError('必须显式确认 exclusive_world=true，仅使用本任务独立 CARLA 世界')
        if type(config.get('world_plan_debug',False)) is not bool:
            raise ValueError('world_plan_debug必须为布尔值')
        if type(config.get('allow_world_reset', False)) is not bool:
            raise ValueError('allow_world_reset必须为布尔值')
        camera_storage_due(0., config.get('camera_save_hz', 1))
        self.runtime_mode = config.get('runtime_mode', 'sync')
        if self.runtime_mode not in ('sync', 'async'):
            raise ValueError('runtime_mode必须为sync或async')
        if type(config.get('rgb_enabled', False)) is not bool:
            raise ValueError('rgb_enabled必须为布尔值')
        self.rgb_enabled = config.get('rgb_enabled', False)
        if isinstance(config.get('camera_hz', 10), bool) or config.get('camera_hz', 10) != 10:
            raise ValueError('RGB消费/录像频率camera_hz当前必须为10，原始图片归档独立为1Hz')
        if type(config.get('camera_capture_hz', 50)) not in (int, float) or config.get('camera_capture_hz', 50) != 50:
            raise ValueError('RGB真实采集频率camera_capture_hz必须为50，与物理帧一致')
        self._display_enabled = bool(set(config.get('consumers', [])) & {'video', 'live'})
        self.display_capacity = config.get('display_queue_frames', 50)
        if type(self.display_capacity) is not int or self.display_capacity < 10:
            raise ValueError('display_queue_frames必须为至少10的整数')
        if type(config.get('rgb_postprocess_effects', True)) is not bool:
            raise ValueError('rgb_postprocess_effects必须为布尔值')
        for key, default in (('timeout_s', 30), ('map_resolution_m', 2), ('same_level_tolerance_m', 1.5),
                             ('perception_radius_m', 120),
                             ('camera_height_m', 2.2), ('map_speed_cap_mps', 30)):
            value = config.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError('CARLA 参数必须为有限正数：' + key)
        import carla
        try:
            from .carla_experiments import create_experiment
            from .carla_initialization import hold_initial_motion, release_initial_motion
            from .carla_evaluation import CarlaEvaluationProbe
        except ImportError:
            from carla_experiments import create_experiment
            from carla_initialization import hold_initial_motion, release_initial_motion
            from carla_evaluation import CarlaEvaluationProbe
        self.carla = carla
        self.cfg = config
        if config.get('world_plan_debug',False):
            print('已开启CARLA世界规划线：RGB相机可能拍到调试线；严格无叠加采集请使用--no-world-plan。',
                  file=sys.stderr,flush=True)
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout = float(config.get('timeout_s', 30.0))
        client = carla.Client(config.get('host', '127.0.0.1'), int(config.get('port', 2000)))
        client.set_timeout(self.timeout)
        client_version, server_version = client.get_client_version(), client.get_server_version()
        if client_version.split('-')[0] != '0.9.15' or server_version.split('-')[0] != '0.9.15':
            raise ValueError('仅支持 CARLA 0.9.15；客户端={}，服务端={}'.format(client_version, server_version))
        self.world = client.get_world()
        initial = self.world.get_settings()
        self._check_world_ownership(initial)
        requested_map = str(config.get('map', 'Town05'))
        if type(config.get('physics_hz', 50)) is not int or type(config.get('planner_hz', 10)) is not int:
            raise ValueError('CARLA物理与规划频率必须为整数')
        self.physics_hz = config.get('physics_hz', 50)
        self.planner_hz = config.get('planner_hz', 10)
        if self.physics_hz <= 0 or self.planner_hz <= 0 or self.physics_hz % self.planner_hz:
            raise ValueError('CARLA 物理频率必须为规划频率的整数倍')
        if self.planner_hz not in (1, 2, 10):
            raise ValueError('CARLA planner_hz只支持1、2、10')
        if type(config.get('control_step_s', .1)) not in (int, float):
            raise ValueError('MPC控制网格必须为有限数字')
        self.control_step_s = float(config.get('control_step_s', .1))
        if (not math.isfinite(self.control_step_s) or self.control_step_s <= 0 or
                abs(self.control_step_s*self.physics_hz-round(self.control_step_s*self.physics_hz)) > 1e-8 or
                abs(1./self.planner_hz/self.control_step_s-round(1./self.planner_hz/self.control_step_s)) > 1e-8):
            raise ValueError('MPC控制网格必须与物理及规划时钟整除对齐')
        if self.physics_hz != 50 or abs(self.control_step_s-.1) > 1e-8:
            raise ValueError('CARLA保留50Hz物理与0.1秒MPC控制网格')
        self.duration_s = config.get('duration_s')
        if self.duration_s is not None and (isinstance(self.duration_s, bool) or
                not isinstance(self.duration_s, (int, float)) or
                not math.isfinite(self.duration_s) or self.duration_s <= 0):
            raise ValueError('duration_s必须为有限正数')
        self.timeline = ControlTimeline(config.get('acceleration_min_mps2', -6),
            config.get('acceleration_max_mps2', 3), config.get('steering_abs_max_rad', .6), self.physics_hz)
        self._configure_experiment_world(client, requested_map, config.get('allow_world_reset', False))
        self.environment_objects, self.environment_audit = environment_catalogue(self.world, carla)
        audit_path = self.root / 'carla_environment_audit.json'
        audit_path.write_text(json.dumps(self.environment_audit, ensure_ascii=False,
                                        allow_nan=False, indent=2), encoding='utf-8')
        # 全地图类别/规范化明细只保存一次，避免每帧重复写入全部环境物体。
        self.environment_audit = {key: value for key, value in self.environment_audit.items()
                                  if key not in ('signed_extent_normalizations', 'category_decisions')}
        self.environment_audit['audit_path'] = str(audit_path)
        if self.environment_audit['signed_extent_normalized_count']:
            print('已对{}条镜像环境包围框作等价半尺寸规范化，未删除障碍物；明细：{}'.format(
                self.environment_audit['signed_extent_normalized_count'], audit_path),
                file=sys.stderr, flush=True)
        experiment = dict(config.get('experiment', {}))
        experiment['seed'] = int(seed)
        selection_path = self.root / 'carla_scene_selection.json'
        try:
            self.experiment = create_experiment(self.world, client, experiment, carla)
        except Exception as exc:
            if hasattr(exc,'report'):
                selection_path.write_text(json.dumps(exc.report,ensure_ascii=False,
                    allow_nan=False,indent=2),encoding='utf-8')
            raise
        (self.root/'resolved_instance.json').write_text(json.dumps(self.experiment.manifest,
            ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
        # 新spawn演员须先注册到世界，get_speed_limit才能读取真实地图限速。
        # 此时尚未创建RGB；仅属初始化，不计入本轮控制或相机采集时间。
        warmup = self._initialization_tick('actor_registration')
        self._initialization_warmup = {'frame': warmup.frame,
            'timestamp': warmup.timestamp.elapsed_seconds,
            'purpose': '注册静止自车并读取地图限速，尚未创建RGB传感器'}
        # get_control读取最近ActorSnapshot；注册后再保存parked等演员的原手刹。
        hold_initial_motion(self.experiment, carla)
        selection = getattr(self.experiment,'selection_report',None)
        if selection is not None:
            selection_path.write_text(json.dumps(selection,ensure_ascii=False,
                allow_nan=False,indent=2),encoding='utf-8')
            print('场景选址完成：起点{}，局部交互段{}m，完整导航{}m；诊断：{}'.format(
                selection['selected_spawn_index'],selection['interaction_length_m'],
                selection['route_length_m'],selection_path),file=sys.stderr,flush=True)
        self.ego = self.experiment.ego
        maximum_steering = max(float(wheel.max_steer_angle)
            for wheel in self.ego.get_physics_control().wheels)
        self._execution_controller = SignedAccelerationController(math.radians(maximum_steering),
            dt=1./self.physics_hz, acceleration_min=self.timeline.minimum,
            acceleration_max=self.timeline.maximum,
            steering_rate=float(config.get('steering_rate_abs_max_radps', .8)),
            settings=config.get('longitudinal_control'))
        self.route_wps = self.experiment.route_waypoints
        if len(self.route_wps) < 2:
            raise ValueError('场景未提供有效自车导航路线')
        self.map = self.world.get_map()
        self.width = int(config.get('camera_width', 640))
        self.height = int(config.get('camera_height', 360))
        self.fov = float(config.get('camera_fov', 100.0))
        if self.width < 64 or self.height < 64 or not 20 < self.fov < 150:
            raise ValueError('相机尺寸或视场配置非法')
        self._create_sensors()
        if self.rgb_enabled:
            # 固定初始化制动保持，每帧等待完整六路RGB再推进；不寻帧、不追图。
            self._align_initial_camera_phase()
        self._initial_speed_audit = release_initial_motion(self.experiment, carla)
        snap = self._initialization_tick('initial_motion_activation')
        self._collect(self.frame)
        self._initial_source_frame = self.frame
        self.t0 = snap.timestamp.elapsed_seconds
        self.experiment.after_tick(self.frame, 0, snap)
        self.evaluation_probe = CarlaEvaluationProbe(self.world, carla, self.ego, self.route_wps,
            settings=experiment, directory=self.root, physics_hz=self.physics_hz)
        self._evaluation = self.evaluation_probe.observe(snap, 0, events=list(self.events),
            scenario_events=self.experiment.evaluation_state(), acceleration_sample_valid=False,
            acceleration_sample_reason='initial_motion_activation_set_target_velocity')
        if self._rgb_initialization is not None:
            self._rgb_initialization.update(source_frame=self.frame, source_timestamp=self.t0,
                activation_frame_gap=self.frame-self._rgb_initialization['stable_frame'])
        if self.collision:
            raise RuntimeError('初始化期间发生碰撞，保留初始化事件并拒绝作为正常实验起点')
        self._sample_history(force=True)
        result = self.snapshot()
        self._queue_display_frame(result.get('cameras', []))
        self._publish(result)
        result = self._latest_result()
        result['map'] = self._map_payload()
        result['bridge_version'] = 'carla-0.9.15-range-runtime-3.3'
        result['sdk_version'] = client_version
        result['server_version'] = server_version
        result['world_reset'] = copy.deepcopy(self.world_reset)
        result['initialization_warmup'] = copy.deepcopy(self._initialization_warmup)
        result['initial_speed_audit'] = copy.deepcopy(self._initial_speed_audit)
        result['rgb_initialization'] = copy.deepcopy(self._rgb_initialization)
        initialization = {'bridge_version': result['bridge_version'],
            'client_version': client_version, 'server_version': server_version,
            'map_identity': map_identity(self.map),
            'instance_hash': experiment.get('instance_hash'),
            'actors_at_source': [dict(self._actor(actor, actor.id == self.ego.id),
                carla_actor_id=str(actor.id),
                role=next((role for role, identity in self.experiment.evaluation_state()['actor_roles'].items()
                           if str(actor.id) == str(identity)), 'unassigned')) for actor in self.experiment.actors],
            'world_reset': result['world_reset'], 'initialization_warmup': result['initialization_warmup'],
            'initial_speed_audit': result['initial_speed_audit'], 'source_frame': self.frame, 't0': self.t0,
            'physics_hz': self.physics_hz, 'planner_hz': self.planner_hz,
            'rgb_enabled': self.rgb_enabled, 'camera_hz': 10.,
            'camera_capture_hz': float(self.physics_hz) if self.rgb_enabled else 0.,
            'sensor_tick_s': 0., 'camera_barrier': 'six_views_before_next_physical_tick',
            'rgb_initialization': copy.deepcopy(self._rgb_initialization),
            'initialization_ticks': copy.deepcopy(self._initialization_ticks),
            'initial_motion': copy.deepcopy(self.experiment.initial_motion_audit),
            'initialization_events': copy.deepcopy(self.events),
            'traffic_lights_at_source': [
                {'id': str(actor.id), 'state': str(actor.get_state()),
                 'opendrive_id': str(actor.get_opendrive_id()) if hasattr(actor, 'get_opendrive_id') else None,
                 'pose': [actor.get_transform().location.x, actor.get_transform().location.y,
                          actor.get_transform().location.z, actor.get_transform().rotation.yaw],
                 'elapsed_s': float(actor.get_elapsed_time())}
                for actor in self.world.get_actors().filter('traffic.traffic_light*')],
            'note': (('先固定制动预热0.5秒，每物理帧收齐六路RGB，再激活初速并建立t0；'
                      '50Hz采集、10Hz输出和录像、1Hz原图归档；') if self.rgb_enabled else
                     '无图像消费者，不创建RGB相机或执行相机预热；注册后激活初速并建立t0；')+
                    '初始化推进与事件单列，灯态由冻结实例的整数帧程序控制。'}
        (self.root/'carla_initialization.json').write_text(json.dumps(initialization, ensure_ascii=False,
            allow_nan=False, indent=2), encoding='utf-8')
        self._initialized_at = time.monotonic()
        return result

    def _initialization_tick(self, stage):
        """单独记账的初始化推进；不运行控制时间线或触发场景交互。"""
        previous = self.world.get_snapshot().frame
        if self.frame is not None and previous != self.frame:
            raise RuntimeError('初始化阶段检测到其他客户端推进世界')
        timeout = self.timeout
        if self._rgb_initialization_deadline is not None:
            timeout = min(timeout, self._rgb_initialization_deadline-time.monotonic())
            if timeout <= 0:
                raise TimeoutError('RGB初始化相位对齐超过总墙钟预算')
        started = time.monotonic()
        self.frame = self.world.tick(timeout)
        self._add_timing('tick_s', time.monotonic()-started)
        snap = self.world.get_snapshot()
        if self.frame != previous+1 or snap.frame != self.frame:
            raise RuntimeError('初始化单次tick帧不连续；只能有一个tick客户端')
        self._initialization_ticks.append({'stage': stage, 'frame': self.frame,
            'timestamp': float(snap.timestamp.elapsed_seconds)})
        if self.collision:
            raise RuntimeError('初始化期间发生碰撞，拒绝继续；事件保留在诊断文件中')
        return snap

    def _save_initialization_failure(self, error):
        root = getattr(self, 'root', None)
        if root is None:
            return
        report = {'error': type(error).__name__+': '+str(error), 'frame': self.frame,
            'world_ownership': self.world_ownership,
            'rgb_initialization': self._rgb_initialization,
            'initialization_ticks': self._initialization_ticks,
            'initial_motion': getattr(self.experiment, 'initial_motion_audit', None),
            'sensor_events': self.events,
            'scenario_events': getattr(self.experiment, 'events', [])}
        try:
            path = Path(root)/'carla_initialization_failure.json'
            path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2),
                            encoding='utf-8')
        except Exception as exc:
            print('初始化失败诊断保存异常：'+str(exc), file=sys.stderr, flush=True)

    @staticmethod
    def _settings_record(settings):
        fields = ('synchronous_mode', 'fixed_delta_seconds', 'no_rendering_mode',
                  'substepping', 'max_substep_delta_time', 'max_substeps')
        return {key: getattr(settings, key) for key in fields}

    @staticmethod
    def _actor_record(actor):
        return {'id': int(actor.id), 'type_id': actor.type_id,
                'role_name': actor.attributes.get('role_name', '')}

    def _check_world_ownership(self, settings):
        try:
            from .carla_world_ownership import inspect_world_actors
        except ImportError:
            from carla_world_ownership import inspect_world_actors
        classification = inspect_world_actors(self.world.get_actors())
        existing = classification['blocking_actors']
        reasons = []
        if settings.synchronous_mode:
            reasons.append('world_already_synchronous')
        if existing:
            reasons.append('existing_experiment_actor_types')
        self.world_ownership = {'version': 'carla-world-ownership-1.1',
            'map': self.world.get_map().name, 'frame': int(self.world.get_snapshot().frame),
            'settings': self._settings_record(settings), 'rejection_reasons': reasons,
            'accepted': not reasons, 'policy': '不接管同步世界，不删除既有演员；'
                '仅按0.9.15原生注册属性识别地图网格，未知身份仍拒绝'}
        self.world_ownership.update(classification)
        path = self.root / 'carla_world_ownership.json'
        path.write_text(json.dumps(self.world_ownership, ensure_ascii=False, allow_nan=False, indent=2),
                        encoding='utf-8')
        if reasons:
            raise RuntimeError('拒绝接管正在同步运行或已有车辆/行人/脚本道具的世界；'
                '原因={}，阻塞演员数={}；详细身份见{}'.format(','.join(reasons), len(existing), path))

    def _align_initial_camera_phase(self):
        """固定制动预热；每物理帧收齐六路，绝不越过缺图帧寻图。

        保留约0.5秒初始化保持以免缩短原有落地/制动时间；这不是通用
        接地证明，布局漂移和碰撞检查仍由初始化运动审计负责。
        """
        count = max(3, self.physics_hz // 2)
        audit = {'version': 'carla-rgb-startup-every-tick-2.0', 'status': 'calibrating',
            'expected_period_frames': 1, 'warmup_frame_count': count,
            'capture_hz': self.physics_hz, 'output_hz': 10., 'sensor_tick_s': 0.,
            'warmup_simulated_seconds': count/float(self.physics_hz),
            'barrier': 'six_views_before_next_physical_tick',
            'sample_frames': [], 'intervals_frames': [], 'initialization_only': True}
        self._rgb_initialization = audit
        self._rgb_initialization_deadline = time.monotonic()+self.timeout
        try:
            previous = None
            for index in range(count):
                probe = self._initialization_tick('camera_bootstrap' if index == 0 else 'camera_phase_probe')
                self._collect(self.frame)
                if previous is not None:
                    gap = self.frame-previous
                    audit['intervals_frames'].append(gap)
                    if gap != 1:
                        raise RuntimeError('逐物理帧RGB初始化出现非连续帧')
                audit['sample_frames'].append(self.frame)
                previous = self.frame
            audit.update(status='aligned', stable_frame=self.frame,
                stable_timestamp=float(probe.timestamp.elapsed_seconds))
        except Exception as exc:
            audit.update(status='failed', error=type(exc).__name__+': '+str(exc))
            # _collect可能已经保存部分成功相机；不能被外层异常覆盖为空。
            self._save_camera_sync_failure(self.frame, exc, None)
            raise

    def _configure_experiment_world(self, client, requested_map, allow_reset):
        """先设同步固定步长再加载本轮世界；重载许可不等于接管已有实验。"""
        current_map = self.world.get_map().name.rstrip('/').split('/')[-1]
        different_map = current_map != requested_map.rstrip('/').split('/')[-1]
        if different_map and not allow_reset:
            raise RuntimeError('当前地图与配置不符；换图必须显式允许 allow_world_reset')
        self.saved_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0/self.physics_hz
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = .01
        settings.max_substeps = max(1, int(math.ceil(settings.fixed_delta_seconds/.01)))
        # RPC异常也可能发生在服务端已应用设置之后，收尾仍需尝试恢复并核验。
        self._settings_change_attempted = True
        self.world.apply_settings(settings)
        operation = 'keep_current_world'
        if allow_reset:
            # CARLA官方确定性顺序：同步设置必须在加载世界之前生效并保留。
            # 换图本身已创建新世界，不再执行第二次reload。
            if different_map:
                client.load_world(requested_map, reset_settings=False)
                operation = 'load_world_preserve_settings'
            else:
                client.reload_world(False)
                operation = 'reload_world_preserve_settings'
            self.world = client.get_world()
        actual = self.world.get_settings()
        if (not actual.synchronous_mode or actual.fixed_delta_seconds is None or
                abs(actual.fixed_delta_seconds-1.0/self.physics_hz) > 1e-8 or actual.no_rendering_mode):
            raise RuntimeError('本轮世界未保留同步固定步长或RGB渲染设置；拒绝继续')
        self.world_reset = {'authorized': allow_reset, 'operation': operation,
            'map': self.world.get_map().name, 'settings_preserved': True,
            'initial_frame': self.world.get_snapshot().frame}

    def _create_sensors(self):
        c = self.carla
        views = [('front', 0), ('front_left', -60), ('front_right', 60),
                 ('back', 180), ('back_left', -120), ('back_right', 120)]
        self.camera_actors = {}
        for view, yaw in views if self.rgb_enabled else []:
            bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            for key, value in (('image_size_x', self.width), ('image_size_y', self.height),
                               ('fov', self.fov), ('sensor_tick', 0.)):
                bp.set_attribute(key, str(value))
            if bp.has_attribute('enable_postprocess_effects'):
                enabled = self.cfg.get('rgb_postprocess_effects', True)
                bp.set_attribute('enable_postprocess_effects', 'true' if enabled else 'false')
            pose = c.Transform(c.Location(z=float(self.cfg.get('camera_height_m', 2.2))), c.Rotation(yaw=yaw))
            sensor = self.world.spawn_actor(bp, pose, attach_to=self.ego, attachment_type=c.AttachmentType.Rigid)
            self.sensors.append(sensor)
            key = view + '_rgb'
            q = queue.Queue(maxsize=4)
            self.queues[key] = q
            def callback(image, target=q, camera_key=key):
                self._on_camera_image(camera_key, target, image)
            sensor.listen(callback)
            self.camera_actors[key] = sensor
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor = self.world.spawn_actor(bp, c.Transform(), attach_to=self.ego)
        self.sensors.append(sensor)
        def collision(event):
            self.collision = True
            self.events.append({'kind': 'collision', 'frame': event.frame,
                                'phase': self._event_phase(event.frame),
                                'other_id': event.other_actor.id, 'other_type': event.other_actor.type_id})
        sensor.listen(collision)
        bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        sensor = self.world.spawn_actor(bp, c.Transform(), attach_to=self.ego)
        self.sensors.append(sensor)
        sensor.listen(lambda event: self.events.append({'kind': 'lane_invasion', 'frame': event.frame,
            'phase': self._event_phase(event.frame),
            'markings': [str(mark.type) for mark in event.crossed_lane_markings],
            'marking_evidence': [{'type': str(mark.type), 'lane_change': str(mark.lane_change)}
                                 for mark in event.crossed_lane_markings]}))

    def _event_phase(self, frame):
        # 事件可能延迟回调；按发生帧分类，不按回调抵达墙钟时间分类。
        return ('initialization' if self._initial_source_frame is None or
                frame <= self._initial_source_frame else 'runtime')

    def _add_timing(self, name, elapsed):
        self._timing[name] = self._timing.get(name, 0.) + float(elapsed)

    def _record_camera_event(self, key, event, image):
        """仅保留8个最近元数据；不保存图像内容，不改变采样或消费顺序。"""
        stamp = float(image.timestamp)
        item = {'frame': int(image.frame), 'timestamp': stamp if math.isfinite(stamp) else None}
        with self._camera_audit_lock:
            audit = self._camera_audit.setdefault(key, {})
            audit[event+'_count'] = audit.get(event+'_count', 0)+1
            recent = audit.setdefault('recent_'+event, [])
            recent.append(item)
            del recent[:-8]

    def _on_camera_image(self, key, target, image):
        self._record_camera_event(key, 'received', image)
        try:
            target.put_nowait(image)
        except queue.Full:
            try:
                discarded = target.get_nowait()
                self._record_camera_event(key, 'queue_dropped', discarded)
            except queue.Empty:
                pass
            try:
                target.put_nowait(image)
            except queue.Full:
                self._record_camera_event(key, 'queue_dropped', image)
        finally:
            # Any camera can release a shared collector wait. Waiting on the
            # first missing camera must not hide a conflicting peer header.
            self._camera_ready.set()

    def _save_camera_sync_failure(self, frame, error, selected):
        """记录失败现场的真实计数，不能用上一已发布场景的缓存运行状态代替。"""
        with self._camera_audit_lock:
            cameras = copy.deepcopy(self._camera_audit)
        for key, q in self.queues.items():
            audit = cameras.setdefault(key, {})
            audit['queue_size'] = q.qsize()
            sensor = getattr(self, 'camera_actors', {}).get(key)
            if sensor is not None:
                try:
                    audit.update(actor_id=sensor.id, is_alive=bool(sensor.is_alive),
                                 is_listening=bool(sensor.is_listening))
                except Exception as exc:
                    audit['actor_status_error'] = type(exc).__name__+': '+str(exc)
        previous = self.camera_sync_failure or {}
        accepted = (copy.deepcopy(previous.get('accepted_for_request', {}))
                    if selected is None and previous.get('requested_frame') == frame else
                    {key: {'frame': image.frame, 'timestamp': image.timestamp}
                     for key, image in (selected or {}).items()})
        report = {'version': 'carla-rgb-sync-diagnostic-1.2',
            'error': type(error).__name__+': '+str(error), 'requested_frame': frame,
            'error_kind': 'rgb_metadata_conflict' if isinstance(error, CameraMetadataError) else 'rgb_sync_failure',
            'metadata_conflict': copy.deepcopy(getattr(error, 'details', None)),
            'bridge_frame': self.frame, 'sim_time': self._sim_time,
            't0': self.t0, 'runtime_metrics': copy.deepcopy(self._runtime_metrics),
            'requested_camera_hz': float(getattr(self, 'physics_hz', 50)), 'camera_output_hz': 10.,
            'sensor_tick_s': 0., 'camera_barrier': 'six_views_before_next_physical_tick',
            'accepted_for_request': accepted,
            'pending_camera_keys': sorted(set(self.queues)-set(accepted)),
            'collection_policy': 'all_camera_queues_polled_under_one_shared_deadline',
            'cameras': cameras, 'world_reset': copy.deepcopy(self.world_reset),
            'initialization_warmup': copy.deepcopy(self._initialization_warmup),
            'initial_speed_audit': copy.deepcopy(self._initial_speed_audit),
            'rgb_initialization': copy.deepcopy(self._rgb_initialization),
            'initialization_ticks': copy.deepcopy(self._initialization_ticks),
            'initial_motion': copy.deepcopy(getattr(self.experiment, 'initial_motion_audit', None)),
            'initialization_events': copy.deepcopy(self.events) if self.t0 is None else [],
            'note': '同时核验原始帧号与时间戳；时间戳相同不能覆盖原始帧号冲突。'
                    '初始化相位推进单独记录，运行期不追加补图tick，不复用旧图。'}
        try:
            snap = self.world.get_snapshot()
            report['world_frame'] = snap.frame
            report['world_timestamp'] = snap.timestamp.elapsed_seconds
            settings = self.world.get_settings()
            report['world_settings'] = {key: getattr(settings, key, None) for key in
                ('synchronous_mode', 'fixed_delta_seconds', 'no_rendering_mode', 'substepping',
                 'max_substep_delta_time', 'max_substeps')}
        except Exception as exc:
            report['world_diagnostic_error'] = type(exc).__name__+': '+str(exc)
        self.camera_sync_failure = report
        root = getattr(self, 'root', None)
        if root is not None:
            path = Path(root)/'camera_sync_failure.json'
            temporary = path.with_suffix('.json.tmp')
            try:
                temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2),
                                     encoding='utf-8')
                os.replace(str(temporary), str(path))
                print('RGB同步失败现场已保存：'+str(path), file=sys.stderr, flush=True)
            except Exception as exc:
                print('RGB同步诊断保存失败：'+type(exc).__name__+': '+str(exc), file=sys.stderr, flush=True)

    def _collect(self, frame, prefetched=None):
        # 无图像消费者时绝不访问front_rgb，也不等待不存在的传感器。
        if not self.queues:
            self.frames = {}
            return
        started = time.monotonic()
        result = {}
        prefetched = dict(prefetched or {})
        try:
            snap = self.world.get_snapshot()
            if snap.frame != frame:
                raise RuntimeError('RGB等待开始时世界帧与目标帧不一致；拒绝旧图配新状态')
            target_time = float(snap.timestamp.elapsed_seconds)
            if (set(self.frames) == set(self.queues) and
                    all(image.frame == frame for image in self.frames.values())):
                for key, image in self.frames.items():
                    camera_matches_snapshot(key, image, frame, target_time)
                return
            deadline = time.monotonic() + self.timeout
            if self.t0 is None and self._rgb_initialization_deadline is not None:
                deadline = min(deadline, self._rgb_initialization_deadline)
            while len(result) != len(self.queues):
                if self._stop.is_set() and threading.current_thread() is self._ticker:
                    raise RuntimeError('RGB等待被运行暂停中断')
                # Clear before inspecting queues: a callback racing with this
                # scan leaves either its image queued or the event set.
                self._camera_ready.clear()
                for key, q in self.queues.items():
                    if key in result:
                        continue
                    while True:
                        try:
                            item = prefetched.pop(key) if key in prefetched else q.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            matched = camera_matches_snapshot(key, item, frame, target_time)
                        except CameraMetadataError:
                            self._record_camera_event(key, 'rejected_metadata', item)
                            raise
                        if matched:
                            result[key] = item
                            self._record_camera_event(key, 'selected', item)
                            break
                        self._record_camera_event(key, 'discarded_stale', item)
                if len(result) == len(self.queues):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    pending = ','.join(key for key in self.queues if key not in result)
                    raise TimeoutError('逐物理帧RGB同步超时：{}，目标帧{}'.format(pending, frame))
                self._camera_ready.wait(min(remaining, .1))
            timestamps = [item.timestamp for item in result.values()]
            if max(timestamps) - min(timestamps) > 1e-6:
                raise RuntimeError('同帧RGB时间戳不一致')
            snap = self.world.get_snapshot()
            if snap.frame != frame or any(abs(value-snap.timestamp.elapsed_seconds) > 1e-5 for value in timestamps):
                raise RuntimeError('RGB与目标世界帧时间戳不一致；拒绝旧图配新状态')
            self.frames = result
            self._camera_payload_cache = None
            self._runtime_metrics['rgb_synchronized_frames'] = self._runtime_metrics.get('rgb_synchronized_frames', 0)+1
            if self.t0 is not None:
                self._runtime_metrics['rgb_runtime_synchronized_frames'] = self._runtime_metrics.get('rgb_runtime_synchronized_frames', 0)+1
            self._runtime_metrics['last_rgb_synchronized_frame'] = frame
        except Exception as exc:
            self._save_camera_sync_failure(frame, exc, result)
            raise
        finally:
            self._add_timing('sensor_wait_s', time.monotonic()-started)

    def _camera_payload(self):
        if not self.rgb_enabled:
            return []
        if self._camera_payload_cache is not None and self._camera_payload_cache[0] == self.frame:
            return self._camera_payload_cache[1]
        started = time.monotonic()
        try:
            folder = self.root / 'camera_frames' / '{:08d}'.format(self.frame)
            if self.t0 is None or 'front_rgb' not in self.frames:
                raise ValueError('RGB源帧未就绪')
            saved = camera_storage_due(float(self.frames['front_rgb'].timestamp)-self.t0,
                self.cfg.get('camera_save_hz', 1), self._runtime_metrics['physical_ticks'], self.physics_hz)
            cameras = []
            for view in ('front', 'front_left', 'front_right', 'back', 'back_left', 'back_right'):
                rgb = self.frames[view + '_rgb']
                if rgb.frame != self.frame or not math.isfinite(rgb.timestamp):
                    raise ValueError('RGB必须来自当前冻结观测帧：'+view)
                if rgb.width != self.width or rgb.height != self.height:
                    raise ValueError('RGB尺寸不符合相机配置：'+view)
                rgb_bytes = bytes(rgb.raw_data)
                if len(rgb_bytes) != self.width*self.height*4:
                    raise ValueError('RGB BGRA字节数不符合相机尺寸：'+view)
                rgb_path = folder / (view + '.png')
                if saved and not rgb_path.is_file():
                    folder.mkdir(parents=True, exist_ok=True)
                    rgb.save_to_disk(str(rgb_path))
                transport = {'encoding': 'zlib-base64-bgra8-1.0', 'frame': int(rgb.frame),
                    'timestamp': float(rgb.timestamp), 'width': self.width, 'height': self.height,
                    'data': base64.b64encode(zlib.compress(rgb_bytes, 1)).decode('ascii')}
                cameras.append({'view': view, 'rgb_path': str(rgb_path) if saved else None,
                    'rgb_saved': saved, 'rgb_transport': transport,
                    'rgb_postprocess_effects': self.cfg.get('rgb_postprocess_effects', True),
                    'width': self.width, 'height': self.height, 'fov_deg': self.fov,
                    'frame': self.frame, 'timestamp': float(rgb.timestamp), 'sample_hz': 10.,
                    'world_to_camera_ue': rgb.transform.get_inverse_matrix(),
                    'camera_to_world_ue': rgb.transform.get_matrix()})
            self._camera_payload_cache = (self.frame, cameras)
            return cameras
        finally:
            self._add_timing('image_s', time.monotonic()-started)

    def _actor(self, actor, is_ego=False):
        transform, bb = actor.get_transform(), actor.bounding_box
        box_center_x = float(bb.location.x)
        center = bounding_box_world_center(transform,bb,self.carla)
        velocity, acceleration, omega = actor.get_velocity(), actor.get_acceleration(), actor.get_angular_velocity()
        yaw = math.radians(transform.rotation.yaw + bb.rotation.yaw)
        identity = 'ego' if is_ego else 'carla_{}'.format(actor.id)
        kind, classification_evidence = actor_category(actor)
        if kind is None:
            raise ValueError('对象没有交通参与者/路面障碍类别证据：' + actor.type_id)
        waypoint = self.map.get_waypoint(center, project_to_road=False, lane_type=self.carla.LaneType.Driving)
        offset = [center.x - transform.location.x, center.y - transform.location.y]
        angular = math.radians(omega.z)
        elapsed = self.world.get_snapshot().timestamp.elapsed_seconds
        previous = self.angular_history.get(actor.id)
        alpha = (angular - previous[1]) / (elapsed - previous[0]) if previous and elapsed > previous[0] + 1e-8 else 0.0
        self.angular_history[actor.id] = (elapsed, angular)
        result = {'id': identity, 'kind': kind, 'position': xy(center),
            'velocity': [velocity.x - angular * offset[1], -(velocity.y + angular * offset[0])],
            'acceleration': [acceleration.x - alpha * offset[1] - angular * angular * offset[0],
                             -(acceleration.y + alpha * offset[0] - angular * angular * offset[1])],
            'heading': -yaw, 'length': 2.0 * bb.extent.x, 'width': 2.0 * bb.extent.y,
            'height_m': 2.0 * bb.extent.z,
            'lane_id': lane_key(waypoint) if waypoint is not None else None,
            'visible': True, 'radius': max(bb.extent.x, bb.extent.y) if kind == 'pedestrian' else None,
            'z': center.z, 'origin_z': transform.location.z,
            'classification': {'version': OBJECT_CATEGORY_VERSION, 'kind': kind,
                               'evidence': classification_evidence, 'type_id': actor.type_id},
            'state_sources': {'position': 'bounding_box_world_center', 'velocity': 'actor_velocity_plus_angular_offset',
                              'acceleration': 'actor_acceleration_plus_causal_angular_offset',
                              'angular_acceleration': 'past_difference' if previous else 'initial_zero_assumption'}}
        if is_ego:
            wheels = actor.get_physics_control().wheels
            wheelbase, rear_distance = wheel_geometry(wheels, transform.get_inverse_matrix(), box_center_x)
            result['wheelbase_m'], result['rear_axle_to_center_m'] = wheelbase, rear_distance
            result['state_sources']['wheel_geometry'] = 'world_cm_to_actor_m;immutable_actor_local_bbox_center'
            result['state_sources']['wheel_geometry_version'] = 'carla-wheel-reference-1.1'
        return result

    def _environment_actor(self, entry):
        box = entry['box']
        # 倾斜静态框投影成沿原yaw的包络矩形，不缩小其真实平面占据。
        yaw = math.radians(box.rotation.yaw)
        vertices = box.get_world_vertices(self.carla.Transform())
        ex = max(abs((p.x-box.location.x)*math.cos(yaw)+(p.y-box.location.y)*math.sin(yaw)) for p in vertices)
        ey = max(abs(-(p.x-box.location.x)*math.sin(yaw)+(p.y-box.location.y)*math.cos(yaw)) for p in vertices)
        waypoint = self.map.get_waypoint(box.location, project_to_road=False,
                                        lane_type=self.carla.LaneType.Driving)
        return {'id': entry['id'], 'kind': entry['kind'], 'position': xy(box.location),
            'velocity': [0., 0.], 'acceleration': [0., 0.], 'heading': -math.radians(box.rotation.yaw),
            'length': 2.*ex, 'width': 2.*ey, 'height_m': max(p.z for p in vertices)-min(p.z for p in vertices),
            'lane_id': lane_key(waypoint) if waypoint is not None else None, 'visible': True,
            'radius': max(ex, ey) if entry['kind'] == 'pedestrian' else None,
            'z': box.location.z, 'origin_z': box.location.z,
            'environment_label': entry['label'], 'environment_ids': entry['source_ids'],
            'classification': {'version': OBJECT_CATEGORY_VERSION, 'kind': entry['kind'],
                               'evidence': entry['classification_evidence'],
                               'environment_label': entry['label']},
            'state_sources': {'position': 'environment_bbox_world_center',
                              'geometry': 'yaw_aligned_envelope_of_world_bbox_xy_projection',
                              'velocity': 'fixed_map_mesh_zero', 'acceleration': 'fixed_map_mesh_zero'}}

    def _range_environment(self, ego, radius):
        result = []
        for entry in self.environment_objects:
            box = entry['box']
            if math.hypot(box.location.x-ego['position'][0], -box.location.y-ego['position'][1]) > radius + math.sqrt(box.extent.x**2+box.extent.y**2+box.extent.z**2):
                continue
            # 仅作同层候选抓取；沿车道的最终范围由主进程公共对象筛选器决定。
            vertices = box.get_world_vertices(self.carla.Transform())
            if max(p.z for p in vertices) < ego['z']-ego['height_m']/2 or min(p.z for p in vertices) > ego['z']+ego['height_m']/2:
                continue
            result.append(self._environment_actor(entry))
        return result

    def _map_payload(self):
        points = self.map.generate_waypoints(float(self.cfg.get('map_resolution_m', 2.0)))
        # 添加真实路段端点及导航采样，避免只用均匀采样造成道路端部人工缺口。
        points.extend(self.route_wps)
        for start, end in self.map.get_topology():
            points.extend((start, end))
        endpoints = native_lane_endpoints(self.map)
        points.extend(endpoints)
        route_positions = [wp.transform.location for wp in self.route_wps]
        low_x, high_x = min(p.x for p in route_positions) - 120, max(p.x for p in route_positions) + 120
        low_y, high_y = min(p.y for p in route_positions) - 120, max(p.y for p in route_positions) + 120
        base_z = self.ego.get_location().z
        tolerance = float(self.cfg.get('same_level_tolerance_m', 1.5))
        elevation = RouteElevationSupport(self.route_wps,tolerance)
        groups = {}
        rejected_height = 0
        for wp in points:
            p = wp.transform.location
            if wp.lane_type != self.carla.LaneType.Driving:
                continue
            if low_x <= p.x <= high_x and low_y <= p.y <= high_y:
                if elevation.contains(p):
                    groups.setdefault(lane_key(wp), []).append(wp)
                else:
                    rejected_height += 1
        lanes = []
        for key, waypoints in sorted(groups.items()):
            unique = {}
            for waypoint in waypoints:
                unique[round(waypoint.s, 6)] = waypoint
            waypoints = list(unique.values())
            waypoints.sort(key=lambda w: w.s, reverse=waypoints[0].lane_id > 0)
            if len(waypoints) < 2:
                continue
            center, left, right, boundaries = [], [], [], []
            for wp in waypoints:
                p, yaw, width = wp.transform.location, math.radians(wp.transform.rotation.yaw), wp.lane_width
                center.append(xy(p))
                left.append([p.x + math.sin(yaw) * width / 2, -p.y + math.cos(yaw) * width / 2])
                right.append([p.x - math.sin(yaw) * width / 2, -p.y - math.cos(yaw) * width / 2])
            for side, coordinates in (('left', left), ('right', right)):
                for i in range(len(waypoints) - 1):
                    mark = getattr(waypoints[i], side + '_lane_marking')
                    permission = waypoints[i].lane_change
                    direction = int(self.carla.LaneChange.Left if side == 'left' else self.carla.LaneChange.Right)
                    semantics = boundary_semantics(mark.type, bool(int(permission) & direction),
                        bool(int(mark.lane_change) & direction), bool(waypoints[i].is_junction))
                    start, end = waypoints[i], waypoints[i + 1]
                    start_junction, end_junction = getattr(start, 'junction_id', None), getattr(end, 'junction_id', None)
                    junction = (start_junction if start.is_junction and end.is_junction
                                and type(start_junction) is int and start_junction >= 0
                                and start_junction == end_junction else None)
                    boundaries.append({'id': '{}:{}:{}'.format(key, side, i), 'points': coordinates[i:i + 2],
                        'owner_lane_id': key, 'side': side, 'is_junction': junction is not None,
                        'junction_id': junction, 'line_type': str(mark.type), **semantics})
            middle = waypoints[len(waypoints) // 2]
            adjacent = {}
            for side in ('left', 'right'):
                other = getattr(middle, 'get_' + side + '_lane')()
                same = other is not None and other.lane_type == self.carla.LaneType.Driving
                if same:
                    dyaw = math.radians(other.transform.rotation.yaw - middle.transform.rotation.yaw)
                    same = math.cos(dyaw) > 0.5
                adjacent[side] = lane_key(other) if same and lane_key(other) in groups else None
            # Direct topology, not a 4m leap that can skip short road sections.
            successors = sorted(set(lane_key(w) for w in waypoints[-1].next(.01)
                                    if lane_key(w) != key and lane_key(w) in groups))
            lanes.append({'id': key, 'center': center, 'polygon': left + right[::-1],
                'width': middle.lane_width, 'speed_limit': float(self.cfg.get('map_speed_cap_mps', 30)),
                'successors': successors, 'left': adjacent['left'], 'right': adjacent['right'],
                'priority': 0, 'boundaries': boundaries})
        route_ids = []
        for wp in self.route_wps:
            key = lane_key(wp)
            if not route_ids or key != route_ids[-1]:
                route_ids.append(key)
        missing=set(route_ids)-{lane['id'] for lane in lanes}
        if missing:
            raise ValueError('导航高度地图缺少路线车道：'+','.join(sorted(missing)))
        return {'lanes': lanes, 'route': [xy(p) for p in route_positions], 'route_lane_ids': route_ids,
                'identity': map_identity(self.map),
                'source': 'CARLA_public_waypoints_navigation_relative_elevation', 'priority_available': False,
                'topology_sampling': {'native_lane_section_endpoint_count': len(endpoints),
                    'direct_successor_query_m': .01,
                    'geometry_policy': 'native_endpoint_positions_widths_and_yaws;no_polygon_padding'},
                'elevation_filter': {'version':elevation.VERSION,'tolerance_m':tolerance,
                    'map_corridor_radius_m':elevation.radius_m,'rejected_waypoint_samples':rejected_height,
                    'rule':'nearest_xy_route_segment_interpolated_height;reject_multilevel_route_overlap',
                    'scope':'地图坡度延伸；当前对象仅按范围与同层高度准入，不作遮挡判断'},
                'crosswalks': closed_crosswalks(self.map.get_crosswalks(), base_z, tolerance,elevation),
                'lane_speed_source': 'configured_map_speed_cap_mps;current_ego_limit_from_CARLA'}

    def _entities(self):
        # 同物理帧只读取一次，避免历史采样与输出各自重复查询对象。
        if self._entities_cache is not None and self._entities_cache[0] == self.frame:
            return self._entities_cache[1:]
        ego = self._actor(self.ego, True)
        radius = float(self.cfg.get('perception_radius_m', 120.))
        actors = []
        category_decisions = []
        for actor in self.world.get_actors():
            if actor.id == self.ego.id or not (actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.pedestrian.') or actor in self.experiment.actors and actor.type_id.startswith('static.')):
                continue
            kind, evidence = actor_category(actor)
            category_decisions.append({'id': 'carla_{}'.format(actor.id), 'type_id': actor.type_id,
                                       'kind': kind, 'included': kind is not None, 'evidence': evidence})
            if kind is None:
                continue
            center = bounding_box_world_center(actor.get_transform(), actor.bounding_box, self.carla)
            if abs(center.z-ego['z']) > float(self.cfg.get('same_level_tolerance_m', 1.5)):
                continue
            if math.hypot(center.x-ego['position'][0], -center.y-ego['position'][1]) > radius:
                continue
            actors.append(self._actor(actor))
        static_actors = self._range_environment(ego, radius)
        actors.extend(static_actors)
        actors.sort(key=lambda actor: actor['id'])
        self.environment_audit['runtime_actor_category_decisions'] = category_decisions
        self._entities_cache = (self.frame, ego, actors, len(static_actors))
        return ego, actors, len(static_actors)

    def _sample_history(self, force=False):
        elapsed = self.world.get_snapshot().timestamp.elapsed_seconds-self.t0
        tick = self._runtime_metrics['physical_ticks']
        period = self.physics_hz // 2
        index = tick // period
        if not force and (tick % period or index <= self._history_sample_index):
            return
        started = time.monotonic()
        ego, actors, _ = self._entities()
        for actor in [ego]+actors:
            if actor['kind'] == 'static_obstacle':
                continue
            sample = {'time': float(elapsed), 'physical_tick': tick, 'frame': self.frame,
                'position': list(actor['position']),
                'velocity': list(actor['velocity']), 'acceleration': list(actor['acceleration']),
                'heading': actor['heading'], 'observed': True}
            samples = self._history.setdefault(actor['id'], [])
            if not samples or abs(samples[-1]['time']-elapsed) > 1e-7:
                samples.append(sample)
        # 不在范围内的对象只保留已有历史，绝不外推填充当前对象。
        for identity in list(self._history):
            self._history[identity] = [row for row in self._history[identity]
                                      if row['time'] >= elapsed-3.-1e-6]
            if not self._history[identity]:
                del self._history[identity]
        self._history_sample_index = index
        self._add_timing('snapshot_state_s', time.monotonic()-started)

    def _runtime_payload(self):
        source = self.timeline.source_time
        applied = self._last_applied_control
        now = time.monotonic()
        return dict(self._runtime_metrics, mode=self.runtime_mode,
            world_driver='application_wall_paced_sync_ticks' if self.runtime_mode == 'async' else 'synchronous_plan_then_tick',
            carla_native_synchronous=True, physics_hz=self.physics_hz, planner_hz=self.planner_hz,
            control_step_s=self.control_step_s, camera_hz=10. if self.rgb_enabled else 0.,
            camera_capture_hz=float(self.physics_hz) if self.rgb_enabled else 0.,
            sensor_tick_s=0., camera_barrier='six_views_before_next_physical_tick',
            rgb_enabled=self.rgb_enabled, depth_sensor_count=0,
            duration_s=self.duration_s,
            duration_target_ticks=duration_tick_limit(self.duration_s, self.physics_hz),
            terminal_reason=self._terminal_reason,
            sequence_exhausted=self._terminal_reason == 'sequence_exhausted',
            paused=self._paused, realtime_started=self._ticker is not None,
            startup_policy='world_paused_until_first_valid_control_sequence',
            startup_paused_wall_s=None if self._initialized_at is None else
                max(0., (self._started_at or now)-self._initialized_at),
            current_sim_time=self._sim_time,
            plan_source_time=source,
            plan_age_s=None if source is None else max(0., self._sim_time-source),
            control_age_s=None if applied is None else max(0., self._sim_time-applied['applied_time']),
            observation_age_s=None if self._latest is None else max(0., self._sim_time-self._latest['time']),
            observation_wall_age_s=None if self._latest_published_at is None else max(0., now-self._latest_published_at),
            worker_error=self._worker_error,
            execution_controller=(None if self._execution_controller is None else
                dict(version=self._execution_controller.VERSION,
                     period_s=self._execution_controller.dt,
                     maximum_steering_rad=self._execution_controller.maximum_steering,
                     settings=dict(self._execution_controller.settings))),
            last_applied_control=copy.deepcopy(self._last_applied_control))

    def snapshot(self, include_cameras=True):
        started = time.monotonic()
        image_before = self._timing.get('image_s', 0.)
        snap = self.world.get_snapshot()
        if snap.frame != self.frame:
            raise RuntimeError('检测到其他客户端推进世界；拒绝输出不一致的观测帧')
        ego, actors, static_count = self._entities()
        lights = []
        for light in self.world.get_actors().filter('traffic.traffic_light*'):
            locations = light.get_stop_waypoints()
            affected = [{'lane_id': lane_key(wp), 'position': xy(wp.transform.location)} for wp in locations]
            if affected:
                lights.append({'id': 'traffic_light_{}'.format(light.id), 'state': str(light.get_state()).split('.')[-1].lower(), 'stops': affected})
        actual = self.ego.get_control()
        current_wp = self.map.get_waypoint(bounding_box_world_center(self.ego.get_transform(),self.ego.bounding_box,self.carla),
                                          project_to_road=False, lane_type=self.carla.LaneType.Driving)
        road_evidence = None
        if current_wp is not None:
            road_evidence = {'lane_id': lane_key(current_wp)}
            for side in ('left', 'right'):
                other = getattr(current_wp, 'get_' + side + '_lane')()
                exists = other is not None and other.lane_type == self.carla.LaneType.Driving
                if exists:
                    exists = math.cos(math.radians(other.transform.rotation.yaw-current_wp.transform.rotation.yaw)) > .5
                road_evidence[side] = lane_key(other) if exists else None
        cameras = self._camera_payload() if include_cameras else []
        weather = self.world.get_weather()
        weather_record = {name: float(getattr(weather, name)) for name in (
            'cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity',
            'sun_azimuth_angle', 'sun_altitude_angle', 'fog_density', 'fog_distance',
            'wetness', 'fog_falloff', 'scattering_intensity', 'mie_scattering_scale',
            'rayleigh_scattering_scale', 'dust_storm') if hasattr(weather, name)}
        if self.world.get_snapshot().frame != self.frame:
            raise RuntimeError('读取场景期间世界被其他客户端推进；拒绝输出不同步场景')
        self._add_timing('snapshot_state_s', max(0., time.monotonic()-started-
            (self._timing.get('image_s', 0.)-image_before)))
        return {'frame': self.frame, 'time': snap.timestamp.elapsed_seconds - self.t0,
            'timestamp': snap.timestamp.elapsed_seconds, 'ego': ego, 'actors': actors,
            'cameras': cameras, 'traffic_lights': lights,
            'speed_limit_mps': self.ego.get_speed_limit() / 3.6,
            # Script transitions and hidden actor intent stay in the independent
            # scenario/evaluation channel, never in the planner observation.
            'collision': any(event.get('kind') == 'collision' and event.get('phase') == 'runtime'
                             for event in self.events), 'events': list(self.events),
            'actual_actuation': {'throttle': actual.throttle, 'brake': actual.brake,
                'steer_normalized_right': actual.steer, 'hand_brake': actual.hand_brake,
                'reverse': actual.reverse, 'gear': actual.gear},
            'control_response': copy.deepcopy(self._last_control_response),
            'current_road_evidence': road_evidence,
            'environment_obstacles': dict(self.environment_audit, in_range_count=static_count),
            'weather': weather_record, 'weather_source': ('scenario_weather_then_current_world_get_weather'
                if self.cfg.get('experiment',{}).get('weather') is not None else 'current_world_get_weather_no_override'),
            'source': 'range_limited_ground_truth_no_occlusion',
            'visibility_radius_m': float(self.cfg.get('perception_radius_m', 120.)),
            'observation_support': {'kind': 'range_disk', 'center_world_xy': list(ego['position']),
                'radius_m': float(self.cfg.get('perception_radius_m', 120.)), 'occlusion_test': False},
            'observation_evidence': {'mode': 'range_limited_ground_truth_no_occlusion',
                'occlusion_test': False, 'not_camera_visibility': True,
                'height_filter': 'same_level_geometry', 'hidden_routes_or_future': False},
            'observed_history': copy.deepcopy(self._history),
            'last_applied_control': copy.deepcopy(self._last_applied_control),
            'runtime': self._runtime_payload(), 'bridge_timing': dict(self._timing),
            'evaluation': copy.deepcopy(self._evaluation),
            'public_road_rules': self.experiment.public_rule_payload(),
            'scenario': 'carla'}

    def _show_plan(self, plan):
        """显示层使用真实世界坐标；不参与感知/控制。debug可能进入相机，不能声称原图隔离。"""
        enabled = self.cfg.get('world_plan_debug', False)
        result = {'version': 'carla-plan-display-1.0', 'world_debug_enabled': enabled,
                  'points_carla_xyz': [], 'segments_drawn': 0,
                  'source_carla_frame': self.frame, 'life_time_s': 1.1 / self.planner_hz,
                  'sensor_debug_exposure': 'possible' if enabled else 'disabled'}
        if not plan:
            return result
        if (not isinstance(plan,list) or len(plan)>1000 or
                any(not isinstance(p,list) or len(p)!=2 or
                    any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in p)
                    for p in plan)):
            raise ValueError('显示规划必须为有限世界二维点列表')
        c = self.carla
        origin_z = float(self.ego.get_location().z)
        for x, y in plan:
            point = c.Location(x=float(x), y=-float(y), z=origin_z)
            waypoint = self.map.get_waypoint(point, project_to_road=True, lane_type=c.LaneType.Driving)
            z = float(waypoint.transform.location.z) if waypoint is not None else float(point.z)
            result['points_carla_xyz'].append([float(x), -float(y), z+.15])
        if enabled:
            try:
                for a,b in zip(result['points_carla_xyz'],result['points_carla_xyz'][1:]):
                    if sum((x-y)**2 for x,y in zip(a,b)) < 1e-10:
                        continue
                    self.world.debug.draw_line(c.Location(x=a[0],y=a[1],z=a[2]),
                        c.Location(x=b[0],y=b[1],z=b[2]), thickness=.08,
                        color=c.Color(0,230,255), life_time=result['life_time_s'], persistent_lines=False)
                    result['segments_drawn'] += 1
                # 同步查询降低异步绘制请求迟到风险，但不是官方保证的debug flush。
                self.world.get_settings()
            except Exception as exc:
                result['error'] = type(exc).__name__+': '+str(exc)
                print('CARLA世界规划线绘制失败：'+result['error'], file=sys.stderr, flush=True)
        return result

    def _publish(self, result):
        # 状态/图片仅由推进世界的一方读取；RPC读取不可变副本。
        with self._condition:
            runtime = result.get('runtime', {})
            runtime.update(source_frame=result['frame'], source_time=result['time'],
                current_sim_time=result['time'], observation_age_s=0., observation_wall_age_s=0.)
            self._latest = copy.deepcopy(result)
            self._published_sources[result['frame']] = (result['time'], runtime['physical_ticks'])
            # 控制地平线只有3秒；保留8秒冻结源记录供异步提交校验。
            cutoff = runtime['physical_ticks']-8*self.physics_hz
            self._published_sources = {frame: source for frame, source in self._published_sources.items()
                                       if source[1] >= cutoff}
            self._terminal_published = result.get('runtime', {}).get('terminal_reason') is not None
            self._latest_published_at = time.monotonic()
            self._condition.notify_all()

    def _latest_result(self):
        with self._condition:
            result = copy.deepcopy(self._latest)
            display_frames, self._display_frames = self._display_frames, []
        if result is None:
            raise RuntimeError('CARLA尚未初始化观测')
        # 源帧的终态和控制证据必须随源状态一起冻结。后台推进信息单独标识，
        # 否则终止发布期间的live标志会把旧ego/图片误判成最终观测。
        result['runtime']['live'] = self._runtime_payload()
        result['display_frames'] = display_frames
        return result

    def _install(self, controls, control_step_s, source_time, plan, source_speed_mps, source_frame=None):
        if self.closed or self._terminal_reason is not None:
            raise RuntimeError('CARLA执行已终止：{}'.format(self._terminal_reason or 'closed'))
        if (type(control_step_s) not in (int, float) or not math.isfinite(control_step_s) or
                abs(control_step_s-self.control_step_s) > 1e-8):
            raise ValueError('控制网格必须保持0.1秒')
        # 校验完整批次，失败不能损坏当前正在执行的序列。
        candidate = ControlTimeline(self.timeline.minimum, self.timeline.maximum, self.timeline.steering_max,
                                    self.physics_hz)
        if source_speed_mps is None:
            raise ValueError('必须传递规划原帧自车速度，不按每段实测速误差重新生成目标速度')
        recorded_source = self._published_sources.get(source_frame) if type(source_frame) is int else None
        if recorded_source is None:
            raise ValueError('控制必须引用本轮已发布的冻结世界帧')
        recorded_time, source_tick = recorded_source
        if abs(recorded_time-source_time) > 1e-7:
            raise ValueError('控制源帧与源时间不匹配')
        candidate.install(controls, source_time, control_step_s, source_speed_mps, source_tick=source_tick)
        if len(candidate.controls) != 30:
            raise ValueError('CARLA必须提交完整30段MPC控制序列')
        if plan is not None and (not isinstance(plan, list) or len(plan) > 1000 or
                any(not isinstance(point, list) or len(point) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                        for v in point) for point in plan)):
            raise ValueError('显示规划必须为有限世界二维点列表')
        with self._condition:
            if self.closed or self._terminal_reason is not None:
                raise RuntimeError('CARLA执行已终止：{}'.format(self._terminal_reason or 'closed'))
            if self.timeline.source_time is not None and source_time < self.timeline.source_time-1e-8:
                raise ValueError('旧源帧控制不能覆盖新控制序列')
            state = candidate.at(self._sim_time, self._runtime_metrics['physical_ticks'])
            if not state['available']:
                raise ValueError('新控制序列无法用于当前仿真时间：'+state['reason'])
            candidate.revision = self.timeline.revision+1
            previous = self.timeline.at(self._sim_time, self._runtime_metrics['physical_ticks'])
            previous_speed = previous.get('target_speed_mps') if previous.get('available') else None
            self._reference_transition = {'source_frame': source_frame,
                'source_speed_mps': source_speed_mps,
                'previous_segment_endpoint_speed_mps': previous_speed,
                'new_segment_endpoint_speed_mps': state['target_speed_mps'],
                'endpoint_speed_jump_mps': (None if previous_speed is None else
                    state['target_speed_mps']-previous_speed),
                'policy': 'source_state_feedback_reference;actuator_feedback_state_retained'}
            self.timeline = candidate
            self._runtime_metrics['submitted_sequences'] = self._runtime_metrics.get('submitted_sequences', 0)+1
            self._runtime_metrics['last_submission_skipped_segments'] = state['index']
            self._pending_plan = (candidate.revision, copy.deepcopy(plan), source_frame)
            self._paused = False

    def _apply_current(self):
        with self._condition:
            state = self.timeline.at(self._sim_time, self._runtime_metrics['physical_ticks'])
            if not state['available']:
                self._terminal_reason = state['reason']
                if state['reason'] == 'sequence_exhausted':
                    self._worker_error = 'RuntimeError: MPC控制序列耗尽；未补写控制、未保持末项推进'
                return False
            pending = self._pending_plan
            self._pending_plan = None
            source_time = self.timeline.source_time
            source_tick = self.timeline.source_tick
            source_frame = self._initial_source_frame+source_tick
        if pending is not None:
            self._visualization = self._show_plan(pending[1])
            self._visualization['sequence_revision'] = pending[0]
            self._visualization['source_carla_frame'] = pending[2]
        # 同计划同物理帧重复调用幂等。异步等待期间的新计划允许替换本帧命令，
        # 控制器回到该帧反馈积分前状态再计算，不能重复积累同一帧误差。
        tick = self._runtime_metrics['physical_ticks']
        key = (tick, state['revision'], state['index'])
        if key != self._applied_key:
            acceleration, steering = state['control']
            speed, measured_acceleration = self._forward_kinematics()
            # 源时标中途接纳的段只执行剩余部分，过期段不重放。
            remaining = state['remaining_segment_s']
            # v参考仍由规划原帧积分，取下一物理步目标；有符号a直接参与踏板前馈，
            # 不再让小速度误差经Ackermann两层增量PID弱化制动/起步意图。
            elapsed_in_segment = self.control_step_s-remaining
            desired = max(0., state['segment_start_speed_mps']+
                acceleration*min(self.control_step_s, elapsed_in_segment+1./self.physics_hz))
            if self._execution_controller is None:
                raise RuntimeError('车辆执行控制器尚未按真实轮转角初始化')
            actual = self.ego.get_control()
            # tick 0观测紧随初始化set_target_velocity，其加速度含人工速度跳变；
            # tick 1起反馈来自正常物理控制响应，不过滤或钳制真实制动/加速测量。
            feedback_valid = tick != 0 or self._initial_speed_audit is None
            requested = self._execution_controller.step(tick, acceleration, steering, desired,
                speed, measured_acceleration,
                -float(actual.steer)*self._execution_controller.maximum_steering,
                acceleration_feedback_valid=feedback_valid,
                acceleration_feedback_reason=('physical_step_response' if feedback_valid else
                    'initial_motion_activation_set_target_velocity'))
            command = self.carla.VehicleControl(throttle=requested['throttle'], brake=requested['brake'],
                steer=requested['steer_normalized_right'], hand_brake=False, reverse=False,
                manual_gear_shift=False)
            self.ego.apply_control(command)
            self._last_command = {'acceleration_mps2': acceleration, 'steering_rad_left': steering,
                'planned_segment_endpoint_speed_mps': state['target_speed_mps'],
                'target_speed_mps': desired, 'measured_speed_mps': speed,
                'speed_error_mps': desired-speed, 'requested_actuation': requested,
                'reference_transition': copy.deepcopy(self._reference_transition),
                'actuation_mapping': SignedAccelerationController.VERSION}
            self._last_applied_control = dict(self._last_command, sequence_index=state['index'],
                sequence_revision=state['revision'], source_time=source_time,
                source_frame=source_frame, source_tick=source_tick, applied_tick=tick,
                applied_time=self._sim_time, frame=self.frame,
                control=[acceleration, steering], remaining_segment_s=remaining)
            self._applied_key = key
        return True

    def _forward_kinematics(self):
        transform = self.ego.get_transform()
        forward = transform.get_forward_vector()
        velocity, acceleration = self.ego.get_velocity(), self.ego.get_acceleration()
        speed = velocity.x*forward.x+velocity.y*forward.y+velocity.z*forward.z
        measured_acceleration = (acceleration.x*forward.x+acceleration.y*forward.y+
                                 acceleration.z*forward.z)
        return float(speed), float(measured_acceleration)

    def _read_control_response(self):
        if self._last_applied_control is None:
            return None
        actual = self.ego.get_control()
        speed, acceleration = self._forward_kinematics()
        response = copy.deepcopy(self._last_applied_control)
        response.update(observed_frame=self.frame,
            observed_tick=self._runtime_metrics['physical_ticks'],
            observed_time=self._sim_time, measured_speed_after_mps=speed,
            measured_acceleration_after_mps2=acceleration,
            actual_actuation={'throttle': float(actual.throttle), 'brake': float(actual.brake),
                'steer_normalized_right': float(actual.steer), 'hand_brake': bool(actual.hand_brake),
                'reverse': bool(actual.reverse), 'gear': int(actual.gear)},
            semantics='command_applied_before_tick;actual_control_and_motion_observed_after_tick')
        requested = response['requested_actuation']
        response['pedal_readback_matches_request'] = (abs(actual.throttle-requested['throttle']) <= .01 and
            abs(actual.brake-requested['brake']) <= .01)
        response['braking_intent_with_actual_throttle'] = (response['acceleration_mps2'] < -.05 and
            actual.throttle > .01)
        # 只记录实际响应，不把踏板指令一致当成加速度/安全已验收。
        response['acceleration_tracking_error_after_mps2'] = response['acceleration_mps2']-acceleration
        return response

    def _tick_once(self):
        previous_frame = self.world.get_snapshot().frame
        if previous_frame != self.frame:
            raise RuntimeError('本次tick之前世界已被额外推进：桥接帧{}，世界帧{}；只能有一个tick客户端'.format(
                self.frame, previous_frame))
        started = time.monotonic()
        self.experiment.before_tick_tick(self._runtime_metrics['physical_ticks'], 1./self.physics_hz)
        self._add_timing('scenario_s', time.monotonic()-started)
        started = time.monotonic()
        self.frame = self.world.tick(self.timeout)
        self._add_timing('tick_s', time.monotonic()-started)
        snap = self.world.get_snapshot()
        self._sim_time = float(snap.timestamp.elapsed_seconds-self.t0)
        self._runtime_metrics['physical_ticks'] += 1
        if self.frame != previous_frame+1 or snap.frame != self.frame:
            raise RuntimeError('单次tick检测到额外世界推进：推进前{}，tick返回{}，世界{}；只能有一个tick客户端'.format(
                previous_frame, self.frame, snap.frame))
        self._sample_history()
        self.experiment.after_tick(self.frame, self._runtime_metrics['physical_ticks'], snap)
        self._last_control_response = self._read_control_response()
        if self.evaluation_probe is not None:
            self._evaluation = self.evaluation_probe.observe(snap, self._runtime_metrics['physical_ticks'],
                events=list(self.events), scenario_events=self.experiment.evaluation_state(),
                control_execution=self._last_control_response)
            if any(item.get('kind') == 'initialization_collision'
                   for item in self._evaluation.get('infrastructure_errors', [])):
                self._terminal_reason = 'initialization_collision'
            elif self._evaluation.get('collision'):
                self._terminal_reason = 'collision'
            elif self._evaluation.get('arrival'):
                self._terminal_reason = 'arrival'
            elif self._evaluation.get('stalled'):
                self._terminal_reason = 'stall'
        if self.rgb_enabled:
            # 包含非输出帧和终止帧：收到全部当前图像前绝不发下一tick，
            # 避免CARLA0.9.15异步读回期间世界状态继续推进。
            self._collect(self.frame)
        if self.rgb_enabled and self._at_frequency(10.):
            # 录像/观看独立消费10Hz；模型及归档仍只消费整秒原帧。
            if self._display_enabled or self._at_frequency(1.):
                self._queue_display_frame(self._camera_payload())

    def _queue_display_frame(self, cameras):
        if not cameras or not self._display_enabled:
            return
        packet = {'frame': self.frame, 'time': self._sim_time, 'cameras': cameras,
                  'physical_tick': self._runtime_metrics['physical_ticks'],
                  'plan_source_frame': (None if self.timeline.source_tick is None else
                                        self._initial_source_frame+self.timeline.source_tick),
                  'plan_visualization': copy.deepcopy(self._visualization or {})}
        with self._condition:
            if self._display_frames and self._display_frames[-1]['frame'] == self.frame:
                return
            if len(self._display_frames) >= self.display_capacity:
                self._display_frames.pop(0)
                self._runtime_metrics['display_frames_dropped'] = self._runtime_metrics.get('display_frames_dropped', 0)+1
            self._display_frames.append(packet)

    def _capture(self, include_cameras=True):
        if include_cameras:
            self._collect(self.frame)
        result = self.snapshot(include_cameras=include_cameras)
        if self._visualization is not None:
            result['plan_visualization'] = dict(self._visualization, capture_carla_frame=self.frame)
        if self._last_command is not None:
            result['command'] = copy.deepcopy(self._last_command)
        self._publish(result)
        return result

    def _at_source_frame(self):
        return self._at_frequency(self.planner_hz)

    def _at_frequency(self, frequency):
        # t0对齐的是相机正式源帧；运行期采样和规划只按已推进的真实帧计数。
        period = self.physics_hz / frequency
        if period < 1 or not period.is_integer():
            raise ValueError('采样周期必须与物理帧整除对齐')
        return self._runtime_metrics['physical_ticks'] % int(period) == 0

    def _duration_reached(self):
        with self._condition:
            if self._terminal_reason is not None:
                return True
            target = duration_tick_limit(self.duration_s, self.physics_hz)
            # 初始化帧不计入轮长。3秒=150个正式物理tick，即使CARLA报告
            # 2.9999999329秒也已结束；原始时间戳仍原样保留在快照与日志中。
            if target is not None and self._runtime_metrics['physical_ticks'] >= target:
                self._terminal_reason = 'duration'
                return True
            return False

    def _run_realtime(self):
        dt = 1./self.physics_hz
        started = time.monotonic()
        tick_count = 0
        try:
            while not self._stop.is_set():
                if self._duration_reached() or not self._apply_current():
                    break
                deadline = started+(tick_count+1)*dt
                delay = max(0., deadline-time.monotonic())
                wait_started = time.monotonic()
                interrupted = self._stop.wait(delay)
                waited = time.monotonic()-wait_started
                self._add_timing('pacing_s', waited)
                self._runtime_metrics['wall_pacing_wait_s'] += waited
                if interrupted:
                    break
                lag = max(0., time.monotonic()-deadline)
                self._runtime_metrics['wall_lag_max_s'] = max(self._runtime_metrics['wall_lag_max_s'], lag)
                if lag >= dt:
                    self._runtime_metrics['wall_deadline_misses'] += 1
                # 提交可能发生在等待期间；推进前重新选择源时标控制。
                if not self._apply_current():
                    break
                self._tick_once()
                tick_count += 1
                if self._at_source_frame():
                    self._capture()
        except Exception as exc:
            if not self._stop.is_set():
                self._worker_error = type(exc).__name__+': '+str(exc)
                self._terminal_reason = 'bridge_error'
        finally:
            self._paused = True
            if self._terminal_reason is not None:
                try:
                    # 终止可能落在非RGB采样帧；不将旧图挂到新状态。
                    self._capture(include_cameras=False)
                except Exception as exc:
                    self._worker_error = (self._worker_error or '')+'; terminal_snapshot: '+str(exc)
            with self._condition:
                self._condition.notify_all()

    def step(self, controls, control_step_s, source_time, plan=None, source_speed_mps=None, source_frame=None):
        with self._condition:
            if self.runtime_mode == 'async' and self._terminal_reason is not None:
                # 规划期间后台可能已到轮末。等终态与真实最终状态一起发布，
                # 不把旧快照当终态，也不为这条迟到的规划追加物理推进。
                deadline = time.monotonic()+self.timeout
                while not self._terminal_published:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('等待CARLA最终观测发布超时')
                    self._condition.wait(remaining)
                return self._latest_result()
            # 与_duration_reached共用锁，避免检查结束后、安装计划前再到达轮末。
            self._install(controls, control_step_s, source_time, plan, source_speed_mps, source_frame)
        if self._started_at is None:
            self._started_at = time.monotonic()
        if self.runtime_mode == 'async':
            with self._condition:
                if self._ticker is None:
                    self._ticker = threading.Thread(target=self._run_realtime, name='carla-50hz-ticker', daemon=True)
                    self._ticker.start()
                return self._latest_result()
        # 1/2/10Hz分别执行10/5/1段控制；每段0.1秒、五个物理子步。
        try:
            for _ in range(self.physics_hz // self.planner_hz):
                if self._duration_reached() or not self._apply_current():
                    break
                self._tick_once()
            self._duration_reached()
            self._paused = True
            self._capture(include_cameras=self._at_frequency(10.))
            return self._latest_result()
        except Exception:
            self._paused = True
            raise

    def observe(self, after_frame=None):
        if after_frame is not None and (type(after_frame) is not int or after_frame < 0):
            raise ValueError('after_frame必须为非负整数或空')
        with self._condition:
            deadline = time.monotonic()+self.timeout
            while (after_frame is not None and self._latest is not None and
                    self._latest['frame'] <= after_frame and self.runtime_mode == 'async' and
                    self._ticker is not None and self._ticker.is_alive() and
                    not self._terminal_published and not self._stop.is_set()):
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('等待下一规划时钟CARLA源帧超时')
                self._condition.wait(remaining)
            return self._latest_result()

    def pause(self):
        self._stop.set()
        ticker = self._ticker
        if ticker is not None and ticker is not threading.current_thread():
            ticker.join(self.timeout+2.)
            if ticker.is_alive():
                raise RuntimeError('CARLA后台推进未停止；拒绝并发销毁传感器或恢复设置')
        self._paused = True
        if self.world is not None and self._latest is not None:
            if self._terminal_reason is None:
                self._terminal_reason = 'paused'
                self._capture(include_cameras=False)
            return self._latest_result()
        return {'paused': True}

    def _verify_world_cleanup(self, report, destroyed_after_frame, owned_ids):
        """只读新快照核验；不以Actor描述缓存或旧帧判断销毁成功。"""
        expected = self._settings_record(self.saved_settings)
        actual = self._settings_record(self.world.get_settings())
        report['expected_settings'] = expected
        report['restored_settings'] = actual
        report['settings_mismatches'] = [key for key in expected if actual[key] != expected[key]]
        if actual['synchronous_mode']:
            raise RuntimeError('收尾后世界仍为同步模式；不主动tick或等待外来同步世界')
        if report['settings_mismatches']:
            raise RuntimeError('收尾世界设置回读不一致：'+','.join(report['settings_mismatches']))
        started = time.monotonic()
        deadline = started+5.
        snapshot = self.world.get_snapshot()
        while True:
            report['observed_frame'] = int(snapshot.frame)
            report['fresh_snapshot'] = snapshot.frame > destroyed_after_frame
            report['remaining_owned_ids'] = sorted(owned_ids & {int(actor.id) for actor in snapshot})
            if report['fresh_snapshot'] and not report['remaining_owned_ids']:
                break
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise RuntimeError('5秒内未确认自建演员消失/销毁后新快照；残留ID={}'.format(
                    report['remaining_owned_ids']))
            try:
                snapshot = self.world.wait_for_tick(remaining)
            finally:
                report['passive_wait_seconds'] = time.monotonic()-started
        # 等待期间有其他客户端改设置时，不把本轮恢复记为成功。
        actual = self._settings_record(self.world.get_settings())
        report['restored_settings'] = actual
        report['settings_mismatches'] = [key for key in expected if actual[key] != expected[key]]
        if report['settings_mismatches']:
            raise RuntimeError('清理核验期间世界设置改变：'+','.join(report['settings_mismatches']))
        report['verified'] = True

    def _save_cleanup_report(self, report, errors):
        report['errors'] = list(errors)
        report['status'] = 'failed' if errors else ('complete' if report.get('verified') else 'not_acquired')
        self.cleanup_report = report
        root = getattr(self, 'root', None)
        if root is not None:
            try:
                (Path(root)/'carla_cleanup.json').write_text(
                    json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
            except Exception as exc:
                errors.append('清理审计保存失败：'+str(exc))
                report['status'] = 'failed'
                report['errors'] = list(errors)
        self.cleanup_errors = list(errors)

    def close(self):
        if self.closed:
            return
        # join成功后才允许销毁演员和恢复原世界设置。
        errors = []
        report = {'version': 'carla-world-cleanup-1.0', 'verified': False,
                  'settings_change_attempted': self._settings_change_attempted,
                  'owned_actors': [], 'passive_wait_budget_seconds': 5.,
                  'passive_wait_seconds': 0., 'policy': '仅销毁本轮自建演员；核验不主动tick'}
        try:
            self.pause()
        except Exception as exc:
            errors.append('停止推进/终态快照：'+type(exc).__name__+': '+str(exc))
        if self._ticker is not None and self._ticker.is_alive():
            message = 'CARLA后台推进仍活跃；拒绝并发销毁传感器或恢复设置'
            errors.append(message)
            self._save_cleanup_report(report, errors)
            raise RuntimeError(message)
        # 暂停后的快照失败不能阻止已停止世界的资源清理。
        self.closed = True
        owned_ids = set()
        for actor in list(self.sensors)+list(getattr(self.experiment, 'actors', [])):
            try:
                owned_ids.add(int(actor.id))
                report['owned_actors'].append(self._actor_record(actor))
            except Exception as exc:
                errors.append('记录自建演员身份：'+str(exc))
        for sensor in reversed(self.sensors):
            try:
                sensor.stop()
            except Exception as exc:
                errors.append('sensor.stop：'+str(exc))
            try:
                if sensor.destroy() is False:
                    errors.append('sensor.destroy返回False')
            except Exception as exc:
                errors.append('sensor.destroy：'+str(exc))
        # 世界已经暂停，传感器已停止；只让已排队的事件回调归并，不额外tick。
        if self.evaluation_probe is not None:
            try:
                drain_started = time.monotonic()
                previous_count, quiet_since = len(self.events), drain_started
                while time.monotonic()-drain_started < .5:
                    time.sleep(.025)
                    count = len(self.events)
                    if count != previous_count:
                        previous_count, quiet_since = count, time.monotonic()
                    if time.monotonic()-quiet_since >= .15:
                        break
                self.evaluation_probe.ingest_events(list(self.events))
                if self._worker_error:
                    self.evaluation_probe.infrastructure_error('bridge_worker', self._worker_error)
                report['event_drain'] = {'elapsed_wall_s': time.monotonic()-drain_started,
                    'event_count': len(self.events), 'extra_physics_ticks': 0,
                    'policy': 'bounded_passive_callback_drain_after_sensor_stop'}
                self._evaluation = self.evaluation_probe.report(self._terminal_reason, save=True)
            except Exception as exc:
                errors.append('独立评价收尾：'+type(exc).__name__+': '+str(exc))
        if self.experiment is not None:
            try:
                self.experiment.close()
                errors.extend(str(event) for event in getattr(self.experiment, 'events', [])
                              if event.get('event') in ('cleanup_error', 'scenario_cleanup_error'))
            except Exception as exc:
                errors.append('experiment.close：'+str(exc))
        if self.world is not None and self.saved_settings is not None and self._settings_change_attempted:
            destroyed_after_frame = None
            try:
                destroyed_after_frame = int(self.world.get_snapshot().frame)
                report['destroyed_after_frame'] = destroyed_after_frame
            except Exception as exc:
                errors.append('记录销毁完成时世界帧：'+str(exc))
            try:
                self.world.apply_settings(self.saved_settings)
            except Exception as exc:
                errors.append('world.apply_settings：'+str(exc))
            if destroyed_after_frame is not None:
                try:
                    self._verify_world_cleanup(report, destroyed_after_frame, owned_ids)
                except Exception as exc:
                    errors.append('清理后状态核验：'+type(exc).__name__+': '+str(exc))
        self._save_cleanup_report(report, errors)
        if self.evaluation_probe is not None and errors:
            self.evaluation_probe.infrastructure_error('cleanup', list(errors))
            self._evaluation = self.evaluation_probe.report(self._terminal_reason, save=True)
        if errors:
            raise RuntimeError('CARLA 资源收尾异常，请检查独立世界：' + '; '.join(errors))


def main():
    session = CarlaSession()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                method, params = request['method'], request.get('params', {})
                if method == 'init':
                    result = session.initialize(**params)
                elif method == 'inspect_scene':
                    result = inspect_scene(**params)
                elif method == 'step':
                    result = session.step(**params)
                elif method == 'observe':
                    result = session.observe(**params)
                elif method == 'pause':
                    result = session.pause()
                elif method == 'close':
                    session.close()
                    result = {'closed': True}
                else:
                    raise ValueError('未知桥接操作：' + str(method))
                reply = {'id': request['id'], 'result': result}
            except Exception as exc:
                if session.evaluation_probe is not None:
                    session.evaluation_probe.infrastructure_error('bridge_'+str(request.get('method')),
                        type(exc).__name__+': '+str(exc))
                if request.get('method') == 'init':
                    session._save_initialization_failure(exc)
                reply = {'id': request.get('id'), 'error': '{}: {}'.format(type(exc).__name__, exc)}
            print(json.dumps(reply, ensure_ascii=False, allow_nan=False), flush=True)
            if request.get('method') == 'close' or 'error' in reply:
                break
    finally:
        try:
            session.close()
        except Exception as exc:
            print(str(exc), file=sys.stderr, flush=True)
        if session.cleanup_errors:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
