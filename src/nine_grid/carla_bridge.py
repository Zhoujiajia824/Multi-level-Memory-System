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
from pathlib import Path

try:
    from .carla_execution import ControlTimeline
except ImportError:
    from carla_execution import ControlTimeline


def validate_camera_source_clock(clock):
    """文件型模型输入只能使用本轮归档源帧，禁止高频调用时混入旧相机图。"""
    frequency = clock.get('fast_hz', 1)
    if isinstance(frequency, bool) or frequency != 1:
        raise ValueError('CARLA相机按1Hz归档，clock.fast_hz当前也必须为1；不复用旧图调用模型')


def camera_storage_due(sim_time, camera_save_hz=1):
    """从真实10Hz采集帧中选择1Hz原始图片归档，不影响显示/录像时钟。"""
    if isinstance(camera_save_hz, bool) or camera_save_hz != 1:
        raise ValueError('camera_save_hz当前必须为1，与快模型原帧采样对齐')
    if isinstance(sim_time, bool) or not isinstance(sim_time, (int, float)) or not math.isfinite(sim_time) or sim_time < -1e-5:
        raise ValueError('相机落盘需要有限非负仿真时间')
    # 只容忍物理时钟的浮点误差，不把相邻控制帧舍入成源帧。
    return abs(sim_time-round(sim_time)) <= 1e-5


def lane_key(waypoint):
    return '{}|{}|{}'.format(waypoint.road_id, waypoint.section_id, waypoint.lane_id)


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
    from carla_scenarios import select_route
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
    _, report = select_route(road_map,settings,carla,audit_all=True)
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


ENVIRONMENT_OBSTACLE_LABELS = ('Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle',
    'Vehicles', 'Static', 'Other', 'Fences', 'GuardRail', 'Walls', 'Poles', 'TrafficSigns', 'TrafficLight')


def environment_catalogue(world, carla):
    """只读取环境网格，不启停对象。0.9.15官方ObjectLabel.h区分Car/Truck等。

    get_environment_objects的框已经位于世界坐标，不能再乘EnvironmentObject.transform。
    在创建实验车辆之前读取，配合独立空世界准入避免把动态actor重复成静态物。
    不读取Building/Road/Sidewalk/Ground等大区域框；不以尺寸或数量截断障碍物。
    """
    labels = {getattr(carla.CityObjectLabel, name): name for name in ENVIRONMENT_OBSTACLE_LABELS
              if hasattr(carla.CityObjectLabel, name)}
    objects, seen, excluded, normalizations = [], {}, {}, []
    for item in sorted(world.get_environment_objects(carla.CityObjectLabel.Any), key=lambda obj: obj.id):
        if item.type not in labels:
            name = str(item.type)
            excluded[name] = excluded.get(name, 0) + 1
            continue
        box = item.bounding_box
        values = (box.location.x, box.location.y, box.location.z, box.extent.x, box.extent.y,
                  box.extent.z, box.rotation.yaw, box.rotation.pitch, box.rotation.roll)
        if not all(math.isfinite(value) for value in values) or any(value == 0 for value in values[3:6]):
            raise ValueError('环境障碍物框包含非有限值或零半尺寸：id={}，name={!r}，label={}，'
                'center={}，extent={}，rotation={}'.format(item.id, getattr(item, 'name', ''),
                    labels[item.type], values[:3], values[3:6], values[6:]))
        if any(value < 0 for value in values[3:6]):
            # 0.9.15直接将半尺寸乘组件缩放，镜像网格会保留负号。
            # ±半尺寸枚举的八角点集合不变；不删除障碍物，不修改原始地图框。
            extent = tuple(abs(value) for value in values[3:6])
            positive_box = carla.BoundingBox(
                carla.Location(x=values[0], y=values[1], z=values[2]),
                carla.Vector3D(x=extent[0], y=extent[1], z=extent[2]))
            positive_box.rotation = carla.Rotation(yaw=values[6], pitch=values[7], roll=values[8])
            normalizations.append({'id': str(item.id), 'name': getattr(item, 'name', ''),
                'label': labels[item.type], 'source_extent_m': list(values[3:6]),
                'canonical_extent_m': list(extent)})
            box = positive_box
            values = values[:3] + extent + values[6:]
        # 完全相同的环境几何可能有多个组件标签，仅合并精确几何重复，不近似删物体。
        signature = tuple(values)
        if signature in seen:
            seen[signature]['source_ids'].append(str(item.id))
            continue
        entry = {'id': 'environment_{}'.format(item.id), 'box': box, 'label': labels[item.type],
                 'source_ids': [str(item.id)]}
        seen[signature] = entry
        objects.append(entry)
    return objects, {'source': 'CARLA_0.9.15_get_environment_objects_world_bounding_boxes',
        'included_labels': sorted(labels.values()), 'excluded_label_counts': excluded,
        'catalogue_count': len(objects), 'loaded_before_experiment_actors': True,
        'geometry_version': 'carla-environment-box-1.1',
        'signed_extent_normalized_count': len(normalizations),
        'signed_extent_normalizations': normalizations,
        'scope': '已标注环境障碍物；排除建筑/道路/地面等区域语义，未缩小或关闭实体'}


class CarlaSession:
    def __init__(self):
        self.world = None
        self.saved_settings = None
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
        self._applied_key = None
        self._pending_plan = None
        self._visualization = None
        self._camera_payload_cache = None
        self._camera_audit_lock = threading.Lock()
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
            raise ValueError('RGB真实采样频率camera_hz当前必须为10，原始图片归档独立为1Hz')
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
        from carla_scenarios import create_experiment, hold_initial_motion, release_initial_motion
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
        existing = (list(self.world.get_actors().filter('vehicle.*')) +
                    list(self.world.get_actors().filter('walker.pedestrian.*')) +
                    list(self.world.get_actors().filter('static.prop.*')))
        if initial.synchronous_mode or existing:
            raise RuntimeError('拒绝接管正在同步运行或已有车辆/行人/脚本道具的世界；请为实验启动独立 CARLA 服务')
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
            config.get('acceleration_max_mps2', 3), config.get('steering_abs_max_rad', .6))
        self._configure_experiment_world(client, requested_map, config.get('allow_world_reset', False))
        self.environment_objects, self.environment_audit = environment_catalogue(self.world, carla)
        audit_path = self.root / 'carla_environment_audit.json'
        audit_path.write_text(json.dumps(self.environment_audit, ensure_ascii=False,
                                        allow_nan=False, indent=2), encoding='utf-8')
        # 全地图规范化明细只保存一次，避免每帧重复写入全部环境物体。
        self.environment_audit = {key: value for key, value in self.environment_audit.items()
                                  if key != 'signed_extent_normalizations'}
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
            # 首张RGB是启动采样，实测第一间隔为6帧、此后才为5帧。
            # 校准期间演员保持制动，未调用场景脚本、发布观测或开始实验计时。
            self._align_initial_camera_phase()
            # 下一稳态相机帧之前的四帧仍保持，最后一帧才激活初速。
            for _ in range(self.physics_hz // 10 - 1):
                self._initialization_tick('camera_phase_hold')
        self._initial_speed_audit = release_initial_motion(self.experiment, carla)
        snap = self._initialization_tick('initial_motion_activation')
        self._collect(self.frame)
        self._initial_source_frame = self.frame
        self.t0 = snap.timestamp.elapsed_seconds
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
        result['bridge_version'] = 'carla-0.9.15-range-runtime-3.2'
        result['sdk_version'] = client_version
        result['world_reset'] = copy.deepcopy(self.world_reset)
        result['initialization_warmup'] = copy.deepcopy(self._initialization_warmup)
        result['initial_speed_audit'] = copy.deepcopy(self._initial_speed_audit)
        result['rgb_initialization'] = copy.deepcopy(self._rgb_initialization)
        initialization = {'bridge_version': result['bridge_version'],
            'world_reset': result['world_reset'], 'initialization_warmup': result['initialization_warmup'],
            'initial_speed_audit': result['initial_speed_audit'], 'source_frame': self.frame, 't0': self.t0,
            'physics_hz': self.physics_hz, 'planner_hz': self.planner_hz,
            'rgb_enabled': self.rgb_enabled, 'camera_hz': 10., 'sensor_tick_s': .1,
            'rgb_initialization': copy.deepcopy(self._rgb_initialization),
            'initialization_ticks': copy.deepcopy(self._initialization_ticks),
            'initial_motion': copy.deepcopy(self.experiment.initial_motion_audit),
            'initialization_events': copy.deepcopy(self.events),
            'traffic_lights_at_source': [
                {'id': str(actor.id), 'state': str(actor.get_state()),
                 'elapsed_s': float(actor.get_elapsed_time())}
                for actor in self.world.get_actors().filter('traffic.traffic_light*')],
            'note': '先制动预热确认六路RGB稳态五帧相位，再于最后一帧激活初速并建立t0；'
                    '初始化推进与事件单列，交通灯按世界时间推进；运行中不补tick或混用旧图。'}
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

    def _align_initial_camera_phase(self):
        """测量启动相位，不假设创建后的首张图就是稳态10Hz时钟原点。"""
        period = self.physics_hz // 10
        audit = {'version': 'carla-rgb-startup-phase-1.0', 'status': 'calibrating',
            'expected_period_frames': period, 'required_stable_intervals': 2,
            'sample_frames': [], 'intervals_frames': [], 'initialization_only': True}
        self._rgb_initialization = audit
        self._rgb_initialization_deadline = time.monotonic()+self.timeout
        try:
            self._initialization_tick('camera_bootstrap')
            self._collect(self.frame)
            audit['sample_frames'].append(self.frame)
            previous, stable = self.frame, 0
            # 最多0.6s初始化物理推进；每帧留出回调时间，不在缺图帧冻结30s。
            # 仅用于启动相位探测；运行期仍严格等待指定当前帧，不继续推进找图。
            front_key = next(iter(self.queues))
            for _ in range(6*period):
                self._initialization_tick('camera_phase_probe')
                deadline = min(time.monotonic()+.2, self._rgb_initialization_deadline)
                image = None
                while time.monotonic() < deadline:
                    try:
                        item = self.queues[front_key].get(timeout=max(0., deadline-time.monotonic()))
                    except queue.Empty:
                        break
                    if item.frame < self.frame:
                        self._record_camera_event(front_key, 'discarded_startup_delayed', item)
                        continue
                    image = item
                    break
                if image is None:
                    continue
                self._collect(self.frame, prefetched={front_key: image})
                gap = self.frame-previous
                audit['sample_frames'].append(self.frame)
                audit['intervals_frames'].append(gap)
                stable = stable+1 if gap == period else 0
                previous = self.frame
                if stable >= 2:
                    audit.update(status='aligned', stable_frame=self.frame,
                        stable_timestamp=float(image.timestamp))
                    return
            raise RuntimeError('RGB初始化未形成六相机共同的10Hz五帧节拍；实际间隔{}'.format(
                audit['intervals_frames']))
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
                               ('fov', self.fov), ('sensor_tick', .1)):
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
            'markings': [str(mark.type) for mark in event.crossed_lane_markings]}))

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
        report = {'version': 'carla-rgb-sync-diagnostic-1.0',
            'error': type(error).__name__+': '+str(error), 'requested_frame': frame,
            'bridge_frame': self.frame, 'sim_time': self._sim_time,
            't0': self.t0, 'runtime_metrics': copy.deepcopy(self._runtime_metrics),
            'requested_camera_hz': 10., 'sensor_tick_s': .1,
            'accepted_for_request': accepted,
            'cameras': cameras, 'world_reset': copy.deepcopy(self.world_reset),
            'initialization_warmup': copy.deepcopy(self._initialization_warmup),
            'initial_speed_audit': copy.deepcopy(self._initial_speed_audit),
            'rgb_initialization': copy.deepcopy(self._rgb_initialization),
            'initialization_ticks': copy.deepcopy(self._initialization_ticks),
            'initial_motion': copy.deepcopy(getattr(self.experiment, 'initial_motion_audit', None)),
            'initialization_events': copy.deepcopy(self.events) if self.t0 is None else [],
            'note': '相机帧与世界帧严格对应；初始化相位推进单独记录，运行期不追加补图tick。'}
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
        if self.frames and all(image.frame == frame for image in self.frames.values()):
            return
        started = time.monotonic()
        result = {}
        prefetched = dict(prefetched or {})
        try:
            deadline = time.monotonic() + self.timeout
            if self.t0 is None and self._rgb_initialization_deadline is not None:
                deadline = min(deadline, self._rgb_initialization_deadline)
            for key, q in self.queues.items():
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('10Hz RGB同步超时：{}，目标帧{}'.format(key, frame))
                    try:
                        if self._stop.is_set() and threading.current_thread() is self._ticker:
                            raise RuntimeError('RGB等待被运行暂停中断')
                        item = prefetched.pop(key) if key in prefetched else q.get(timeout=min(remaining, .1))
                    except queue.Empty:
                        continue
                    if item.frame == frame:
                        result[key] = item
                        self._record_camera_event(key, 'selected', item)
                        break
                    if item.frame > frame:
                        raise RuntimeError('RGB源帧丢失：{}需要{}收到{}'.format(key, frame, item.frame))
                    self._record_camera_event(key, 'discarded_stale', item)
            timestamps = [item.timestamp for item in result.values()]
            if max(timestamps) - min(timestamps) > 1e-6:
                raise RuntimeError('同帧RGB时间戳不一致')
            snap = self.world.get_snapshot()
            if snap.frame != frame or any(abs(value-snap.timestamp.elapsed_seconds) > 1e-5 for value in timestamps):
                raise RuntimeError('RGB与目标世界帧时间戳不一致；拒绝旧图配新状态')
            self.frames = result
            self._camera_payload_cache = None
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
                                       self.cfg.get('camera_save_hz', 1))
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
        kind = 'pedestrian' if actor.type_id.startswith('walker.') else 'vehicle'
        if actor.type_id.startswith('static.'):
            kind = 'static_obstacle'
        elif actor.type_id in ('vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets'):
            kind = 'bicycle'
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
        return {'id': entry['id'], 'kind': 'static_obstacle', 'position': xy(box.location),
            'velocity': [0., 0.], 'acceleration': [0., 0.], 'heading': -math.radians(box.rotation.yaw),
            'length': 2.*ex, 'width': 2.*ey, 'height_m': max(p.z for p in vertices)-min(p.z for p in vertices),
            'lane_id': lane_key(waypoint) if waypoint is not None else None, 'visible': True,
            'radius': None, 'z': box.location.z, 'origin_z': box.location.z,
            'environment_label': entry['label'], 'environment_ids': entry['source_ids'],
            'state_sources': {'position': 'environment_bbox_world_center',
                              'geometry': 'yaw_aligned_envelope_of_world_bbox_xy_projection',
                              'velocity': 'fixed_map_mesh_zero', 'acceleration': 'fixed_map_mesh_zero'}}

    def _range_environment(self, ego, radius):
        result = []
        for entry in self.environment_objects:
            box = entry['box']
            if math.hypot(box.location.x-ego['position'][0], -box.location.y-ego['position'][1]) > radius + math.sqrt(box.extent.x**2+box.extent.y**2+box.extent.z**2):
                continue
            # 垂直区间重叠而非中心高度差，避免高护栏/立柱被错误排除。
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
                    boundaries.append({'id': '{}:{}:{}'.format(key, side, i), 'points': coordinates[i:i + 2],
                        'line_type': str(mark.type), **semantics})
            middle = waypoints[len(waypoints) // 2]
            adjacent = {}
            for side in ('left', 'right'):
                other = getattr(middle, 'get_' + side + '_lane')()
                same = other is not None and other.lane_type == self.carla.LaneType.Driving
                if same:
                    dyaw = math.radians(other.transform.rotation.yaw - middle.transform.rotation.yaw)
                    same = math.cos(dyaw) > 0.5
                adjacent[side] = lane_key(other) if same and lane_key(other) in groups else None
            successors = sorted(set(lane_key(w) for w in waypoints[-1].next(float(self.cfg.get('map_resolution_m', 2.0)) * 2)
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
        for actor in self.world.get_actors():
            if actor.id == self.ego.id or not (actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.pedestrian.') or actor in self.experiment.actors and actor.type_id.startswith('static.')):
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
        self._entities_cache = (self.frame, ego, actors, len(static_actors))
        return ego, actors, len(static_actors)

    def _sample_history(self, force=False):
        elapsed = self.world.get_snapshot().timestamp.elapsed_seconds-self.t0
        index = int(round(elapsed*2.))
        if not force and (abs(elapsed-index*.5) > 1e-5 or index <= self._history_sample_index):
            return
        started = time.monotonic()
        ego, actors, _ = self._entities()
        for actor in [ego]+actors:
            if actor['kind'] == 'static_obstacle':
                continue
            sample = {'time': float(elapsed), 'position': list(actor['position']),
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
            rgb_enabled=self.rgb_enabled, depth_sensor_count=0,
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
            'collision': self.collision, 'events': list(self.events) + list(self.experiment.events),
            'actual_actuation': {'throttle': actual.throttle, 'brake': actual.brake, 'steer_normalized_right': actual.steer},
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
            'scenario': self.experiment.name}

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

    def _install(self, controls, control_step_s, source_time, plan, source_speed_mps):
        if self.closed or self._terminal_reason is not None:
            raise RuntimeError('CARLA执行已终止：{}'.format(self._terminal_reason or 'closed'))
        if (type(control_step_s) not in (int, float) or not math.isfinite(control_step_s) or
                abs(control_step_s-self.control_step_s) > 1e-8):
            raise ValueError('控制网格必须保持0.1秒')
        # 校验完整批次，失败不能损坏当前正在执行的序列。
        candidate = ControlTimeline(self.timeline.minimum, self.timeline.maximum, self.timeline.steering_max)
        if source_speed_mps is None:
            raise ValueError('必须传递规划原帧自车速度，不按每段实测速误差重新生成目标速度')
        candidate.install(controls, source_time, control_step_s, source_speed_mps)
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
            state = candidate.at(self._sim_time)
            if not state['available']:
                raise ValueError('新控制序列无法用于当前仿真时间：'+state['reason'])
            candidate.revision = self.timeline.revision+1
            self.timeline = candidate
            self._runtime_metrics['submitted_sequences'] = self._runtime_metrics.get('submitted_sequences', 0)+1
            self._runtime_metrics['last_submission_skipped_segments'] = state['index']
            self._pending_plan = (candidate.revision, copy.deepcopy(plan))
            self._paused = False

    def _apply_current(self):
        with self._condition:
            state = self.timeline.at(self._sim_time)
            if not state['available']:
                self._terminal_reason = state['reason']
                if state['reason'] == 'sequence_exhausted':
                    self._worker_error = 'RuntimeError: MPC控制序列耗尽；未补写控制、未保持末项推进'
                return False
            pending = self._pending_plan
            self._pending_plan = None
            source_time = self.timeline.source_time
        if pending is not None:
            self._visualization = self._show_plan(pending[1])
            self._visualization['sequence_revision'] = pending[0]
        key = (state['revision'], state['index'])
        if key != self._applied_key:
            acceleration, steering = state['control']
            velocity = self.ego.get_velocity()
            speed = math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
            # 源时标中途接纳的段只执行剩余部分，过期段不重放。
            remaining = min(self.control_step_s,
                source_time+(state['index']+1)*self.control_step_s-self._sim_time)
            # CARLA0.9.15 Ackermann控制器将speed作为跟踪目标，acceleration的绝对值
            # 是内部速度PID的加速度上限，并非直接加速度命令。每段目标速度必须
            # 从源帧按已规划a序列累计，不能每段重新锚定滞后的实测速。
            desired = state['target_speed_mps']
            steering_rate = float(self.cfg.get('steering_rate_abs_max_radps', .8))
            command = self.carla.VehicleAckermannControl(steer=-steering, speed=desired,
                acceleration=abs(acceleration), steer_speed=steering_rate)
            self.ego.apply_ackermann_control(command)
            self._last_command = {'acceleration_mps2': acceleration, 'steering_rad_left': steering,
                'speed_mps_ackermann': desired, 'measured_speed_mps': speed,
                'speed_error_mps': desired-speed, 'ackermann_acceleration_limit_mps2': abs(acceleration),
                'ackermann_steering_rate_rps': steering_rate,
                'actuation_mapping': 'source_speed_plus_planned_acceleration_integral-1.0'}
            self._last_applied_control = dict(self._last_command, sequence_index=state['index'],
                sequence_revision=state['revision'], source_time=source_time,
                applied_time=self._sim_time, frame=self.frame,
                control=[acceleration, steering], remaining_segment_s=remaining)
            self._applied_key = key
        return True

    def _tick_once(self):
        previous_frame = self.world.get_snapshot().frame
        if previous_frame != self.frame:
            raise RuntimeError('本次tick之前世界已被额外推进：桥接帧{}，世界帧{}；只能有一个tick客户端'.format(
                self.frame, previous_frame))
        started = time.monotonic()
        self.experiment.before_tick(self._sim_time, 1./self.physics_hz)
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
        if self.rgb_enabled and self._at_frequency(10.):
            self._collect(self.frame)
            # 录像/观看独立消费10Hz；模型及归档仍只消费整秒原帧。
            if self._display_enabled or camera_storage_due(self._sim_time):
                self._queue_display_frame(self._camera_payload())

    def _queue_display_frame(self, cameras):
        if not cameras or not self._display_enabled:
            return
        packet = {'frame': self.frame, 'time': self._sim_time, 'cameras': cameras,
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
        phase = self._sim_time*frequency
        return abs(phase-round(phase)) < 1e-4

    def _duration_reached(self):
        with self._condition:
            if self.duration_s is not None and self._sim_time >= self.duration_s-1e-8:
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

    def step(self, controls, control_step_s, source_time, plan=None, source_speed_mps=None):
        self._install(controls, control_step_s, source_time, plan, source_speed_mps)
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

    def close(self):
        if self.closed:
            return
        # join成功后才允许销毁演员和恢复原世界设置。
        errors = []
        try:
            self.pause()
        except Exception as exc:
            errors.append('停止推进/终态快照：'+type(exc).__name__+': '+str(exc))
        if self._ticker is not None and self._ticker.is_alive():
            message = 'CARLA后台推进仍活跃；拒绝并发销毁传感器或恢复设置'
            self.cleanup_errors = errors+[message]
            raise RuntimeError(message)
        # 暂停后的快照失败不能阻止已停止世界的资源清理。
        self.closed = True
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
        if self.experiment is not None:
            try:
                self.experiment.close()
                errors.extend(str(event) for event in getattr(self.experiment, 'events', [])
                              if event.get('event') == 'cleanup_error')
            except Exception as exc:
                errors.append('experiment.close：'+str(exc))
        if self.world is not None and self.saved_settings is not None:
            try:
                self.world.apply_settings(self.saved_settings)
            except Exception as exc:
                errors.append('world.apply_settings：'+str(exc))
        if errors:
            self.cleanup_errors = errors
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
