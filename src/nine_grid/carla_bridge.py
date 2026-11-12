"""CARLA 0.9.15 独立 SDK 桥；兼容 Python 3.7，仅依赖标准库及 carla。"""
import base64
import json
import hashlib
import math
import os
import queue
import signal
import sys
import time
import zlib
from pathlib import Path


def validate_camera_source_clock(clock):
    """文件型模型输入只能使用本轮归档源帧，禁止高频调用时混入旧相机图。"""
    frequency = clock.get('fast_hz', 1)
    if isinstance(frequency, bool) or frequency != 1:
        raise ValueError('CARLA相机按1Hz归档，clock.fast_hz当前也必须为1；不复用旧图调用模型')


def camera_storage_due(sim_time, camera_save_hz=1):
    """RGB资产只落盘整秒原帧；10Hz规划/录像仍消费各自当前帧。"""
    if isinstance(camera_save_hz, bool) or camera_save_hz != 1:
        raise ValueError('camera_save_hz当前必须为1，与快模型原帧采样对齐')
    if isinstance(sim_time, bool) or not isinstance(sim_time, (int, float)) or not math.isfinite(sim_time) or sim_time < -1e-5:
        raise ValueError('相机落盘需要有限非负仿真时间')
    # 只容忍物理时钟的浮点误差，不把相邻控制帧舍入成源帧。
    return abs(sim_time-round(sim_time)) <= 1e-5


def lane_key(waypoint):
    return '{}|{}|{}'.format(waypoint.road_id, waypoint.section_id, waypoint.lane_id)


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


def project_camera(point, inverse, width, height, fov):
    """UE 相机 x 前 y 右 z 上，返回像素及光轴深度，非欧氏射线长度。"""
    p = [point.x, point.y, point.z, 1.0]
    v = [sum(inverse[i][j] * p[j] for j in range(4)) for i in range(3)]
    if v[0] <= 0.05:
        return None
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    u, w = width / 2.0 + focal * v[1] / v[0], height / 2.0 - focal * v[2] / v[0]
    if not (1 <= u < width - 1 and 1 <= w < height - 1):
        return None
    return int(u), int(w), v[0]


def encoded_depth(image, x, y):
    i = (y * image.width + x) * 4
    raw = image.raw_data
    return (raw[i + 2] + 256 * raw[i + 1] + 65536 * raw[i]) * (1000.0 / 16777215.0)


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

    def initialize(self, config, directory, seed):
        if config.get('exclusive_world') is not True:
            raise ValueError('必须显式确认 exclusive_world=true，仅使用本任务独立 CARLA 世界')
        if type(config.get('world_plan_debug',False)) is not bool:
            raise ValueError('world_plan_debug必须为布尔值')
        camera_storage_due(0., config.get('camera_save_hz', 1))
        if type(config.get('rgb_postprocess_effects', True)) is not bool:
            raise ValueError('rgb_postprocess_effects必须为布尔值')
        for key, default in (('timeout_s', 30), ('map_resolution_m', 2), ('same_level_tolerance_m', 1.5),
                             ('perception_radius_m', 120), ('visibility_surface_tolerance_m', .35),
                             ('camera_height_m', 2.2), ('map_speed_cap_mps', 30)):
            value = config.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError('CARLA 参数必须为有限正数：' + key)
        if type(config.get('visibility_min_samples', 2)) is not int or not 1 <= config.get('visibility_min_samples', 2) <= 36:
            raise ValueError('visibility_min_samples 必须在 1 至 36 之间')
        import carla
        from carla_scenarios import create_experiment
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
        if self.world.get_map().name.rstrip('/').split('/')[-1] != requested_map.rstrip('/').split('/')[-1]:
            if not config.get('allow_world_reset', False):
                raise RuntimeError('当前地图与配置不符；换图必须显式允许 allow_world_reset')
            self.world = client.load_world(requested_map)
        self.saved_settings = self.world.get_settings()
        settings = self.world.get_settings()
        self.physics_hz = int(config.get('physics_hz', 50))
        self.planner_hz = int(config.get('planner_hz', 10))
        if self.physics_hz <= 0 or self.planner_hz <= 0 or self.physics_hz % self.planner_hz:
            raise ValueError('CARLA 物理频率必须为规划频率的整数倍')
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.physics_hz
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = max(1, int(math.ceil(settings.fixed_delta_seconds / 0.01)))
        self.world.apply_settings(settings)
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
        self.frame = self.world.tick(self.timeout)
        snap = self.world.get_snapshot()
        self.t0 = snap.timestamp.elapsed_seconds
        self._collect(self.frame)
        result = self.snapshot()
        result['map'] = self._map_payload()
        result['bridge_version'] = 'carla-0.9.15-observed-1.0'
        result['sdk_version'] = client_version
        return result

    def _create_sensors(self):
        c = self.carla
        views = [('front', 0), ('front_left', -60), ('front_right', 60),
                 ('back', 180), ('back_left', -120), ('back_right', 120)]
        self.camera_actors = {}
        for view, yaw in views:
            for kind in ('rgb', 'depth'):
                bp = self.world.get_blueprint_library().find('sensor.camera.' + kind)
                for key, value in (('image_size_x', self.width), ('image_size_y', self.height),
                                   ('fov', self.fov), ('sensor_tick', 0.0)):
                    bp.set_attribute(key, str(value))
                if bp.has_attribute('enable_postprocess_effects'):
                    # CARLA关闭RGB后处理同时会关闭光照并使用线性gamma；恢复原生RGB默认管线。
                    # 深度是几何证据，始终禁用后处理；不额外调色或覆盖默认gamma/曝光参数。
                    enabled = kind == 'rgb' and self.cfg.get('rgb_postprocess_effects', True)
                    bp.set_attribute('enable_postprocess_effects', 'true' if enabled else 'false')
                pose = c.Transform(c.Location(z=float(self.cfg.get('camera_height_m', 2.2))), c.Rotation(yaw=yaw))
                sensor = self.world.spawn_actor(bp, pose, attach_to=self.ego, attachment_type=c.AttachmentType.Rigid)
                self.sensors.append(sensor)
                key = view + '_' + kind
                q = queue.Queue(maxsize=4)
                self.queues[key] = q
                def callback(image, target=q):
                    try:
                        target.put_nowait(image)
                    except queue.Full:
                        try:
                            target.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            target.put_nowait(image)
                        except queue.Full:
                            pass
                sensor.listen(callback)
                self.camera_actors[key] = sensor
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor = self.world.spawn_actor(bp, c.Transform(), attach_to=self.ego)
        self.sensors.append(sensor)
        def collision(event):
            self.collision = True
            self.events.append({'kind': 'collision', 'frame': event.frame,
                                'other_id': event.other_actor.id, 'other_type': event.other_actor.type_id})
        sensor.listen(collision)
        bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        sensor = self.world.spawn_actor(bp, c.Transform(), attach_to=self.ego)
        self.sensors.append(sensor)
        sensor.listen(lambda event: self.events.append({'kind': 'lane_invasion', 'frame': event.frame,
            'markings': [str(mark.type) for mark in event.crossed_lane_markings]}))

    def _collect(self, frame):
        deadline = time.monotonic() + self.timeout
        result = {}
        for key, q in self.queues.items():
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('相机同步超时：{}，目标帧{}'.format(key, frame))
                try:
                    item = q.get(timeout=remaining)
                except queue.Empty:
                    raise TimeoutError('相机同步超时：{}，目标帧{}'.format(key, frame))
                if item.frame == frame:
                    result[key] = item
                    break
                if item.frame > frame:
                    raise RuntimeError('相机丢帧：{}需要{}收到{}'.format(key, frame, item.frame))
        timestamps = [item.timestamp for item in result.values()]
        if max(timestamps) - min(timestamps) > 1e-6:
            raise RuntimeError('同帧相机时间戳不一致')
        if self.world is not None:
            snap = self.world.get_snapshot()
            if snap.frame != frame or any(abs(value-snap.timestamp.elapsed_seconds) > 1e-5 for value in timestamps):
                raise RuntimeError('相机与目标世界帧时间戳不一致')
        self.frames = result

    def _camera_payload(self):
        folder = self.root / 'camera_frames' / '{:08d}'.format(self.frame)
        if self.t0 is None:
            raise ValueError('相机落盘缺少本轮起始仿真时间')
        saved = camera_storage_due(float(self.frames['front_rgb'].timestamp)-self.t0,
                                   self.cfg.get('camera_save_hz', 1))
        cameras = []
        for view in ('front', 'front_left', 'front_right', 'back', 'back_left', 'back_right'):
            rgb, depth = self.frames[view + '_rgb'], self.frames[view + '_depth']
            if (rgb.frame != self.frame or depth.frame != self.frame or
                    not all(math.isfinite(t) for t in (rgb.timestamp, depth.timestamp)) or
                    abs(rgb.timestamp-depth.timestamp) > 1e-6):
                raise ValueError('RGB与临时深度必须来自同一帧和时间戳：'+view)
            if (rgb.width != self.width or depth.width != self.width or
                    rgb.height != self.height or depth.height != self.height):
                raise ValueError('RGB与临时深度尺寸不符合相机配置：'+view)
            depth_bytes = bytes(depth.raw_data)
            if len(depth_bytes) != self.width*self.height*4:
                raise ValueError('临时深度BGRA字节数不符合相机尺寸：'+view)
            rgb_bytes = bytes(rgb.raw_data)
            if len(rgb_bytes) != self.width*self.height*4:
                raise ValueError('临时RGB BGRA字节数不符合相机尺寸：'+view)
            rgb_path = folder / (view + '.png')
            if saved and not rgb_path.is_file():
                folder.mkdir(parents=True, exist_ok=True)
                rgb.save_to_disk(str(rgb_path))
            # 当前RGB始终传输，非落盘帧也用于当前视频/显示，绝不重复上一整秒图片。
            rgb_transport = {'encoding': 'zlib-base64-bgra8-1.0', 'frame': int(rgb.frame),
                'timestamp': float(rgb.timestamp), 'width': self.width, 'height': self.height,
                'data': base64.b64encode(zlib.compress(rgb_bytes, 1)).decode('ascii')}
            # 深度只用于同帧可见性/观测证据，通过内存RPC交付，不生成深度图片。
            transport = {'encoding': 'zlib-base64-bgra8-1.0', 'frame': int(depth.frame),
                'timestamp': float(depth.timestamp), 'width': self.width, 'height': self.height,
                'data': base64.b64encode(zlib.compress(depth_bytes, 1)).decode('ascii')}
            cameras.append({'view': view, 'rgb_path': str(rgb_path) if saved else None,
                'rgb_saved': saved, 'rgb_transport': rgb_transport, 'depth_transport': transport,
                'rgb_postprocess_effects': self.cfg.get('rgb_postprocess_effects', True),
                'width': self.width, 'height': self.height, 'fov_deg': self.fov,
                'frame': self.frame, 'timestamp': rgb.timestamp,
                'world_to_camera_ue': rgb.transform.get_inverse_matrix(),
                'camera_to_world_ue': rgb.transform.get_matrix(),
                'depth_frame': int(depth.frame), 'depth_timestamp': float(depth.timestamp),
                'depth_retention': 'runtime_only_not_saved',
                'depth_encoding': 'carla_rgb24_planar_m;R+256G+65536B;far=1000'})
        return cameras

    def _visible(self, actor):
        """只访问候选车体几何，使用同帧深度验证朝向相机的表面采样。"""
        return self._box_visible(actor.bounding_box, actor.get_transform())

    def _box_visible(self, bbox, transform):
        """环境框传入单位变换；actor局部框传入actor世界变换。"""
        # 面中心与内部采样避免仅车框角点露出天空导致误判。
        samples = []
        for axis in (0, 1):
            for side in (-1, 1):
                for horizontal in (-0.5, 0, 0.5):
                    for vertical in (-0.35, 0, 0.35):
                        p = [horizontal * bbox.extent.x, horizontal * bbox.extent.y, vertical * bbox.extent.z]
                        p[axis] = side * (bbox.extent.x if axis == 0 else bbox.extent.y)
                        local = self.carla.Location(x=p[0], y=p[1], z=p[2])
                        box_transform = self.carla.Transform(bbox.location, bbox.rotation)
                        point = box_transform.transform(local)
                        samples.append(transform.transform(point))
        tolerance = float(self.cfg.get('visibility_surface_tolerance_m', 0.35))
        minimum = int(self.cfg.get('visibility_min_samples', 2))
        for view in ('front', 'front_left', 'front_right', 'back', 'back_left', 'back_right'):
            depth = self.frames[view + '_depth']
            inverse = depth.transform.get_inverse_matrix()
            evidence = set()
            for point in samples:
                projected = project_camera(point, inverse, self.width, self.height, self.fov)
                if projected is None:
                    continue
                u, v, axial = projected
                observed = encoded_depth(depth, u, v)
                if abs(observed - axial) <= tolerance:
                    evidence.add((u, v))
            if len(evidence) >= minimum:
                return True
        return False

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

    def _visible_environment(self, ego, radius):
        result = []
        for entry in self.environment_objects:
            box = entry['box']
            if math.hypot(box.location.x-ego['position'][0], -box.location.y-ego['position'][1]) > radius + math.sqrt(box.extent.x**2+box.extent.y**2+box.extent.z**2):
                continue
            # 垂直区间重叠而非中心高度差，避免高护栏/立柱被错误排除。
            vertices = box.get_world_vertices(self.carla.Transform())
            if max(p.z for p in vertices) < ego['z']-ego['height_m']/2 or min(p.z for p in vertices) > ego['z']+ego['height_m']/2:
                continue
            if self._box_visible(box, self.carla.Transform()):
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
                    permitted = bool(int(permission) & int(self.carla.LaneChange.Left if side == 'left' else self.carla.LaneChange.Right))
                    boundaries.append({'id': '{}:{}:{}'.format(key, side, i), 'points': coordinates[i:i + 2],
                        'crossable': permitted, 'kind': 'lane_boundary', 'line_type': str(mark.type), 'permission_source': 'current_waypoint_lane_change'})
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
                    'scope':'地图坡度延伸；当前对象的可见性和高度准入保持原有规则'},
                'crosswalks': closed_crosswalks(self.map.get_crosswalks(), base_z, tolerance,elevation),
                'lane_speed_source': 'configured_map_speed_cap_mps;current_ego_limit_from_CARLA'}

    def snapshot(self):
        snap = self.world.get_snapshot()
        if snap.frame != self.frame:
            raise RuntimeError('检测到其他客户端推进世界；拒绝使用状态与图像不同步的帧')
        ego = self._actor(self.ego, True)
        radius = float(self.cfg.get('perception_radius_m', 120.0))
        actors = []
        for actor in self.world.get_actors():
            if actor.id == self.ego.id or not (actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.pedestrian.') or actor in self.experiment.actors and actor.type_id.startswith('static.')):
                continue
            center = bounding_box_world_center(actor.get_transform(),actor.bounding_box,self.carla)
            if abs(center.z - ego['z']) > float(self.cfg.get('same_level_tolerance_m', 1.5)):
                continue
            if math.hypot(center.x - ego['position'][0], -center.y - ego['position'][1]) > radius:
                continue
            if self._visible(actor):
                actors.append(self._actor(actor))
        static_actors = self._visible_environment(ego, radius)
        actors.extend(static_actors)
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
        cameras = self._camera_payload()
        weather = self.world.get_weather()
        weather_record = {name: float(getattr(weather, name)) for name in (
            'cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity',
            'sun_azimuth_angle', 'sun_altitude_angle', 'fog_density', 'fog_distance',
            'wetness', 'fog_falloff', 'scattering_intensity', 'mie_scattering_scale',
            'rayleigh_scattering_scale', 'dust_storm') if hasattr(weather, name)}
        if self.world.get_snapshot().frame != self.frame:
            raise RuntimeError('读取场景期间世界被其他客户端推进；拒绝输出不同步场景')
        return {'frame': self.frame, 'time': snap.timestamp.elapsed_seconds - self.t0,
            'timestamp': snap.timestamp.elapsed_seconds, 'ego': ego, 'actors': actors,
            'cameras': cameras, 'traffic_lights': lights,
            'speed_limit_mps': self.ego.get_speed_limit() / 3.6,
            'collision': self.collision, 'events': list(self.events) + list(self.experiment.events),
            'actual_actuation': {'throttle': actual.throttle, 'brake': actual.brake, 'steer_normalized_right': actual.steer},
            'current_road_evidence': road_evidence,
            'environment_obstacles': dict(self.environment_audit, visible_count=len(static_actors)),
            'weather': weather_record, 'weather_source': ('scenario_weather_then_current_world_get_weather'
                if self.cfg.get('experiment',{}).get('weather') is not None else 'current_world_get_weather_no_override'),
            'source': 'current_camera_depth_visible_geometry_then_actor_state',
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

    def step(self, control, plan):
        if len(control) != 2 or not all(math.isfinite(float(v)) for v in control):
            raise ValueError('控制必须是有限的加速度和转角')
        acceleration, steering = map(float, control)
        if not float(self.cfg.get('acceleration_min_mps2', -6)) <= acceleration <= float(self.cfg.get('acceleration_max_mps2', 3)) or abs(steering) > float(self.cfg.get('steering_abs_max_rad', 0.6)) + 1e-8:
            raise ValueError('CARLA 控制超出规划物理边界')
        velocity = self.ego.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        desired = max(0.0, speed + acceleration / self.planner_hz)
        command = self.carla.VehicleAckermannControl(steer=-steering, speed=desired, acceleration=acceleration)
        self.ego.apply_ackermann_control(command)
        # 世界线仅在显式开关下绘制，寿命略长于一控制周期，避免永久残留。
        visualization = self._show_plan(plan)
        for _ in range(self.physics_hz // self.planner_hz):
            elapsed = self.world.get_snapshot().timestamp.elapsed_seconds - self.t0
            self.experiment.before_tick(elapsed, 1.0 / self.physics_hz)
            self.frame = self.world.tick(self.timeout)
        self._collect(self.frame)
        result = self.snapshot()
        visualization['capture_carla_frame'] = self.frame
        result['plan_visualization'] = visualization
        result['command'] = {'acceleration_mps2': acceleration, 'steering_rad_left': steering,
                             'speed_mps_ackermann': desired}
        measured = math.hypot(*result['ego']['velocity'])
        result['control_response'] = {'speed_error_mps': measured - desired,
            'mean_acceleration_mps2': (measured - speed) * self.planner_hz,
            'requested_acceleration_mps2': acceleration, 'executor': 'CARLA_builtin_Ackermann_PID'}
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        for sensor in reversed(self.sensors):
            try:
                sensor.stop()
            except Exception as exc:
                errors.append(str(exc))
            try:
                if sensor.destroy() is False:
                    errors.append('sensor.destroy返回False')
            except Exception as exc:
                errors.append(str(exc))
        if self.experiment is not None:
            try:
                self.experiment.close()
                errors.extend(str(event) for event in getattr(self.experiment, 'events', [])
                              if event.get('event') == 'cleanup_error')
            except Exception as exc:
                errors.append(str(exc))
        if self.world is not None and self.saved_settings is not None:
            try:
                self.world.apply_settings(self.saved_settings)
            except Exception as exc:
                errors.append(str(exc))
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
                elif method == 'close':
                    session.close()
                    result = {'closed': True}
                else:
                    raise ValueError('未知桥接操作：' + str(method))
                reply = {'id': request['id'], 'result': result}
            except Exception as exc:
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
