"""CARLA SDK 子进程到平台无关场景契约；主环境不导入 carla。"""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import queue
import subprocess
import threading
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from .contracts import Actor, Lane, SceneSnapshot, plain
from .paths import filesystem_path
from .carla_bridge import camera_storage_due
from .carla_execution import duration_tick_limit
from .carla_public_priority import PublicPriorityTracker
from .carla_boundary_scope import scope_junction_boundaries
from .geometry import (actor_shape, clean_polyline, polyline_lengths, project_polyline,
                       sample_polyline, checked_route_projection, wrap_angle)
from .object_scope import scope_scene


class BridgeProcess:
    """有限等待的 JSON-lines 请求；EOF 允许 SDK 桥执行 finally 清理。"""
    def __init__(self, python, output_dir, timeout_s=30., bridge_script=None):
        self.timeout = float(timeout_s)
        self.queue = queue.Queue()
        self.counter = 0
        self.closed = False
        self.failed = False
        self.last_error = None
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
        if self.failed:
            raise RuntimeError('CARLA 桥已失败，不能继续提交操作；等待独立进程清理')
        self.counter += 1
        identity = self.counter
        try:
            self.process.stdin.write(json.dumps({'id': identity, 'method': method, 'params': params},
                                               ensure_ascii=False, allow_nan=False) + '\n')
            self.process.stdin.flush()
            response = self.queue.get(timeout=self.timeout)
        except (OSError, queue.Empty) as exc:
            self.failed = True
            self.last_error = {'method': method, 'error': type(exc).__name__}
            raise RuntimeError('CARLA 桥通信失败或超时；详见 carla_bridge.log') from exc
        if response is None:
            self.failed = True
            self.last_error = {'method': method, 'error': 'SDK桥提前退出'}
            raise RuntimeError('CARLA SDK 桥提前退出；请检查 SDK 环境与 carla_bridge.log')
        try:
            result = json.loads(response)
        except ValueError as exc:
            self.failed = True
            self.last_error = {'method': method, 'error': 'SDK桥响应JSON无效'}
            raise RuntimeError('CARLA 桥输出不是有效 JSON') from exc
        if not isinstance(result, dict) or result.get('id') != identity:
            self.failed = True
            self.last_error = {'method': method, 'error': 'SDK桥响应序号不一致'}
            raise RuntimeError('CARLA 桥响应序号不一致，拒绝混用迟到结果')
        if 'error' in result:
            self.failed = True
            self.last_error = {'method': method, 'error': result['error']}
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


def _camera_bgra(camera, key, label):
    """严格解码单帧临时图像；控制长度并检查标定、帧号及时间戳。"""
    payload = camera[key]
    if not isinstance(payload, dict) or payload.get('encoding') != 'zlib-base64-bgra8-1.0':
        raise ValueError('临时'+label+'编码无效')
    width, height = camera.get('width'), camera.get('height')
    if any(type(v) is not int or v <= 0 for v in (width,height)):
        raise ValueError('临时'+label+'相机尺寸无效')
    if (payload.get('width') != width or payload.get('height') != height or
            type(payload.get('frame')) is not int or payload['frame'] != camera.get('frame')):
        raise ValueError('临时'+label+'尺寸或帧号不一致')
    timestamp, source_time = payload.get('timestamp'), camera.get('timestamp')
    if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
            for v in (timestamp,source_time)) or abs(timestamp-source_time) > 1e-6):
        raise ValueError('临时'+label+'时间戳不一致')
    encoded = payload.get('data')
    expected = width*height*4
    if not isinstance(encoded,str) or len(encoded) > 4*((expected*2+128+2)//3):
        raise ValueError('临时'+label+'载荷长度无效')
    try:
        compressed = base64.b64decode(encoded, validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, expected+1)
    except (ValueError,binascii.Error,zlib.error) as exc:
        raise ValueError('临时'+label+'载荷损坏') from exc
    if (len(raw) != expected or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data):
        raise ValueError('临时'+label+'字节数或压缩边界无效')
    return np.frombuffer(raw,dtype=np.uint8).reshape(height,width,4)


def camera_rgb(camera):
    """原始BGRA只换通道为RGB，不拉伸/调色；显式旧路径仅供旧资产和fixture读取。"""
    if 'rgb_transport' in camera:
        return Image.fromarray(_camera_bgra(camera, 'rgb_transport', 'RGB')[:,:,[2,1,0]])
    if not camera.get('rgb_path'):
        raise ValueError('相机缺少同帧RGB载荷或显式旧资产路径')
    with Image.open(camera['rgb_path']) as source:
        result = source.convert('RGB')
    if result.size != (camera['width'], camera['height']):
        raise ValueError('相机图像大小与标定不一致')
    return result


def retained_camera_metadata(camera):
    """只保留RGB路径及标定；临时传输载荷不进入落盘资产。"""
    keys = ('view','rgb_path','width','height','fov_deg','frame','timestamp',
            'world_to_camera_ue','camera_to_world_ue',
            'rgb_saved','rgb_postprocess_effects')
    result = {key:copy.deepcopy(camera[key]) for key in keys if key in camera}
    return result


class CameraVideoRecorder:
    """六相机拼图录像；完成后实际解码核查，不把write调用数当成有效视频证据。"""
    def __init__(self, adapter, path):
        self.adapter, self.path = adapter, filesystem_path(path)
        self.requested_path = self.path
        self.writer = None
        self.frames = 0
        self.closed = False
        self.size = None
        self.codec = None
        self.status = 'waiting_for_rgb'
        self.error = None
        self.decoded_frames = 0
        self.frames_with_plan = 0
        self.duplicate_frames = 0
        self.last_frame = None
        self.first_frame = None
        self.encoder_attempts = []
        self.status_path = self.path.parent / 'video_recording.json'
        self.frame_manifest_path = self.path.parent / 'video_frames.jsonl'
        self.frame_manifest_path.write_text('', encoding='utf-8')
        if not math.isfinite(float(adapter.camera_hz)) or adapter.camera_hz <= 0:
            raise ValueError('CARLA录像帧率必须为有限正数')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._save_status()

    def _save_status(self):
        temporary = self.status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.metadata(), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(self.status_path)

    def _open_writer(self, cv2, size):
        # 部分服务器OpenCV没有MP4编码能力；明确记录AVI后备，而非留一个空MP4。
        for codec, path in (('mp4v', self.requested_path), ('MJPG', self.requested_path.with_suffix('.avi'))):
            if path.exists():
                raise RuntimeError('CARLA录像路径已存在，拒绝覆盖：'+str(path))
            try:
                candidate = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec),
                                            self.adapter.camera_hz, size)
            except cv2.error as exc:
                self.encoder_attempts.append({'codec': codec, 'path': str(path), 'opened': False,
                                              'error': str(exc)})
                logging.getLogger(__name__).warning('CARLA录像编码器创建异常：%s；尝试下一编码器', codec)
                continue
            opened = candidate.isOpened()
            self.encoder_attempts.append({'codec': codec, 'path': str(path), 'opened': bool(opened)})
            if opened:
                self.writer, self.path, self.codec, self.size = candidate, path, codec, size
                self.status = 'recording'
                logging.getLogger(__name__).info('CARLA录像开始：%s，编码=%s，%s，%.1f帧/秒',
                    self.path, codec, size, self.adapter.camera_hz)
                self._save_status()
                return
            candidate.release()
            logging.getLogger(__name__).warning('CARLA录像编码器不可用：%s；尝试下一编码器', codec)
        raise RuntimeError('CARLA录像编码器均启动失败；详见video_recording.json')

    def _capture_frame(self):
        if self.closed:
            raise RuntimeError('CARLA录像已关闭，不能继续写帧')
        try:
            packets = self.adapter._take_display_frames()
            if not packets:
                return False
            captured = False
            for packet in packets:
                image, display = self.adapter._render_packet(packet)
                captured = self._write_frame(image, packet['frame'], display) or captured
            return captured
        except Exception as exc:
            self.status, self.error = 'failed', str(exc)
            self._save_status()
            raise

    def _write_frame(self, image, frame, display):
        try:
            import cv2
            if image is None:
                # 非RGB终止帧允许没有图像，不能重复旧图伪装新采集。
                if self.adapter._raw.get('runtime',{}).get('terminal_reason'):
                    return False
                raise RuntimeError('CARLA未提供录像RGB画面')
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError('CARLA录像帧必须为RGB uint8')
            size = (image.shape[1], image.shape[0])
            if any(length <= 0 or length % 2 for length in size):
                raise ValueError('CARLA录像尺寸必须为正偶数')
            if type(frame) is not int:
                raise ValueError('CARLA录像帧缺少有效仿真帧号')
            if self.last_frame is not None:
                if frame < self.last_frame:
                    raise ValueError('CARLA录像帧号倒退')
                if frame == self.last_frame:
                    self.duplicate_frames += 1
                    return False
            if self.writer is None:
                self._open_writer(cv2, size)
            if size != self.size:
                raise ValueError('同一轮CARLA录像不能改变画面尺寸')
            self.writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            with self.frame_manifest_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'video_index': self.frames, 'carla_frame': frame,
                    'physical_tick': display.get('physical_tick'), 'sim_time': display.get('sim_time'),
                    'active_plan_source_frame': display.get('active_plan_source_frame'),
                    'source_cameras': display.get('source_cameras', []),
                    'plan_overlay': bool(display.get('plan_overlay', False))}, allow_nan=False)+'\n')
            self.frames += 1
            self.frames_with_plan += int(bool(display.get('plan_overlay', False)))
            self.first_frame = frame if self.first_frame is None else self.first_frame
            self.last_frame = frame
            if self.frames == 1 or self.frames % max(1, round(self.adapter.camera_hz)) == 0:
                self._save_status()
            return True
        except Exception as exc:
            self.status, self.error = 'failed', str(exc)
            self._save_status()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.writer is not None:
                self.writer.release()
            if self.status == 'failed':
                return
            if not self.frames:
                self.status = 'no_frames'
                self.error = '仿真未产生可录像RGB帧；请检查初始化和规划终止原因'
                logging.getLogger(__name__).warning('CARLA录像无画面：%s', self.error)
                return
            import cv2
            capture = cv2.VideoCapture(str(self.path))
            try:
                if not capture.isOpened():
                    raise RuntimeError('CARLA录像文件无法打开，未通过可播放检查')
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or abs(fps-self.adapter.camera_hz) > .01:
                    raise RuntimeError('CARLA录像帧率与RGB采集频率不一致')
                while True:
                    okay, image = capture.read()
                    if not okay:
                        break
                    if (image.shape[1],image.shape[0]) != self.size:
                        raise RuntimeError('CARLA录像解码尺寸与写入画面不一致')
                    self.decoded_frames += 1
                if self.decoded_frames != self.frames:
                    raise RuntimeError('CARLA录像解码帧数不一致：写入{}，解码{}'.format(self.frames,self.decoded_frames))
            finally:
                capture.release()
            self.status = 'complete'
            logging.getLogger(__name__).info('CARLA录像已保存并通过解码检查：%s，%d帧', self.path,self.decoded_frames)
        except Exception as exc:
            self.status, self.error = 'failed', str(exc)
            raise
        finally:
            self._save_status()

    def metadata(self):
        return {'path': str(self.path), 'frames': self.frames, 'fps': self.adapter.camera_hz,
                'requested_path': str(self.requested_path), 'status': self.status, 'error': self.error,
                'size': self.size, 'codec': self.codec, 'decoded_frames': self.decoded_frames,
                'playback_verified': self.status == 'complete', 'duplicate_frames_skipped': self.duplicate_frames,
                'first_carla_frame': self.first_frame, 'last_carla_frame': self.last_frame,
                'encoder_attempts': self.encoder_attempts, 'status_path': str(self.status_path),
                'frame_manifest_path': str(self.frame_manifest_path),
                'source': 'CARLA_same_frame_six_rgb_10hz',
                'encoded_duration_s': self.frames / self.adapter.camera_hz,
                'plan_overlay': self.frames_with_plan > 0, 'frames_with_plan': self.frames_with_plan,
                'display_frames_dropped': self.adapter._display_frames_dropped,
                'note': '实际10Hz六相机RGB；规划1/2/10Hz均消费独立图像队列；不补帧。丢帧单列，编码时长不代替仿真时长；原始图片1Hz归档'}


class CarlaAdapter:
    def __init__(self, config, seed, episode_id, live=False, bridge_factory=BridgeProcess):
        from .carla_bridge import validate_camera_source_clock
        validate_camera_source_clock(config.get('clock', {}))
        self.config = copy.deepcopy(config)
        self.cfg = copy.deepcopy(config.get('carla', {}))
        self.seed, self.episode_id, self.live = seed, episode_id, live
        # Goal/reference IDs enter the VLM prompt. Keep their source binding,
        # while withholding the semantic scenario name from the observation.
        self._observation_namespace = 'carla-'+hashlib.sha256(episode_id.encode('utf-8')).hexdigest()[:24]
        self.planner_hz = float(config.get('clock', {}).get('planner_hz', 10))
        if self.planner_hz not in (1., 2., 10.):
            raise ValueError('CARLA规划频率只支持1、2、10Hz')
        self.camera_hz = float(self.cfg.get('camera_hz',10))
        self.runtime_mode = self.cfg.get('runtime_mode','sync')
        if self.runtime_mode not in ('sync','async'):
            raise ValueError('CARLA运行模式必须为sync或async')
        self.dt = 1. / self.planner_hz
        self.cfg.update(physics_hz=int(config.get('clock', {}).get('physics_hz', 50)), planner_hz=int(self.planner_hz))
        for key in ('acceleration_min_mps2', 'acceleration_max_mps2', 'steering_abs_max_rad', 'steering_rate_abs_max_radps'):
            if key in config.get('mpc', {}):
                self.cfg[key] = config['mpc'][key]
        self.cfg['perception_radius_m'] = config.get('perception', {}).get('radius_m', 120.)
        self.cfg['duration_s'] = float(config.get('duration',config.get('clock',{}).get('episode_seconds',30)))
        from .carla_sensor_policy import sensor_policy
        self._sensor_policy=sensor_policy(
            fast_enabled=config.get('models',{}).get('fast_backend','off')!='off',
            reflection_enabled=config.get('models',{}).get('reflection_backend','off')!='off',
            live=bool(live))
        self.cfg.update(self._sensor_policy)
        self.bridge_factory = bridge_factory
        self.bridge = None
        self.output_dir = None
        self._snapshot = None
        self._raw = None
        self._history = {}
        self._public_priority = PublicPriorityTracker()
        self._next_history = 0.
        self._route_s = None
        self._route_projection = None
        self._projection_position = None
        self._projection_frame = None
        self._projection_tick = None
        self._navigation_status = 'following'
        self._previous_control = np.zeros(2)
        self._planned = None
        self._recorder = None
        self._mosaic = None
        self._display_frames = []
        self._display_frames_dropped = 0
        self._representation_capture_enabled = True
        self._display_info = {'plan_overlay':False,'plan_segments':0}
        self._step = 0
        self._last_cycle_frame = None
        self._observation_wait_s = 0.
        self._confirmed_lane = None
        self._pending_lane = None
        self._pending_since = None
        self._reference_revision = 0
        self._progress_offset = 0.
        self.closed = False

    def set_output_directory(self, path):
        if self.bridge is not None:
            raise RuntimeError('CARLA 连接后不能更换资产目录')
        self.output_dir = filesystem_path(path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_representation_capture_enabled(self, enabled):
        """纯基础规划不构建OCC证据；对象输入始终采用范围真值。"""
        if type(enabled) is not bool:
            raise ValueError('表示采集开关必须为布尔值')
        if self.bridge is not None:
            raise RuntimeError('必须在CARLA初始化前确定表示采集模式')
        self._representation_capture_enabled = enabled

    def set_sensor_policy(self, policy):
        """相机由实际消费者启用；必须早于SDK连接，禁止运行时偷偷扩充。"""
        if self.bridge is not None:
            raise RuntimeError('必须在CARLA初始化前确定相机消费者')
        if not isinstance(policy,dict) or policy.get('depth_enabled') is not False:
            raise ValueError('传感器策略必须明确禁用深度')
        self._sensor_policy=copy.deepcopy(policy)
        self.cfg.update(self._sensor_policy)
        self.camera_hz=float(policy['rgb_hz'])

    def begin_cycle(self):
        """异步循环只在入口取得一次冻结观测；同轮其他读取不拼接新状态。"""
        if self.bridge is None:
            self._initialize()
        self._observation_wait_s=0.
        if self.runtime_mode=='async' and self._last_cycle_frame is not None:
            started=time.perf_counter()
            raw=self.bridge.call('observe',after_frame=self._last_cycle_frame)
            self._ingest_display_frames(raw)
            self._observation_wait_s=time.perf_counter()-started
            self._raw,self._snapshot=raw,None
        result=self.snapshot()
        result.diagnostics.setdefault('adapter_timing',{})['observation_wait_s']=self._observation_wait_s
        self._last_cycle_frame=self._raw['frame']
        return result

    def pause_runtime(self):
        """轮末先暂停世界，再等待已接纳模型/反思任务收尾。"""
        try:
            if self.bridge is not None and not self.closed and not getattr(self.bridge, 'closed', False):
                if getattr(self.bridge, 'failed', False):
                    # SDK桥报错后会退出请求循环并在finally释放世界，不能再向它发pause。
                    self.bridge.close()
                else:
                    raw = self.bridge.call('pause')
                    self._ingest_display_frames(raw)
        finally:
            # 即使最后一次步进失败，初始化时已经收到的真实RGB也必须归档为录像。
            if self._recorder is not None and self._display_frames:
                self._recorder._capture_frame()

    def runtime_failure_report(self):
        """失败帧单独报告，不将未完成控制周期冒充成功执行步。"""
        report = {'bridge_error': copy.deepcopy(getattr(self.bridge, 'last_error', None))}
        if self.output_dir is None:
            return report
        for key, filename in (('camera_sync', 'camera_sync_failure.json'),
                              ('initialization', 'carla_initialization_failure.json'),
                              ('world_ownership', 'carla_world_ownership.json'),
                              ('cleanup', 'carla_cleanup.json')):
            path = self.output_dir / filename
            if not path.is_file():
                continue
            try:
                report[key] = json.loads(path.read_text(encoding='utf-8'))
                report.setdefault('diagnostic_paths', {})[key] = str(path)
                if key == 'camera_sync':
                    report['diagnostic_path'] = str(path)
            except (OSError, ValueError) as exc:
                report.setdefault('diagnostic_read_errors', {})[key] = type(exc).__name__
        return report

    def evaluation_report(self):
        """Read the physical evaluator after SDK cleanup and late-event merging."""
        if self.output_dir is not None:
            path = self.output_dir/'evaluation.json'
            if path.is_file():
                return json.loads(path.read_text(encoding='utf-8'))
        return copy.deepcopy((self._raw or {}).get('evaluation'))

    def _initialize(self):
        if self.output_dir is None:
            raise RuntimeError('CARLA 采集必须先绑定本轮输出目录')
        self.bridge = self.bridge_factory(self.cfg.get('python', 'python'), self.output_dir,
            timeout_s=float(self.cfg.get('timeout_s', 30)) * 2 + 10)
        try:
            raw = self.bridge.call('init', config=self.cfg, directory=str(self.output_dir), seed=int(self.seed))
            self._ingest_display_frames(raw)
            mapping = scope_junction_boundaries(raw.pop('map'))
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
            self._goal_position = self._navigation_route[-1].copy()
            self._goal_lane_id = self._navigation_lane_ids[-1]
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

    def _ingest_display_frames(self, raw):
        """图像批次从RPC抽离，不进入场景真值/记忆资产或无限堆积。"""
        packets = raw.pop('display_frames', [])
        if not isinstance(packets, list):
            raise ValueError('CARLA显示帧批次必须为列表')
        if self._recorder is None:
            # 独立observer持续取流；主线程预览显示本轮最新冻结图像。
            return
        self._display_frames.extend(packets)
        capacity = int(self.cfg.get('display_queue_frames', 50))
        if len(self._display_frames) > capacity:
            count = len(self._display_frames)-capacity
            self._display_frames_dropped += count
            del self._display_frames[:count]
        bridge_count = raw.get('runtime', {}).get('live', {}).get('display_frames_dropped', 0)
        self._display_frames_dropped = max(self._display_frames_dropped, int(bridge_count))

    def _take_display_frames(self):
        self.snapshot()
        packets, self._display_frames = self._display_frames, []
        return packets

    def _render_packet(self, packet):
        from .carla_display import plan_mosaic_overlay
        mosaic = np.asarray(self._build_mosaic(packet['cameras']))
        visualization = packet.get('plan_visualization', {})
        image, segments = (plan_mosaic_overlay(mosaic, packet['cameras'], visualization)
            if self.cfg.get('display_plan_overlay', True) else (mosaic, 0))
        display = {'plan_overlay': segments > 0, 'plan_segments': segments,
                   'capture_carla_frame': packet['frame'], 'sim_time': packet['time'],
                   'physical_tick': packet.get('physical_tick'),
                   'active_plan_source_frame': packet.get('plan_source_frame'),
                   'source_cameras': [{key: camera[key] for key in ('view', 'frame', 'timestamp')}
                                      for camera in packet['cameras']]}
        return image, display

    def _convert_actor(self, value):
        fields = {key: value[key] for key in ('id', 'kind', 'heading', 'length', 'width', 'lane_id', 'visible', 'radius')}
        for key in ('position', 'velocity', 'acceleration'):
            fields[key] = np.asarray(value[key], float)
        for key in ('wheelbase_m', 'rear_axle_to_center_m'):
            if key in value:
                fields[key] = value[key]
        return Actor(**fields)

    def _confirm_lane(self, ego, timestamp):
        """半车体进入邻道持续0.3秒后确认实际车道；冻结任务路线不随之平移。"""
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
        # 导航路线/终点不可因实际换道平移。局部参考负责在合法邻道行驶及归道，
        # 规则、进度和到达始终使用同一条冻结任务路线。
        self._confirmed_lane = candidate
        self._pending_lane = self._pending_since = None
        self._reference_revision += 1

    def _update_navigation(self, ego, frame):
        if self._projection_frame == frame:
            return self._route_projection
        travel = (0. if self._projection_position is None else
                  float(np.linalg.norm(ego.position-self._projection_position)))
        tick = self._raw.get('runtime',{}).get('physical_ticks',0)
        elapsed = 0. if self._projection_tick is None else max(0.,tick-self._projection_tick)/self.cfg['physics_hz']
        physical_reach = max(2.,float(np.linalg.norm(ego.velocity))*elapsed+3.*elapsed**2+1.)
        measured_jump = self._projection_position is not None and travel > physical_reach
        travel = min(travel,physical_reach)
        window = 20. if self._route_s is None else max(4., 2.*travel+2.)
        projection = checked_route_projection(ego.position, self.route,
            0. if self._route_s is None else self._route_s, ego.heading, window=window)
        if (not projection['valid'] and projection['reason']=='distance_exceeded' and
                abs(projection['distance_m']-abs(projection['lateral_m'])) < .5):
            maximum=self._cross_section_limit(ego.lane_id,projection['position'])
            current_lane=next((lane for lane in self.lanes if lane.id==ego.lane_id),None)
            if (maximum>6. and current_lane is not None and
                    Polygon(current_lane.polygon).buffer(.01).covers(Point(ego.position))):
                projection=checked_route_projection(ego.position,self.route,
                    0. if self._route_s is None else self._route_s,ego.heading,
                    window=window,maximum_distance=maximum)
        if measured_jump:
            projection.update(valid=False,reason='observed_displacement_exceeds_reachable_distance')
        if projection['valid']:
            self._route_s = projection['s_m']
        elif self._route_s is None:
            self._route_s = 0.
        self._projection_position = ego.position.copy()
        self._projection_frame = frame
        self._projection_tick = tick
        self._route_projection = projection
        total = float(polyline_lengths(self._navigation_route)[-1])
        end_delta=self._navigation_route[-1]-self._navigation_route[-2]
        heading = float(np.arctan2(end_delta[1],end_delta[0]))
        beyond = float((ego.position-self._goal_position) @ np.array([np.cos(heading),np.sin(heading)]))
        missed = (self._route_s >= total-15. and
                  (beyond > 1. or (beyond >= -3. and ego.lane_id != self._goal_lane_id)))
        # 一旦错过任务终点只受控制动，禁止把下一圈解释为同一任务的延伸。
        if self._navigation_status != 'destination_missed':
            self._navigation_status = ('destination_missed' if missed else
                'following' if projection['valid'] else 'route_projection_invalid')
        return projection

    def _cross_section_limit(self, current_id, navigation_position):
        """只按同路段公开相邻车道的真实横截面扩展投影距离，不接纳平行道路。"""
        lanes={lane.id:lane for lane in self.lanes}
        current=lanes.get(current_id)
        if current is None:return 6.
        section=tuple(str(current_id).split('|')[:2])
        roots=[lane for lane in self.lanes if lane.id in self._navigation_lane_ids
            and tuple(str(lane.id).split('|')[:2])==section
            and Polygon(lane.polygon).buffer(.01).covers(Point(navigation_position))]
        seen=set();pending=roots[:]
        while pending:
            lane=pending.pop()
            if lane.id in seen:continue
            seen.add(lane.id)
            if lane.id==current_id:
                s,_,heading=project_polyline(navigation_position,current.center)
                base=sample_polyline(current.center,s)
                delta=base-np.asarray(navigation_position)
                # 只允许横向宽度，不能以纵向端点残差扩展搜索走廊。
                tangent=np.array([np.cos(heading),np.sin(heading)])
                if abs(float(delta@tangent))>.5:return 6.
                return max(6.,float(np.linalg.norm(delta))+current.width/2.)
            for key in (lane.left,lane.right):
                other=lanes.get(key)
                if other is None or tuple(str(key).split('|')[:2])!=section:continue
                a=project_polyline(navigation_position,lane.center)[2]
                b=project_polyline(navigation_position,other.center)[2]
                if abs(float(wrap_angle(a-b)))<.3:pending.append(other)
        return 6.

    def _project_public_stop(self, position, lane_id=None):
        """Project public stop geometry onto the upcoming navigation, without a 40m cap.

        A lateral residual alone is insufficient at a clipped polyline endpoint:
        a point 100m ahead can otherwise look like a stop at the 40m search edge.
        """
        position = np.asarray(position, float)
        arc = polyline_lengths(self.route)
        start = max(0., float(self._route_s)-3.)
        remaining = self.route[arc > start+1e-8]
        if not len(remaining):
            return None
        path = clean_polyline(np.vstack((sample_polyline(self.route, start), remaining)))
        s, _, _ = project_polyline(position, path)
        # 邻道停止线可有一条车道宽的横向偏差，仍沿冻结导航弧长约束。
        # 只为实际当前车道/导航声明车道放宽几何关联；其他道路不参与。
        maximum = (self._cross_section_limit(lane_id,sample_polyline(path,s))
                   if lane_id == self._raw['ego']['lane_id'] else 3.)
        tangent = project_polyline(position, path)[2]
        lane = next((item for item in self.lanes if item.id == lane_id), None)
        if lane is not None and abs(float(wrap_angle(project_polyline(position,lane.center)[2]-tangent))) > .7:
            return None
        if np.linalg.norm(position-sample_polyline(path, s)) >= maximum:
            return None
        s += start
        return float(s) if s >= self._route_s-.1 else None

    def snapshot(self):
        if self.closed:
            raise RuntimeError('CARLA 适配器已关闭')
        if self.bridge is None:
            self._initialize()
        if self._snapshot is not None:
            return self._snapshot
        build_started=time.perf_counter()
        raw = self._raw
        ego = self._convert_actor(raw['ego'])
        actors = [self._convert_actor(a) for a in raw['actors']]
        timestamp = float(raw['time'])
        self._confirm_lane(ego, timestamp)
        history_cfg = self.config.get('clock', {})
        history_window = float(history_cfg.get('history_seconds', 3.))
        if 'observed_history' in raw:
            # SDK物理tick采集的2Hz因果历史，不用1Hz规划快照补造中间样本。
            self._history=copy.deepcopy(raw['observed_history'])
        elif timestamp + 1e-8 >= self._next_history:
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
        self._update_navigation(ego, raw['frame'])
        applied_evidence=raw.get('last_applied_control')
        if isinstance(applied_evidence,dict):
            applied_evidence=applied_evidence.get('control')
        if applied_evidence is not None:
            applied=np.asarray(applied_evidence,float)
            if applied.shape!=(2,) or not np.isfinite(applied).all():
                raise ValueError('实际执行控制必须是有限[a,delta]')
            self._previous_control=applied.copy()
        policy='range_limited_ground_truth_no_occlusion-1.0'
        if self._representation_capture_enabled:
            radius=float(self.cfg['perception_radius_m'])
            observed=Point(ego.position).buffer(radius,quad_segs=128)
            # 范围内真值对象提供完整框，范围之外保持未知；没有遮挡计算。
            observed = unary_union([observed, actor_shape(ego)] + [actor_shape(a) for a in actors])
            observation_audit={'method':policy,'version':policy,'radius_m':radius,
                'valid':True,'occlusion_tested':False,'depth_sensor_count':0,
                'limitation':'规定范围内完整真值输入，不能等同于真实相机可见或感知系统输出'}
            observed_wkt = observed.wkt
        else:
            # 纯基础规划不消费OCC空区证据。None表示未计算，不能伪造为空或满覆盖。
            observed_wkt = None
            observation_audit = {'method':'not_computed_for_pure_base', 'valid':False,
                'reason':'纯基础规划不消费空区掩码；对象按范围真值提供，不判断遮挡',
                'input_policy':policy,'occlusion_tested':False,'depth_sensor_count':0}
        frame_id = '{}:{:06d}'.format(self._observation_namespace, raw['frame'])
        cameras=raw.get('cameras',[])
        mosaic = self._build_mosaic(cameras) if cameras else None
        self._mosaic = np.asarray(mosaic) if mosaic is not None else None
        runtime = raw.get('runtime', {})
        if cameras and (type(runtime.get('physical_ticks')) is not int or
                        runtime.get('physics_hz') != self.cfg['physics_hz']):
            raise ValueError('RGB归档缺少一致的冻结观测物理时钟')
        saved = bool(cameras) and camera_storage_due(timestamp, self.cfg.get('camera_save_hz', 1),
            runtime.get('physical_ticks'), self.cfg['physics_hz'])
        for camera in raw['cameras']:
            if 'rgb_saved' in camera and camera['rgb_saved'] is not saved:
                raise ValueError('SDK桥RGB落盘时刻与快模型原帧时钟不一致')
        camera_images, originals = [], []
        manifest_path, state_path = None, None
        if saved:
            # 只保存实际1Hz源帧，不重复旧图，也不创建深度资产。
            originals = [camera.get('rgb_path') for camera in raw['cameras']]
            if not all(isinstance(path,str) and Path(path).is_file() for path in originals):
                raise ValueError('源帧缺少完整的六路原始RGB资产')
            folder = self.output_dir / 'camera_frames' / '{:08d}'.format(raw['frame'])
            folder.mkdir(parents=True, exist_ok=True)
            mosaic_path = folder / 'multi_camera_mosaic.png'
            mosaic.save(mosaic_path)
            retained_cameras = [retained_camera_metadata(camera) for camera in raw['cameras']]
            manifest = {'frame_id': frame_id, 'time': timestamp, 'carla_frame': raw['frame'],
                'carla_timestamp': raw['timestamp'], 'cameras': retained_cameras,
                'calibration_version': 'carla-pinhole-rgb-2.0',
                'storage_version': 'carla-rgb-source-1hz-2.0', 'camera_save_hz': 1,
                'sensor_policy':self._sensor_policy,'observation_policy':policy}
            manifest_path = folder / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            retained_raw = {**{key: value for key, value in raw.items() if key != 'evaluation'},
                'cameras': retained_cameras, 'storage_version': 'carla-rgb-source-1hz-2.0',
                'camera_save_hz': 1, 'observed_support_wkt': observed_wkt, 'observation_evidence': observation_audit}
            state_path = folder / 'current_observed_state.json'
            state_path.write_text(json.dumps(retained_raw, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            camera_images = [{'view': 'multi_camera_mosaic', 'path': str(mosaic_path.resolve()),
                              'frame_id': frame_id, 'time': timestamp}]
        stop_s, light_records = None, []
        unmapped_current_red = False
        for light in raw.get('traffic_lights', []):
            relevant = []
            current_red_unmapped = False
            considered_stops = []
            for stop in light['stops']:
                if stop['lane_id'] not in self.route_lane_ids and stop['lane_id'] != ego.lane_id:
                    continue
                considered_stops.append(stop)
                s = self._project_public_stop(stop['position'], stop['lane_id'])
                if s is not None:
                    relevant.append(s)
                elif stop['lane_id'] == ego.lane_id and light['state'] in ('red','yellow'):
                    lane = next((item for item in self.lanes if item.id == ego.lane_id), None)
                    if lane is not None:
                        ego_lane_s = project_polyline(ego.position,lane.center)[0]
                        stop_lane_s = project_polyline(stop['position'],lane.center)[0]
                        current_red_unmapped |= bool(0. <= stop_lane_s-ego_lane_s <= 40.)
                        unmapped_current_red |= current_red_unmapped
            if not relevant:
                if current_red_unmapped:
                    light_records.append({'id':light['id'],'state':light['state'],'stop_s':None,
                        'association':'current_lane_stop_unmapped','stops':copy.deepcopy(considered_stops)})
                continue
            s = min(relevant)
            light_records.append({'id': light['id'], 'state': light['state'], 'stop_s': s,
                                  'association':'controlled_lane_and_geometry','stops':copy.deepcopy(considered_stops)})
            if light['state'] in ('red', 'yellow'):
                stop_s = s if stop_s is None else min(stop_s, s)
        desired = float(self.cfg.get('desired_speed_mps', 10.))
        remaining = float(polyline_lengths(self.route)[-1])-self._route_s
        desired = min(desired, float(np.sqrt(2.*3.*max(0.,remaining-1.5))))
        public = raw.get('public_road_rules') or {'rules': []}
        if public.get('rules') or public.get('stop_obligations'):
            if (public.get('frame') != raw['frame'] or
                    public.get('physical_tick') != raw.get('runtime', {}).get('physical_ticks')):
                raise ValueError('公共道路规则与冻结场景帧不一致')
        known_lanes = {lane.id for lane in self.lanes}
        for lane in self.lanes:
            lane.priority = 0
        priority_complete = True
        for rule in public.get('rules', []):
            controlled = set(rule['controlled_lane_ids'])
            priority = set(rule['priority_lane_ids'])
            if not (controlled | priority).issubset(known_lanes):
                priority_complete = False
            # Only public lane priority is supplied, never actor-script intent.
            for lane in self.lanes:
                if lane.id in priority:
                    lane.priority = max(lane.priority, 1)
        stop_reasons = []
        for obligation in public.get('stop_obligations', []):
            if not obligation.get('must_stop') or obligation.get('lane_id') not in self.route_lane_ids:
                continue
            s = self._project_public_stop(obligation['position'])
            if s is not None:
                stop_s = s if stop_s is None else min(stop_s, s)
                stop_reasons.append(obligation['id'])
        evidence = raw.get('current_road_evidence')
        identities = {lane.id for lane in self.lanes}
        map_complete = bool(evidence is not None and evidence.get('lane_id') == ego.lane_id and
            ego.lane_id in identities and all(evidence.get(side) is None or evidence[side] in identities for side in ('left', 'right')))
        if map_complete:
            current_lane = next(lane for lane in self.lanes if lane.id == ego.lane_id)
            current_lane.left, current_lane.right = evidence['left'], evidence['right']
        classifications = {a['id']: copy.deepcopy(a['classification']) for a in raw['actors']
                           if isinstance(a.get('classification'), dict)}
        # 路权的动态触发也必须使用本帧同一车道弧长范围；否则即使MPC和提示词
        # 已筛选，范围外来车仍能经 must_wait/stop_s 间接控制自车。
        scope_input = SceneSnapshot(frame_id, timestamp, ego, actors, self.lanes, self.route,
            self.route_lane_ids, self.drivable, 'carla', diagnostics={
                'lane_map_complete': map_complete, 'stable_navigation_lane_id': self._confirmed_lane,
                'platform': 'carla', 'actor_classification': classifications})
        priority_scene, object_scope = scope_scene(scope_input, self.config)
        priority_ids = {a.id for a in priority_scene.actors}
        priority_state = self._public_priority.update(public, raw['ego'],
            [a for a in raw['actors'] if a['id'] in priority_ids],
            self._navigation_route, timestamp,
            lane_paths={lane.id: {'center': lane.center, 'successors': lane.successors}
                        for lane in self.lanes})
        priority_state['evidence']['input_policy'] = object_scope['version']
        priority_state['evidence']['object_scope'] = {
            'version': object_scope['version'], 'settings': copy.deepcopy(object_scope['settings']),
            'selected_ids': sorted(priority_ids), 'selected_count': len(priority_ids)}
        priority_stop = priority_state.get('stop_position')
        if priority_stop is not None:
            s = self._project_public_stop(priority_stop)
            if s is not None:
                stop_s = s if stop_s is None else min(stop_s, s)
                stop_reasons.append('公共优先通行对象尚未清空冲突区')
        rules = {'desired_speed': desired, 'speed_limit': float(raw['speed_limit_mps']),
            'must_wait': stop_s is not None, 'stop_s': stop_s,
            'wait_reason': ('公共道路停止或让行义务：'+','.join(map(str, stop_reasons)) if stop_reasons
                            else '当前导航车道交通灯要求停车') if stop_s is not None else None,
            'traffic_lights': light_records, 'crosswalks': self.crosswalks or [],
            'map_available': {'crosswalk': self.crosswalks is not None, 'traffic_light': True, 'stop_line': True},
            'road_priority_available': bool(public.get('rules')) and priority_complete and priority_state['available'],
            'goal_s': self._goal_s, 'goal_position':self._goal_position.tolist(),
            'goal_lane_id':self._goal_lane_id, 'goal_remaining_m':remaining,
            'destination_stop_s':float(polyline_lengths(self.route)[-1])+ego.length/2.-1.5,
            'navigation_status':self._navigation_status,
            'navigation_brake_required':self._navigation_status != 'following' or unmapped_current_red,
            'unmapped_current_red':unmapped_current_red,
            'public_road_rules': copy.deepcopy(public),
            'priority_map_complete': priority_complete, 'public_priority_screen': priority_state}
        cycle_result = self._result(raw)
        cycle_result.pop('evaluation', None)
        diagnostics = {'route_s': self._route_s, 'reference_anchor_s': self._route_s,
            'route_projection':copy.deepcopy(self._route_projection),
            'navigation_status':self._navigation_status,
            'navigation_progress_m': self._route_s+self._progress_offset, 'stable_navigation_lane_id': self._confirmed_lane,
            'previous_control': self._previous_control.tolist(), 'reference_revision': self._reference_revision,
            'lane_map_complete': map_complete, 'visibility_wkt': observed_wkt,
            'object_scope': object_scope, 'actor_classification': classifications,
            'visibility_evidence': observation_audit, 'physics_hz': self.cfg['physics_hz'],
            'historical_only_objects': {k: v for k, v in self._history.items() if k not in {a.id for a in [ego] + actors}},
            'platform': 'carla', 'carla_frame': raw['frame'], 'carla_timestamp': raw['timestamp'],
            'actual_actuation': raw.get('actual_actuation'), 'control_response': raw.get('control_response'),
            'physical_events': raw.get('events', []), 'acceleration_source': raw['ego'].get('state_sources'),
            'observation_policy':policy,'sensor_policy':copy.deepcopy(self._sensor_policy),
            'runtime':copy.deepcopy(raw.get('runtime',{})),
            'bridge_timing':copy.deepcopy(raw.get('bridge_timing',{})),
            'cycle_result':cycle_result,
            'environment_obstacles': raw.get('environment_obstacles', {}),
            'weather': raw.get('weather', {}), 'weather_source': raw.get('weather_source', 'unknown'),
            'camera_storage': {'version': 'carla-rgb-source-1hz-2.0', 'save_hz': 1, 'saved': saved,
                               'runtime_frame': raw['frame'], 'runtime_time': timestamp},
            'platform_limitations': ['同层平面道路；路权仅来自已声明的公共道路规则',
                '范围内真值输入，不判断遮挡，不等同于真实传感器感知；尚待服务器实测']}
        self._snapshot = SceneSnapshot(frame_id, timestamp, ego, actors, self.lanes, self.route,
            self.route_lane_ids, self.drivable, 'carla', rules=rules,
            assets={'camera_images': camera_images,
                    'camera_manifest_json': str(manifest_path.resolve()) if manifest_path else None,
                    'original_camera_images': originals,
                    'depth_images': [],
                    'current_state_json': str(state_path.resolve()) if state_path else None,
                    'physical_lane_centers': {l.id: l.center for l in self.lanes}, 'bev_drivable': self.drivable},
            diagnostics=diagnostics)
        self._snapshot.diagnostics['adapter_timing']={
            'snapshot_build_s':time.perf_counter()-build_started,
            'observation_wait_s':self._observation_wait_s}
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
            current = camera_rgb(indexed[name])
            if current.size != (width, height):
                raise ValueError('相机图像大小与标定不一致')
            x, y = index % 3 * width, index // 3 * height
            mosaic.paste(current, (x, y))
            draw.rectangle((x, y, x + 140, y + 20), fill='black')
            draw.text((x + 4, y + 3), name, fill='white')
        return mosaic

    def _result(self,raw):
        """控制提交和独立时钟观测采用相同的终态与导航审计。"""
        ego = self._convert_actor(raw['ego'])
        projection = self._update_navigation(ego,raw['frame'])
        progress, lateral = self._route_s, projection['lateral_m']
        offroad = not self.drivable.buffer(float(self.config.get('mpc', {}).get('body_constraint_tolerance_m', .001))).covers(actor_shape(ego))
        collision = bool(raw['collision'])
        arrival = (projection['valid'] and progress >= self._goal_s and abs(lateral) < 2.5
            and np.linalg.norm(ego.position-self._goal_position) <= float(self.cfg.get('arrival_margin_m',3.))
            and ego.lane_id == self._goal_lane_id and not collision and not offroad)
        measured = raw.get('evaluation')
        if measured:
            collision = bool(measured.get('collision')) or collision
            arrival = arrival and bool(measured.get('arrival'))
        runtime=raw.get('runtime',{})
        ticks=runtime.get('physical_ticks')
        if type(ticks) is not int or ticks < 0:
            raise RuntimeError('CARLA冻结观测缺少有效的本轮物理帧计数')
        # 只用该观测冻结的计数；异步runtime.live可能属于更晚的世界状态。
        timeout = ticks >= duration_tick_limit(self.cfg['duration_s'], self.cfg['physics_hz'])
        runtime_reason=runtime.get('terminal_reason')
        runtime_failure=runtime_reason if runtime_reason not in (None,'duration','arrival','collision') else None
        if runtime_reason == 'arrival' and not arrival:
            runtime_failure = 'arrival_audit_mismatch'
        navigation_stop = (self._navigation_status if self._navigation_status != 'following'
            and np.linalg.norm(ego.velocity) < .2 else None)
        runtime_failure = runtime_failure or navigation_stop
        reason=runtime_failure or ('collision' if collision else 'arrival' if arrival else 'timeout' if timeout else 'running')
        return {'terminated': bool(collision or arrival or timeout or runtime_reason or navigation_stop),
            'collision': collision, 'arrival': arrival,'offroad': offroad, 'timeout': timeout,
            'time': raw['time'], 'progress': progress+self._progress_offset,'reason':reason,
            'route_projection':copy.deepcopy(projection),'navigation_status':self._navigation_status,
            'carla_frame': raw['frame'], 'control_response': raw.get('control_response'), 'events': raw.get('events', []),
            'runtime':copy.deepcopy(runtime),'bridge_timing':copy.deepcopy(raw.get('bridge_timing',{})),
            'evaluation': copy.deepcopy(measured),
            'last_applied_control':copy.deepcopy(raw.get('last_applied_control')),
            'plan_visualization':raw.get('plan_visualization',{})}

    def step_plan(self, plan):
        """传递已验算完整序列，由同步前缀或异步源时标执行器消费。"""
        current=self.snapshot()
        controls=np.asarray(plan.controls,float)
        if not plan.valid or plan.frame_id!=current.frame_id or abs(plan.time-current.time)>1e-7:
            raise ValueError('只能提交当前冻结帧通过验收的规划')
        if controls.ndim!=2 or controls.shape[1]!=2 or not len(controls) or not np.isfinite(controls).all():
            raise ValueError('CARLA控制序列必须为有限(N,2)')
        if self.runtime_mode=='sync':
            physics=self.cfg['physics_hz']
            current_ticks=self._raw.get('runtime',{}).get('physical_ticks')
            if type(current_ticks) is not int or current_ticks < 0:
                raise RuntimeError('CARLA冻结观测缺少有效的本轮物理帧计数')
            remaining=duration_tick_limit(self.cfg['duration_s'],physics)-current_ticks
            if remaining<=0:
                raise RuntimeError('本轮已达到时长上限，不能继续提交控制')
            expected_ticks=min(physics//self.cfg['planner_hz'],remaining)
        raw=self.bridge.call('step',controls=controls.tolist(),
            control_step_s=float(self.config['mpc']['step_s']),source_time=float(plan.time),plan=self._planned,
            source_frame=int(self._raw['frame']),
            source_speed_mps=float(np.linalg.norm(current.ego.velocity)))
        self._ingest_display_frames(raw)
        if self.runtime_mode=='sync':
            # 校验本次真实帧差和时间增量；不把累计时间强行截到理想结束秒数。
            # 部分周期只消费剩余物理帧，外部额外tick仍必须报错。
            actual_ticks=raw['frame']-self._raw['frame']
            terminal=raw.get('runtime',{}).get('terminal_reason') is not None
            if ((actual_ticks!=expected_ticks and not (terminal and 0<=actual_ticks<=expected_ticks)) or
                    raw.get('runtime',{}).get('physical_ticks')!=current_ticks+actual_ticks or
                    abs((raw['time']-current.time)-actual_ticks/physics)>1e-5):
                raise RuntimeError('CARLA同步步进时间或帧号不符合执行前缀')
        elif raw['frame']<self._raw['frame'] or raw['time']+1e-8<current.time:
            raise RuntimeError('CARLA异步观测时间倒退')
        self._raw,self._snapshot=raw,None
        self._step+=1
        return self._result(raw)

    def step(self, control):
        """兼容单控制调用；主程序使用step_plan，禁止把首控制维持整秒。"""
        current=self.snapshot()
        values = np.asarray(control, float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError('CARLA 控制必须是有限的[a,delta]')
        # 仅允许一个轨迹采样间隔等于重规划周期的旧调用者使用。
        if abs(float(self.config['mpc']['step_s'])-self.dt)>1e-8:
            raise ValueError('重规划周期包含多个控制段，必须提交完整step_plan')
        from .contracts import PlanResult
        return self.step_plan(PlanResult(current.frame_id,current.time,True,np.empty((0,4)),values[None,:]))

    def set_planned_trajectory(self, states):
        try:
            values = np.asarray(states, float)
            self._planned = values[:, :2].tolist() if values.ndim == 2 and values.shape[1] >= 2 and np.isfinite(values).all() else None
        except (ValueError, TypeError):
            self._planned = None

    def start_recording(self, path):
        if self._recorder is not None:
            raise RuntimeError('CARLA 录像已经启动')
        if self.bridge is None and not self._sensor_policy['rgb_enabled']:
            from .carla_sensor_policy import sensor_policy
            self.set_sensor_policy(sensor_policy(video=True,live=bool(self.live)))
        if not self._sensor_policy['rgb_enabled']:
            raise RuntimeError('相机已关闭，不能在SDK连接后临时开始录像')
        self._recorder = CameraVideoRecorder(self, path)
        return self._recorder

    def render(self):
        self.snapshot()
        if self._mosaic is None:
            return None
        from .carla_display import plan_mosaic_overlay
        visualization = self._raw.get('plan_visualization', {})
        image, segments = (plan_mosaic_overlay(self._mosaic, self._raw['cameras'], visualization)
            if self.cfg.get('display_plan_overlay',True) else (self._mosaic.copy(),0))
        self._display_info = {'plan_overlay':segments>0,'plan_segments':segments,
            'world_debug_enabled':bool(self.cfg.get('world_plan_debug',False)),
            'source_carla_frame':visualization.get('source_carla_frame'),
            'capture_carla_frame':self._raw['frame']}
        if self.live:
            import cv2
            cv2.imshow('CARLA 六视角实时画面', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return image

    def display_metadata(self):
        return dict(self._display_info)

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
