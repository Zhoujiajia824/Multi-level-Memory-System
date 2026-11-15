"""后台快模型：一个在途任务和一个可被更新帧替换的待处理任务。

这里只处理冻结原帧的调用与协议校验，不接触规划器或当前控制。
完成结果保留到前台 poll；关闭时等待已接纳任务收尾，不丢弃返回结果。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import logging
import math
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time

from .contracts import plain
from .models import CommunicationFailure, parse_answer


logger = logging.getLogger(__name__)
ASYNC_INFERENCE_VERSION = 'async-fast-source-frame-1.0'


@dataclass(frozen=True)
class AsyncInferenceTask:
    scene: object
    representations: object
    prompt: str
    images: list
    retrieval: object
    request_version: int = 0
    reference: dict | None = None

    @property
    def frame_id(self):
        return self.scene.frame_id

    @property
    def time(self):
        return float(self.scene.time)

    @property
    def source_order(self):
        return (self.time, self.request_version)


@dataclass
class AsyncInferenceResult:
    task: AsyncInferenceTask
    status: str
    sequence: int
    raw: str | None = None
    answer: object = None
    errors: list = field(default_factory=list)
    error_type: str | None = None
    wall_seconds: float = 0.0
    image_sha256: list = field(default_factory=list)
    audit_error: str | None = None
    version: str = ASYNC_INFERENCE_VERSION

    @property
    def source_order(self):
        return self.task.source_order

    def event(self):
        """持久化原帧身份和精确输入；大场景由已有原帧资产引用核查。"""
        reference = self.task.reference or {}
        scene = self.task.scene
        retrieval = self.task.retrieval
        return dict(version=self.version, frame_id=self.task.frame_id, time=self.task.time,
            request_version=self.task.request_version, source_order=list(self.source_order),
            reference_id=reference.get('reference_id'), reference_version=reference.get('reference_version'),
            sequence=self.sequence, status=self.status, prompt=self.task.prompt,
            images=list(self.task.images), image_sha256=list(self.image_sha256),
            source_scene_asset=getattr(scene, 'assets', {}).get('scene_json'),
            snapshot_id=getattr(retrieval, 'snapshot_id', None), raw=self.raw, answer=plain(self.answer),
            errors=list(self.errors), error_type=self.error_type, wall_seconds=self.wall_seconds,
            audit_error=self.audit_error)


@dataclass
class _QueuedTask:
    task: AsyncInferenceTask
    sequence: int
    directory: Path
    frozen_images: list
    hashes: list


class AsyncFastModelWorker:
    """复用已有 VisionClient；应用层重试仍只由该客户端负责。

    submit 不等待模型推理，但会复制当前图像和上下文以冻结输入。
    poll 返回所有已就绪事件而不等待推理；调用方按 status 区分答案和拒收。
    counts 为线程安全快照，completed 表示已结束的真实调用数。
    """
    def __init__(self, client, config=None, writer=None, parser=None):
        self.client = client
        self.config = copy.deepcopy(config or {})
        self.writer = writer
        self.parser = parse_answer if parser is None else parser
        self.enabled = bool(client.enabled)
        self._condition = threading.Condition()
        self._ready = queue.Queue()
        self._pending = None
        self._active = None
        self._closing = False
        self._closed = False
        self._latest_order = None
        self._sequence = 0
        self._counts = {key: 0 for key in (
            'submitted', 'accepted', 'started', 'replaced_pending', 'completed', 'succeeded',
            'communication_failed', 'invalid_answer', 'failed', 'rejected_disabled',
            'rejected_closed', 'rejected_stale', 'audit_failed')}
        self._temporary = tempfile.TemporaryDirectory(prefix='ep_driver_async_fast_') if self.enabled else None
        self._root = Path(self._temporary.name).resolve() if self._temporary else None
        self.thread = None
        if self.enabled:
            self.thread = threading.Thread(target=self._run, name='async-fast-model', daemon=False)
            self.thread.start()

    @property
    def counts(self):
        with self._condition:
            return {**self._counts, 'in_flight': int(self._active is not None),
                    'pending': int(self._pending is not None)}

    def _freeze(self, task, sequence):
        if not isinstance(task, AsyncInferenceTask):
            raise ValueError('后台快模型任务必须为 AsyncInferenceTask')
        if not isinstance(task.frame_id, str) or not task.frame_id:
            raise ValueError('后台快模型任务缺少源帧ID')
        if not math.isfinite(task.time) or task.time < 0:
            raise ValueError('后台快模型源帧时间必须为有限非负秒值')
        if type(task.request_version) is not int or task.request_version < 0:
            raise ValueError('后台快模型请求版本必须为非负整数')
        if not isinstance(task.prompt, str) or not task.prompt.strip():
            raise ValueError('后台快模型任务缺少完整提示词')
        if not isinstance(task.images, (list, tuple)) or not task.images:
            raise ValueError('后台快模型任务至少需要一张源帧图像')
        reps = task.representations
        if (getattr(reps, 'frame_id', None) != task.frame_id or
                not math.isclose(float(reps.time), task.time, rel_tol=0., abs_tol=1e-6)):
            raise ValueError('后台快模型表示与源帧身份/时间不一致')
        reference = task.reference if task.reference is not None else task.scene.diagnostics.get('goal_reference')
        if (not isinstance(reference, dict) or reference.get('frame_id') != task.frame_id or
                not math.isclose(float(reference.get('time', math.nan)), task.time, rel_tol=0., abs_tol=1e-6)):
            raise ValueError('后台快模型缺少与源帧一致的冻结目标参考')
        if reference.get('request_version', task.request_version) != task.request_version:
            raise ValueError('后台快模型请求版本与原帧参考不一致')
        frozen = copy.deepcopy(task)
        object.__setattr__(frozen, 'reference', copy.deepcopy(reference))
        object.__setattr__(frozen, 'images', tuple(str(Path(path).resolve()) for path in task.images))
        directory = self._root / str(sequence)
        directory.mkdir()
        copied, hashes = [], []
        try:
            for index, source in enumerate(frozen.images):
                original = Path(source)
                raw = original.read_bytes()
                if not raw:
                    raise ValueError('后台快模型源帧图像为空')
                suffix = original.suffix.lower() or '.png'
                destination = directory / ('%03d' % index + suffix)
                destination.write_bytes(raw)
                copied.append(str(destination))
                hashes.append(hashlib.sha256(raw).hexdigest())
        except BaseException:
            self._dispose_directory(directory)
            raise
        return _QueuedTask(frozen, sequence, directory, copied, hashes)

    def _dispose_directory(self, directory):
        # 只删除本worker分配的直接子目录，永不清理调用方图像资产。
        directory = Path(directory).resolve()
        if self._root is None or directory.parent != self._root:
            raise RuntimeError('后台快模型图像临时目录越界')
        shutil.rmtree(directory)

    def _publish(self, result):
        if self.writer is not None:
            try:
                self.writer.append('async_fast_model', result.event())
            except Exception as exc:
                # 不回显异常文本，避免文件系统或第三方异常携带凭据。
                result.audit_error = type(exc).__name__
                with self._condition:
                    self._counts['audit_failed'] += 1
                logger.error('后台快模型审计写入失败：%s，类型=%s', result.task.frame_id, result.audit_error)
        self._ready.put(result)

    def submit(self, task):
        """接纳最新源帧；忙碌时仅替换尚未开始的待处理任务。"""
        replaced = None
        with self._condition:
            self._counts['submitted'] += 1
            if self._closing or self._closed:
                self._counts['rejected_closed'] += 1
                return False
            if not self.enabled:
                self._counts['rejected_disabled'] += 1
                return False
            self._sequence += 1
            queued = self._freeze(task, self._sequence)
            if self._latest_order is not None and queued.task.source_order <= self._latest_order:
                self._counts['rejected_stale'] += 1
                self._dispose_directory(queued.directory)
                return False
            self._latest_order = queued.task.source_order
            if self._pending is not None:
                previous = self._pending
                replaced = AsyncInferenceResult(previous.task, 'pending_replaced', previous.sequence,
                    errors=['待处理源帧被更新源帧替换；未向模型发起调用'], image_sha256=previous.hashes)
                self._dispose_directory(previous.directory)
                self._counts['replaced_pending'] += 1
            self._pending = queued
            self._counts['accepted'] += 1
            # 在交出提交锁前发布替换事件，保证并发close返回时该审计已收尾。
            if replaced is not None:
                self._publish(replaced)
            self._condition.notify_all()
        return True

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closing)
                if self._pending is None:
                    return
                queued, self._pending = self._pending, None
                self._active = queued
                self._counts['started'] += 1
            result = AsyncInferenceResult(queued.task, 'failed', queued.sequence, image_sha256=queued.hashes)
            started = time.monotonic()
            stage = 'communication'
            try:
                result.raw = self.client.call(queued.task.prompt, queued.frozen_images)
                stage = 'parse'
                result.answer = self.parser(result.raw, queued.task.scene, source='fast',
                    max_characters=self.config.get('models', {}).get('answer_max_characters', 6000),
                    reference=queued.task.reference, config=self.config)
                result.status = 'ok'
            except (CommunicationFailure, TimeoutError, ConnectionError) as exc:
                result.status, result.error_type = 'communication_error', type(exc).__name__
                result.errors = ['后台快模型通信失败；未生成可用答案']
            except ValueError as exc:
                result.status = 'invalid_answer' if stage == 'parse' else 'communication_error'
                result.error_type = type(exc).__name__
                result.errors = ['模型答案未通过原帧结构或目标校验' if stage == 'parse' else '后台快模型调用失败']
            except Exception as exc:
                result.status, result.error_type = 'failed', type(exc).__name__
                result.errors = ['后台快模型任务发生异常；未使用该答案']
            finally:
                result.wall_seconds = time.monotonic() - started
                try:
                    self._dispose_directory(queued.directory)
                except OSError as exc:
                    result.audit_error = type(exc).__name__
                    logger.warning('后台快模型临时图像清理失败：%s', result.task.frame_id)
                with self._condition:
                    self._counts['completed'] += 1
                    key = {'ok': 'succeeded', 'communication_error': 'communication_failed',
                           'invalid_answer': 'invalid_answer'}.get(result.status, 'failed')
                    self._counts[key] += 1
                self._publish(result)
                with self._condition:
                    self._active = None
                    self._condition.notify_all()

    def poll(self):
        """返回当前已就绪事件；没有结果时立即返回空列表。"""
        results = []
        while True:
            try:
                results.append(self._ready.get_nowait())
            except queue.Empty:
                return results

    def close(self):
        """拒绝新任务，排空在途及最后待处理任务，保留未poll的完成事件。"""
        with self._condition:
            if self._closed:
                return
            self._closing = True
            self._condition.notify_all()
        if self.thread is not None:
            if threading.current_thread() is self.thread:
                raise RuntimeError('后台快模型线程不能等待自身关闭')
            self.thread.join()
        if self._temporary is not None:
            self._temporary.cleanup()
        with self._condition:
            self._closed = True
