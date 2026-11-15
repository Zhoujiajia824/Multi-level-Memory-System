"""异步答案的原帧验收；独立规划器不占用当前控制线程。"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import logging
import queue
import threading
import time

from .contracts import ANSWER_PROTOCOL_VERSION, ReflectionTask
from .planner_factory import create_planner
from .planning_context import prepare_planning_context
from .quality import evaluate, admissible, reflection_reasons
from .representations import render_goal_overlay
from .runtime import make_record

logger = logging.getLogger(__name__)


@dataclass
class AsyncAdmissionTask:
    result: object
    predictions: object
    reference_state: object = None


class AsyncAdmissionWorker:
    """一个活动任务及最多两个等待任务；忙碌时明确拒收，不伪造验收。"""
    def __init__(self, config, store, writer, reflection_worker, write_enabled=False,
                 memory_mode='collect', planner_factory=None, queue_capacity=2):
        if type(queue_capacity) is not int or queue_capacity < 1:
            raise ValueError('原帧验收等待容量必须为正整数')
        self.config = copy.deepcopy(config)
        self.store, self.writer, self.reflection_worker = store, writer, reflection_worker
        self.write_enabled, self.memory_mode = write_enabled, memory_mode
        self.planner_factory = planner_factory or (lambda: create_planner(self.config))
        self.queue = queue.Queue(maxsize=queue_capacity)
        self.lock = threading.Lock()
        self.closed, self.seen = False, set()
        self.counts = dict(accepted=0, queue_full=0, completed=0, failed=0,
                           quality_valid=0, quality_total=0, safety_alarms=0,
                           reflection_submitted=0, reflection_not_accepted=0,
                           written_fast=0, rejected=0)
        self.timings = dict(mpc=[], reference=[], quality=[], goal_overlay=[], total=[])
        self.thread = threading.Thread(target=self._run, name='异步原帧验收', daemon=False)
        self.thread.start()

    def submit(self, task):
        result, predictions = task.result, task.predictions
        source = result.task.scene
        if (predictions.frame_id != source.frame_id or predictions.time != source.time or
                result.task.representations.frame_id != source.frame_id or
                result.task.representations.time != source.time):
            raise ValueError('异步验收必须使用同一原帧场景、表示与预测')
        if result.raw is None or result.status not in ('ok', 'invalid_answer'):
            raise ValueError('未调用或没有原始回答的任务不能伪造为原帧验收任务')
        answer = result.answer
        if result.status == 'ok' and (answer is None or answer.frame_id != source.frame_id or
                answer.time != source.time or answer.goal_reference != result.task.reference):
            raise ValueError('异步验收答案必须绑定已冻结的原帧与目标参考')
        if result.status == 'invalid_answer' and answer is not None:
            raise ValueError('非法答案事件不能携带可用语义答案')
        with self.lock:
            if self.closed or source.frame_id in self.seen:
                return False
            self.seen.add(source.frame_id)
            try:
                self.queue.put_nowait(copy.deepcopy(task))
            except queue.Full:
                self.counts['queue_full'] += 1
                self.writer.append('async_admission', dict(frame_id=source.frame_id,
                    time=source.time, status='原帧验收队列已满，拒收且不补处理'))
                return False
            self.counts['accepted'] += 1
            return True

    def _run(self):
        while True:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                return
            started = time.perf_counter()
            result, predictions = task.result, task.predictions
            source_task = result.task
            scene, reps, answer = source_task.scene, source_task.representations, result.answer
            event = dict(frame_id=scene.frame_id, time=scene.time,
                         request_version=source_task.request_version,
                         status='开始原帧验收', controls_current_vehicle=False)
            try:
                planner = self.planner_factory()
                if task.reference_state is not None:
                    planner.restore_reference_state(task.reference_state)
                if answer is None:
                    planner.clear_semantic_guidance()
                stage = time.perf_counter()
                scoped, predictions = prepare_planning_context(scene,self.config,answer,existing=predictions)
                plan = planner.solve(scoped, predictions, answer)
                event['mpc_s'] = time.perf_counter() - stage
                self.timings['mpc'].append(event['mpc_s'])
                identity = hashlib.sha256(scene.frame_id.encode('utf-8')).hexdigest()[:20]
                stage = time.perf_counter()
                try:
                    overlay_path = self.writer.path / 'async_admission' / (identity + '_goal_overlay.png')
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    render_goal_overlay(scene, plan).save(overlay_path)
                except Exception as exc:
                    # 诊断图不能取消有效的原帧规划/质量验收，更不能影响前台控制。
                    event['goal_overlay_error'] = type(exc).__name__
                    logger.warning('异步原帧 %s：目标诊断图生成失败（%s），继续验收',
                                   scene.frame_id, type(exc).__name__)
                else:
                    scene.assets['goal_overlay_png'] = str(overlay_path.resolve())
                    event['goal_overlay_path'] = scene.assets['goal_overlay_png']
                event['goal_overlay_s'] = time.perf_counter() - stage
                self.timings['goal_overlay'].append(event['goal_overlay_s'])
                # 参照从未获得旧目标/关注，不复用主规划器的语义状态。
                baseline_planner = self.planner_factory()
                stage = time.perf_counter()
                base_scene,base_predictions=prepare_planning_context(scene,self.config,existing=predictions)
                baseline = baseline_planner.solve(base_scene, base_predictions, None)
                event['reference_s'] = time.perf_counter() - stage
                self.timings['reference'].append(event['reference_s'])
                stage = time.perf_counter()
                quality = evaluate(scoped, plan, baseline, predictions, self.config)
                event['quality_s'] = time.perf_counter() - stage
                self.timings['quality'].append(event['quality_s'])
                self.counts['quality_total'] += 1
                self.counts['quality_valid'] += int(quality.valid)
                self.counts['safety_alarms'] += int(quality.precollision or quality.collision)
                event.update(plan=plan, baseline=baseline, quality=quality,
                             selected_answer=answer, raw_fast_answer=result.raw)
                # 沿用同步原帧证据字段，独立标记后台验收；不能把前台当前
                # 轨迹/分数拷来给旧答案背书，也不能丢失原有质量审计记录。
                self.writer.append('quality', dict(frame_id=scene.frame_id, time=scene.time,
                    quality=quality, reference_plan=baseline, source='async_source_frame'))
                scene.assets['planning_evidence_json'] = self.writer.save(
                    'async_admission/' + identity + '.json', dict(event,
                        version=scene.version, answer_protocol_version=ANSWER_PROTOCOL_VERSION,
                        predictions=predictions, reference=baseline,
                        retrieval=source_task.retrieval.scores,
                        goal_reference=source_task.reference))
                triggers = reflection_reasons(quality, bool(result.errors),
                                             self.config['quality']['minimum_score'])
                if triggers:
                    accepted = self.reflection_worker.submit(ReflectionTask(
                        scene, reps, predictions, answer, result.raw,
                        triggers + list(result.errors), plan, baseline, quality,
                        source_task.retrieval))
                    self.counts['reflection_submitted' if accepted else 'reflection_not_accepted'] += 1
                    event['status'] = '原帧提交反思' if accepted else '原帧反思未接纳，不入库'
                elif (self.write_enabled and self.memory_mode == 'collect' and
                      admissible(answer, plan, quality, self.config['quality']['minimum_score'])):
                    record = make_record(scene, reps, answer, quality, scene.assets.get('scene_json'))
                    self.store.write(record, reps)
                    self.counts['written_fast'] += 1
                    event.update(status='快答案原帧验收通过并入库', record_id=record.id)
                else:
                    self.counts['rejected'] += 1
                    event['status'] = '原帧验收完成，未满足当前入库条件或写入关闭'
                self.counts['completed'] += 1
            except Exception as exc:
                self.counts['failed'] += 1
                event.update(status='原帧验收失败，不入库', error_type=type(exc).__name__)
            finally:
                event['wall_seconds'] = time.perf_counter() - started
                self.timings['total'].append(event['wall_seconds'])
                try:
                    self.writer.append('async_admission', event)
                except Exception:
                    # 单条审计落盘故障不能杀死线程，导致剩余任务收尾永久等待。
                    self.counts['failed'] += 1
                finally:
                    self.queue.task_done()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.queue.join()
        self.queue.put(None)
        self.thread.join()
