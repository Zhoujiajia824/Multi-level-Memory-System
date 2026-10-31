"""原帧反思队列、轮次落盘和经验验收。"""
from __future__ import annotations
import copy
import hashlib
import json
import logging
from pathlib import Path
import queue
import threading
import numpy as np
from .contracts import ExperienceRecord, VERSION, plain
from .models import build_prompt, parse_answer, CommunicationFailure
from .quality import evaluate, admissible
from .paths import filesystem_path

logger=logging.getLogger(__name__)


class EpisodeWriter:
    def __init__(self,root,episode_id):
        if not episode_id or Path(episode_id).name!=episode_id or episode_id in ('.','..'):
            raise ValueError('episode ID 必须是单个有效目录名')
        self.path=filesystem_path(root)/episode_id
        self.path.mkdir(parents=True,exist_ok=False)
        self.lock=threading.RLock()

    def append(self,name,payload):
        with self.lock,(self.path/f'{name}.jsonl').open('a',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(plain(payload),ensure_ascii=False,allow_nan=False)+'\n')

    def save(self,name,payload):
        target=self.path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.tmp')
        with self.lock:
            temporary.write_text(json.dumps(plain(payload),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            temporary.replace(target)
        return str(target.resolve())


def make_record(scene,reps,answer,quality,asset_path=None):
    # 完整真值存在不可变资产中；检索文本保持完整四段与可核对的局部状态。
    situation={'frame_id':scene.frame_id,'time':scene.time,'scenario':scene.scenario,
        '坐标':'local_goal使用历史原帧固定Frenet参考：delta_s_m相对该帧自车投影，d_m相对参考线绝对偏移；不可复制为当前目标',
        '原点世界姿态':[*scene.ego.position,scene.ego.heading],
        '情境':answer.explanation['情境'],'自车速度':float(np.linalg.norm(scene.ego.velocity)),
        '可见对象':[{'id':a.id,'类型':a.kind,'世界位置':plain(a.position),'速度':plain(a.velocity),'尺寸':[a.length,a.width]} for a in scene.actors if a.visible]}
    selected={'attention':answer.attention,'local_goal':answer.local_goal,'explanation':answer.explanation}
    assets={**scene.assets}
    if asset_path:
        assets['original_scene']=asset_path
    return ExperienceRecord(scene.frame_id,scene.frame_id,scene.time,VERSION,situation,selected,
                            plain(quality),assets,{},'active',1,
                            answer_protocol_version=answer.protocol_version,
                            goal_reference=plain(answer.goal_reference))


class ReflectionWorker:
    """一个活动任务和最多两个等待任务；每个原帧只接纳一次。"""
    def __init__(self,client,planner_factory,config,store,writer,write_enabled):
        self.client=client
        self.planner_factory=planner_factory
        self.config=config
        self.store=store
        self.writer=writer
        self.write_enabled=write_enabled
        self.queue=queue.Queue(maxsize=2)
        self.seen=set()
        self.closed=False
        self.lock=threading.Lock()
        self.counts=dict(accepted=0,queue_full=0,completed=0,rejected=0,failed=0,written=0)
        self.thread=threading.Thread(target=self._run,name='原帧反思',daemon=False)
        self.thread.start()

    def submit(self,task):
        with self.lock:
            if self.closed or task.scene.frame_id in self.seen or not self.client.enabled:
                return False
            self.seen.add(task.scene.frame_id)
            try:
                self.queue.put_nowait(copy.deepcopy(task))
            except queue.Full:
                self.counts['queue_full']+=1
                self.writer.append('reflection',{'frame_id':task.scene.frame_id,'status':'队列已满，拒收且不补处理'})
                return False
            self.counts['accepted']+=1
            return True

    def _run(self):
        while True:
            task=self.queue.get()
            if task is None:
                self.queue.task_done()
                return
            event={'frame_id':task.scene.frame_id,'time':task.scene.time,'status':'开始'}
            try:
                planner=self.planner_factory()
                original_reference=task.plan.diagnostics.get('reference_state_before')
                if original_reference is not None:
                    planner.restore_reference_state(original_reference)
                raw=self.client.call(build_prompt(task.scene,task.representations,reflection=task),[task.representations.bev_path])
                event['raw_answer']=raw
                answer=parse_answer(raw,task.scene,'reflection',self.config.get('models',{}).get('answer_max_characters',6000),
                                    reference=task.scene.diagnostics.get('goal_reference'),config=self.config)
                # planner 实例只属于后台，避免求解热启动与前台互相污染。
                plan=planner.solve(task.scene,task.predictions,answer)
                quality=evaluate(task.scene,plan,task.baseline,task.predictions,self.config)
                event.update(answer=plain(answer),plan=plain(plan),quality=plain(quality))
                if not admissible(answer,plan,quality,self.config['quality']['minimum_score']):
                    event['status']='验收不通过，拒收且不再次修正'
                    self.counts['rejected']+=1
                else:
                    event['status']='原帧验收通过'
                    self.counts['completed']+=1
                    if self.write_enabled:
                        identity=hashlib.sha256(task.scene.frame_id.encode('utf-8')).hexdigest()[:20]
                        task.scene.assets['reflection_acceptance_json']=self.writer.save(
                            f'reflection_acceptance/{identity}.json',
                            dict(event,version=task.scene.version,predictions=task.predictions,reference=task.baseline))
                        record=make_record(task.scene,task.representations,answer,quality,task.scene.assets.get('scene_json'))
                        self.store.write(record,task.representations)
                        self.counts['written']+=1
                        event['stored_record_id']=record.id
            except Exception as exc:
                self.counts['failed']+=1
                event.update(status='任务失败',error=str(exc) if isinstance(exc,(ValueError,CommunicationFailure,FileNotFoundError)) else type(exc).__name__)
                logger.warning('反思任务失败 %s：%s',task.scene.frame_id,event['error'])
            finally:
                try:
                    self.writer.append('reflection',event)
                except OSError:
                    logger.error('反思审计写入失败：%s',task.scene.frame_id)
                    self.counts['failed']+=1
                finally:
                    self.queue.task_done()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed=True
        # 先排空已接纳请求；不能以短超时放弃在途模型调用。
        self.queue.join()
        self.queue.put(None)
        self.thread.join()
