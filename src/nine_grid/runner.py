"""按真实仿真时间调度检索、重规划及同步/异步模型。"""
from __future__ import annotations
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
import numpy as np
from shapely.geometry import box, Polygon
from .contracts import ANSWER_PROTOCOL_VERSION, ReflectionTask, RetrievalResult, plain
from .goals import build_goal_reference
from .highway import HighwayAdapter
from .memory import MemoryStore
from .models import VisionClient, CommunicationFailure, parse_answer, build_prompt, local_line, model_image_paths
from .planner_factory import create_planner
from .planning_context import prepare_planning_context
from .prediction import Predictor
from .quality import evaluate, admissible, reflection_reasons, execution_audit
from .representations import build_representations, DINOExtractor, NineGridBuilder, render_goal_overlay
from .retrieval import retrieve
from .runtime import EpisodeWriter, ReflectionWorker, make_record
from .async_inference import AsyncInferenceTask, AsyncFastModelWorker
from .async_admission import AsyncAdmissionTask, AsyncAdmissionWorker
from .carla_sensor_policy import sensor_policy
from .carla_execution import CarlaSourceClock
from .control_clock import execution_prefix_count

logger=logging.getLogger(__name__)


def _timing_summary(values):
    if not values:return {'count':0}
    return dict(count=len(values),p50_ms=float(np.percentile(values,50))*1000,
                p95_ms=float(np.percentile(values,95))*1000,max_ms=float(max(values))*1000)


def _actual_control_evidence(result):
    """只读取执行器报告的物理控制，不把规划首项冒充实际执行。"""
    runtime=result.get('runtime') or {}
    evidence=runtime.get('last_applied_control') or result.get('last_applied_control') or result.get('control_response')
    if isinstance(evidence,dict):
        control=evidence.get('control')
        if control is None and 'acceleration_mps2' in evidence and 'steering_rad_left' in evidence:
            control=[evidence['acceleration_mps2'],evidence['steering_rad_left']]
    else:
        control=evidence
    try:
        values=np.asarray(control,float)
    except (TypeError,ValueError,OverflowError):
        values=None
    if values is None or values.shape!=(2,) or not np.isfinite(values).all():
        return None,evidence
    return values.tolist(),evidence


def _bind_previous_control(scene,cached_control):
    """当前冻结观测的控制优先；异步执行后的新证据不能被上一轮缓存覆盖。"""
    if 'previous_control' in scene.diagnostics:
        value=scene.diagnostics['previous_control']
        source='current_snapshot'
    elif cached_control is not None:
        value=cached_control
        source='previous_execution_result_fallback'
    else:
        return None
    try:
        control=np.asarray(value,float)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError('本帧前次执行控制必须是有限[a,delta]，不能以旧缓存替代非法证据') from exc
    if control.shape!=(2,) or not np.isfinite(control).all():
        raise ValueError('本帧前次执行控制必须是有限[a,delta]，不能以旧缓存替代非法证据')
    scene.diagnostics['previous_control']=control.tolist()
    scene.diagnostics['previous_control_source']=source
    scene.diagnostics['previous_control_frame_id']=scene.frame_id
    return control.tolist()


def capture_payload(scene):
    """冻结可达局部地图，完整地图在轮次资产中只保存一次。"""
    x,y=scene.ego.position
    region=box(x-150,y-150,x+150,y+150)
    lanes=[]
    for lane in scene.lanes:
        poly=Polygon(lane.polygon).intersection(region)
        if poly.is_empty:
            continue
        payload=plain(lane)
        payload['center']=plain(local_line(lane.center,scene.ego.position,150))
        payload['polygon_wkt']=poly.wkt
        payload.pop('polygon',None)
        payload['boundaries']=[{**plain(b),'points':plain(local_line(b['points'],scene.ego.position,150))} for b in lane.boundaries]
        lanes.append(payload)
    return dict(frame_id=scene.frame_id,time=scene.time,version=scene.version,coordinates=scene.coordinates,
        ego=plain(scene.ego),actors=plain(scene.actors),lanes=lanes,route=plain(scene.route),
        route_lane_ids=scene.route_lane_ids,drivable_wkt=scene.drivable.intersection(region).wkt,
        scenario=scene.scenario,rules=plain(scene.rules),diagnostics=plain(scene.diagnostics),
        sensor_assets=plain({key:value for key,value in scene.assets.items() if key in (
            'camera_images','camera_manifest_json','original_camera_images','depth_images','current_state_json')}),
        global_map_asset=scene.assets.get('global_map_json'),map_crop='局部显示为自车中心前后左右各150米；完整地图另存为不可变资产')


def run_episode(config,model_settings,episode_id,episodes_dir,memory_read_dir=None,memory_write_dir=None,
                retrieve_enabled=False,write_enabled=False,memory_mode='collect',seed=1,live=False,video=False,
                fast_client=None,reflection_client=None,extractor=None,adapter_factory=None):
    fast=fast_client if fast_client is not None else VisionClient('fast',model_settings['fast'])
    reflection=reflection_client if reflection_client is not None else VisionClient('reflection',model_settings['reflection'])
    # 基础模式仍保留低频表示与质量诊断；语义和记忆开关不由诊断启用。
    pure_base=not (fast.enabled or reflection.enabled or retrieve_enabled or write_enabled)
    execution_mode='pure_base' if pure_base else 'full_pipeline'
    planner_backend=config.get('planner',{}).get('backend','mpc')
    runtime_mode=config.get('carla',{}).get('runtime_mode','sync') if config.get('platform','highway')=='carla' else 'sync'
    if runtime_mode not in ('sync','async'):
        raise ValueError('CARLA运行模式只支持sync或async')
    asynchronous=runtime_mode=='async'
    planning_period=1./float(config['clock']['planner_hz'])
    source_period=1./float(config['clock']['fast_hz'])
    carla_source_clock=(CarlaSourceClock(config['clock'].get('physics_hz',50),config['clock']['fast_hz'])
                        if config.get('platform','highway')=='carla' else None)
    execution_count=execution_prefix_count(config)
    diagnostic_config=config.get('diagnostics',{})
    diagnostics_enabled=bool(diagnostic_config.get('enabled',True))
    representations_enabled=diagnostics_enabled or not pure_base
    quality_enabled=diagnostics_enabled or not pure_base
    skipped_stages=[] if diagnostics_enabled else ['base_diagnostics_explicitly_disabled']
    # 本地模型导入/配置先于图像解码，且绝不以随机特征替代真实权重。
    if (retrieve_enabled or write_enabled or (diagnostics_enabled and diagnostic_config.get('dino',True))) and extractor is None:
        dino=config.get('dino',{})
        extractor=DINOExtractor(dino.get('model','models/dinov2-base'),device=dino.get('device'),
                                local_files_only=dino.get('local_files_only',True))
    store=MemoryStore(memory_read_dir if retrieve_enabled else None,memory_write_dir if write_enabled else None)
    records=store.snapshot()
    writer=EpisodeWriter(episodes_dir,episode_id)
    file_handler=logging.FileHandler(writer.path/'run.log',encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logging.getLogger().addHandler(file_handler)
    try:
        writer.save('configuration.json',dict(parameters=config,
            models={role:getattr(client,'settings',model_settings[role])
                    for role,client in (('fast',fast),('reflection',reflection))},seed=seed,
            effective_parameters_sha256=hashlib.sha256(json.dumps(plain(config),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest(),
            numerical_threads={name:os.environ.get(name) for name in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            answer_protocol_version=ANSWER_PROTOCOL_VERSION,
            execution_mode=execution_mode,runtime_mode=runtime_mode,skipped_stages=skipped_stages,
            retrieval_enabled=retrieve_enabled,write_enabled=write_enabled,memory_mode=memory_mode,
            memory_read_dir=str(memory_read_dir),memory_write_dir=str(memory_write_dir),snapshot_id=store.snapshot_id,
            snapshot_records=[{'id':r['id'],'revision':r.get('revision',1)} for r in records],excluded_records=store.audit))
    except BaseException:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        raise
    adapter=None
    worker=None
    async_fast=None
    admission_worker=None
    consume_async_results=None
    source_contexts={}
    video_recorder=None
    summary=dict(episode_id=episode_id,seed=seed,scenario=config.get('scenario','highway'),
        simulated_seconds=0.,steps=0,collision=False,arrival=False,planning_failures=0,
        safety_alarms=0,execution_alarms=0,quality_valid=0,quality_total=0,fast_calls=0,communication_failures=0,
        invalid_answers=0,retrieval_hits=0,written_fast=0,offroad_steps=0,
        fallback_controls=0,soft_safety_violation_steps=0,initialization_candidate_steps=0,
        safe_braking_steps=0,emergency_steps=0,
        termination_reason='initialization_failure',progress_m=0.,execution_mode=execution_mode,runtime_mode=runtime_mode,
        planner_backend=planner_backend,reference_backend=planner_backend,reflection_planning_backend=planner_backend,
        fast_requests=0,source_periods_skipped=0,planning_periods_skipped=0,
        async_stale_results=0,async_pending_replaced=0,async_submission_rejected=0,
        async_goal_passed_on_arrival=0,
        skipped_stages=skipped_stages,
        reflection=dict(accepted=0,queue_full=0,completed=0,rejected=0,failed=0,written=0))
    start_wall=time.monotonic()
    logger.info('开始 %s，场景=%s，seed=%s，检索=%s，写入=%s',episode_id,config.get('scenario'),seed,retrieve_enabled,write_enabled)
    logger.info('控制/独立参照/原帧反思规划后端均为 %s',planner_backend.upper())
    if pure_base:
        logger.info('纯基础规划：模型/检索/写入关闭；低频表示与参照评分=%s，规划频率=%sHz',diagnostics_enabled,config['clock']['planner_hz'])
    used=[]
    timing_samples={'mpc_cold':[],'mpc_warm':[],'prediction':[],'decision_pipeline':[],
                    'retrieval':[],'snapshot':[],'snapshot_build':[],'observation_wait':[],
                    'execution_audit':[],'physics_step':[],
                    'video_display':[],'control_cycle':[],'world_plan':[],
                    'representations':[],'reference':[],'quality':[],'goal_overlay':[],
                    'simulation_cycle':[],'wall_cycle_interval':[]}
    mpc_overruns=0
    optimizer_budget_overruns=0
    try:
        writer.append('memory_snapshot',dict(snapshot_id=store.snapshot_id,read_enabled=retrieve_enabled,
            accepted=[{'id':record['id'],'revision':record.get('revision',1)} for record in records],
            excluded=store.audit))
        if store.audit:
            reasons='；'.join(dict.fromkeys(str(item.get('reason','原因未提供')) for item in store.audit[:3]))
            logger.warning('轮初记忆快照接纳%d条、排除%d条；排除原因示例：%s',len(records),len(store.audit),reasons[:400])
        else:
            logger.info('轮初记忆快照接纳%d条、排除0条',len(records))
        if adapter_factory is None:
            if config.get('platform','highway') == 'carla':
                from .carla_adapter import CarlaAdapter
                adapter_factory=CarlaAdapter
            else:
                adapter_factory=HighwayAdapter
        adapter=adapter_factory(config,seed,episode_id,live)
        if hasattr(adapter,'set_output_directory'):
            adapter.set_output_directory(writer.path)
        if hasattr(adapter,'set_representation_capture_enabled'):
            adapter.set_representation_capture_enabled(representations_enabled)
        if hasattr(adapter,'set_sensor_policy'):
            policy=sensor_policy(bool(fast.enabled),bool(reflection.enabled),bool(video),bool(live))
            adapter.set_sensor_policy(policy)
            writer.append('sensor_policy',policy)
        if video:video_recorder=adapter.start_recording(writer.path/'video.mp4')
        planner=create_planner(config)
        reference_planner=create_planner(config) if quality_enabled else None
        predictor=Predictor(config)
        grids=NineGridBuilder(config) if representations_enabled else None
        if not pure_base:
            worker=ReflectionWorker(reflection,lambda:create_planner(config),config,store,writer,write_enabled)
        if asynchronous and fast.enabled:
            async_fast=AsyncFastModelWorker(fast,config=config,writer=writer)
            admission_worker=AsyncAdmissionWorker(config,store,writer,worker,write_enabled,memory_mode)
        active_answer=None
        active_retrieval=RetrievalResult('',snapshot_id=store.snapshot_id)
        active_answer_retrieval=active_retrieval
        latest_semantic_order=None
        newly_received_goal=None
        def consume_async_results(drain_only=False):
            nonlocal active_answer,active_answer_retrieval,latest_semantic_order,newly_received_goal
            if async_fast is None:
                return
            for completed in async_fast.poll():
                context=source_contexts.pop(completed.task.request_version,None)
                event=completed.event()
                event.update(phase='drain_only' if drain_only else 'planning_cycle',
                             runtime_stopped=bool(drain_only))
                if completed.status=='pending_replaced':
                    summary['async_pending_replaced']+=1
                    event['current_control_action']='待处理帧替换，未调用模型且不验收入库'
                    writer.append('model_outputs',event)
                    continue
                summary['fast_calls']+=1
                if completed.status=='invalid_answer':summary['invalid_answers']+=1
                if completed.status in ('communication_error','failed'):summary['communication_failures']+=1
                newer=latest_semantic_order is None or completed.source_order>latest_semantic_order
                event['accepted_for_next_planning']=bool(not drain_only and newer and completed.status=='ok')
                if newer and not drain_only:
                    latest_semantic_order=completed.source_order
                    if completed.status=='ok':
                        active_answer=completed.answer
                        newly_received_goal=completed.answer.goal_id
                        active_answer_retrieval=completed.task.retrieval
                    else:
                        active_answer=None
                        planner.clear_semantic_guidance()
                elif not newer:
                    summary['async_stale_results']+=1
                    event['current_control_action']='过期结果不覆盖较新语义状态'
                if drain_only:
                    event['current_control_action']='本轮世界已停止，仅做原帧验收，不再进入当前控制'
                writer.append('model_outputs',event)
                if completed.raw is not None and completed.status in ('ok','invalid_answer'):
                    if context is None:
                        writer.append('async_admission',dict(frame_id=completed.task.frame_id,
                            time=completed.task.time,status='原帧预测上下文缺失，拒收且不伪造验收'))
                    else:
                        admission_worker.submit(AsyncAdmissionTask(completed,context[0],context[1]))
        first_s=None
        first_time=None
        previous_scene_time=None
        previous_control=None
        previous_cycle_wall=None
        next_source_time=None
        duration=float(config.get('duration',config['clock']['episode_seconds']))
        step=0
        def record_bridge_observation(result,phase,frame_id):
            bridge_timing=result.get('bridge_timing') or {}
            runtime=result.get('runtime') or {}
            if bridge_timing or runtime:
                control,evidence=_actual_control_evidence(result)
                writer.append('bridge_timing',dict(frame_id=frame_id,time=result['time'],phase=phase,
                    bridge_timing=bridge_timing,runtime=runtime,
                    applied_control=control,actual_control=evidence))
                summary['bridge_timing']=copy.deepcopy(bridge_timing)
                summary['carla_runtime']=copy.deepcopy(runtime)
        while True:
            cycle_started=time.perf_counter()
            if previous_cycle_wall is not None:
                timing_samples['wall_cycle_interval'].append(cycle_started-previous_cycle_wall)
            previous_cycle_wall=cycle_started
            timing=dict(representations_s=0.,model_s=0.,reference_s=0.,quality_s=0.,retrieval_s=0.,
                        goal_overlay_s=0.,execution_audit_s=0.,physics_step_s=0.,video_display_s=0.)
            scene=adapter.begin_cycle() if hasattr(adapter,'begin_cycle') else None
            if scene is None:
                scene=adapter.snapshot()
            timing['snapshot_s']=time.perf_counter()-cycle_started
            timing_samples['snapshot'].append(timing['snapshot_s'])
            adapter_timing=scene.diagnostics.get('adapter_timing',{})
            timing['observation_wait_s']=float(adapter_timing.get('observation_wait_s',0.))
            timing['snapshot_build_s']=float(adapter_timing.get('snapshot_build_s',timing['snapshot_s']))
            timing_samples['observation_wait'].append(timing['observation_wait_s'])
            timing_samples['snapshot_build'].append(timing['snapshot_build_s'])
            # 标签只影响模型阅读图；不删减对象或改变DINO使用的无字底图。
            scene.diagnostics['bev_label_options']=copy.deepcopy(config.get('bev',{}).get('labels',{}))
            previous_control=_bind_previous_control(scene,previous_control)
            if first_time is None:
                first_time=float(scene.time)
                next_source_time=first_time
            cycle_result=scene.diagnostics.get('cycle_result')
            cycle_terminal=bool(cycle_result is not None and cycle_result.get('terminated'))
            if not math.isfinite(float(scene.time)) or (previous_scene_time is not None and
                    (scene.time<previous_scene_time or (scene.time==previous_scene_time and not cycle_terminal))):
                raise RuntimeError('仿真快照时间未严格推进；不能按循环次数伪造仿真时钟')
            if previous_scene_time is not None and scene.time>previous_scene_time:
                elapsed=float(scene.time-previous_scene_time)
                timing_samples['simulation_cycle'].append(elapsed)
                summary['planning_periods_skipped']+=max(0,int(math.floor(elapsed/planning_period+1e-7))-1)
            previous_scene_time=float(scene.time)
            # 异步执行器在两次规划之间继续推进；终态来自本次真实世界观测，
            # 不能等到提交下一条规划后才发现上一条轨迹已经碰撞或到达。
            if cycle_result is not None:
                record_bridge_observation(cycle_result,'begin_cycle',scene.frame_id)
            if cycle_result is not None and cycle_result.get('terminated'):
                summary.update(simulated_seconds=cycle_result['time'],collision=cycle_result['collision'],
                    arrival=cycle_result['arrival'],termination_reason=cycle_result['reason'],
                    progress_m=(cycle_result['progress']-first_s) if first_s is not None else 0.)
                if cycle_result.get('collision'):
                    writer.save('accident_review.json',dict(status='待人工复查，不自动停用经验',
                        actual=cycle_result,associated=used))
                # 异步终帧在begin_cycle才出现；保存/显示该真实终帧后再暂停排空。
                display_started=time.perf_counter()
                if video_recorder is not None:
                    video_recorder._capture_frame()
                elif live:
                    adapter.render()
                display_seconds=time.perf_counter()-display_started
                if video_recorder is not None or live:
                    timing_samples['video_display'].append(display_seconds)
                writer.append('terminal_observation',dict(frame_id=scene.frame_id,time=scene.time,
                    actual=cycle_result,video_display_s=display_seconds))
                break
            # CARLA终态由冻结观测的物理帧计数给出，不另用浮点秒数截断。
            if config.get('platform','highway')!='carla' and scene.time-first_time>=duration-1e-8:
                summary.update(simulated_seconds=scene.time,termination_reason='timeout')
                break
            # 按实际规划帧推进投影；完整实体归格仅由检索源帧消费。
            if grids is not None:
                grids.update(scene,build=False)
            if first_s is None:
                first_s=scene.diagnostics.get('navigation_progress_m',scene.diagnostics['route_s'])
                writer.save('map.json',dict(lanes=scene.lanes,route=scene.route,route_lane_ids=scene.route_lane_ids,drivable=scene.drivable,
                    bev_drivable=scene.assets.get('bev_drivable'),physical_lane_centers=scene.assets.get('physical_lane_centers'),
                    ))
            scene.assets['global_map_json']=str((writer.path/'map.json').resolve())
            scene.assets['configuration_json']=str((writer.path/'configuration.json').resolve())
            if carla_source_clock is not None:
                source_clock=carla_source_clock.consume(scene.diagnostics.get('runtime'))
                scene.diagnostics['source_clock']=source_clock
                source_due=source_clock['due']
                summary['source_periods_skipped']+=source_clock['missed']
                if source_due or source_clock['missed']:
                    writer.append('source_clock',dict(frame_id=scene.frame_id,time=scene.time,**source_clock))
            else:
                source_due=scene.time>=next_source_time-1e-7
                if source_due:
                    missed=max(0,int(math.floor((scene.time-next_source_time)/source_period+1e-7)))
                    summary['source_periods_skipped']+=missed
                    next_source_time+=(missed+1)*source_period
            source_frame=representations_enabled and source_due
            raw,answer_errors=None,[]
            reps=None
            pending_task=None
            if source_frame:
                # 原帧的固定参考同时绑定模型输入、答案、反思和记忆，不能中途重解释。
                scene.diagnostics['goal_reference']=build_goal_reference(
                    scene,config,request_version=summary['fast_requests']+1)
                stage_started=time.perf_counter()
                capture_dir=writer.path/'captures'/f'{step:06d}'
                reps=build_representations(scene,capture_dir,extractor,grid_builder=grids)
                timing['representations_s']=time.perf_counter()-stage_started
                timing_samples['representations'].append(timing['representations_s'])
                scene.assets['scene_json']=writer.save(f'captures/{step:06d}/scene.json',capture_payload(scene))
                if retrieve_enabled:
                    stage_started=time.perf_counter()
                    active_retrieval=retrieve(reps,records,snapshot_id=store.snapshot_id,config=config)
                    timing['retrieval_s']=time.perf_counter()-stage_started
                    timing_samples['retrieval'].append(timing['retrieval_s'])
                else:
                    active_retrieval=RetrievalResult(scene.frame_id,snapshot_id=store.snapshot_id,time=scene.time,
                        diagnostics={'reason':'检索开关关闭','candidates':[],'all_records':[]})
                summary['retrieval_hits']+=len(active_retrieval.records)
                writer.append('retrieval',dict(frame_id=scene.frame_id,time=scene.time,snapshot_id=store.snapshot_id,scores=active_retrieval.scores,
                    records=[{'id':r['id'],'revision':r.get('revision',1)} for r in active_retrieval.records],
                    enabled=retrieve_enabled,diagnostics=active_retrieval.diagnostics,
                    wall_seconds=timing['retrieval_s']))
                if not asynchronous:
                    active_answer=None
                if fast.enabled:
                    prompt=build_prompt(scene,reps,active_retrieval)
                    images=model_image_paths(scene,reps)
                    writer.append('model_inputs',dict(frame_id=scene.frame_id,time=scene.time,prompt=prompt,images=images,
                        answer_protocol_version=ANSWER_PROTOCOL_VERSION,
                        goal_reference_id=scene.diagnostics['goal_reference']['reference_id']))
                    summary['fast_requests']+=1
                    if asynchronous:
                        pending_task=AsyncInferenceTask(scene,reps,prompt,images,active_retrieval,
                            request_version=summary['fast_requests'],reference=scene.diagnostics['goal_reference'])
                    else:
                        summary['fast_calls']+=1
                    if not asynchronous:
                        call_start=time.monotonic()
                        try:
                            raw=fast.call(prompt,images)
                            try:
                                active_answer=parse_answer(raw,scene,max_characters=config['models']['answer_max_characters'],
                                    reference=scene.diagnostics['goal_reference'],config=config)
                                active_answer_retrieval=active_retrieval
                            except ValueError as exc:
                                planner.clear_semantic_guidance()
                                answer_errors.append(str(exc))
                                summary['invalid_answers']+=1
                                logger.warning('仿真 %.1fs：快模型回答未通过验收：%s；本帧使用基础MPC输入',scene.time,str(exc))
                            writer.append('model_outputs',dict(frame_id=scene.frame_id,time=scene.time,raw=raw,answer=active_answer,errors=answer_errors,wall_seconds=time.monotonic()-call_start))
                        except CommunicationFailure as exc:
                            planner.clear_semantic_guidance()
                            summary['communication_failures']+=1
                            writer.append('model_outputs',dict(frame_id=scene.frame_id,time=scene.time,error=str(exc),wall_seconds=time.monotonic()-call_start))
                        timing['model_s']=time.monotonic()-call_start
            if asynchronous:
                consume_async_results()
            stage_started=time.perf_counter()
            planning_input,predictions=prepare_planning_context(scene,config,active_answer,predictor=predictor)
            timing['prediction_s']=time.perf_counter()-stage_started
            timing['prediction_preparation_s']=getattr(predictor,'preparation_seconds',0.)
            writer.append('predictions',predictions)
            if pending_task is not None:
                source_contexts[pending_task.request_version]=(copy.deepcopy(predictions),planner.export_reference_state())
                if not async_fast.submit(pending_task):
                    source_contexts.pop(pending_task.request_version,None)
                    summary['async_submission_rejected']+=1
                    writer.append('async_fast_admission',dict(frame_id=scene.frame_id,time=scene.time,status='后台快模型未接纳请求'))
            stage_started=time.perf_counter()
            plan=planner.solve(planning_input,predictions,active_answer)
            timing['mpc_s']=time.perf_counter()-stage_started
            # 最终执行审计先于评分、记录和显示；不会将未通过的正常轨迹
            # 写成将执行的轨迹，也不会用旧控制掩盖新目标的失败。
            stage_started=time.perf_counter()
            audit=execution_audit(planning_input,plan,predictions,config)
            if audit.get('alarm') and not plan.diagnostics.get('emergency',False):
                rejected=plan
                plan=planner.emergency_brake(planning_input,predictions,reason='正常候选执行前几何审计报警')
                writer.append('safety_replanning',dict(frame_id=scene.frame_id,time=scene.time,
                    rejected_plan=rejected,initial_audit=audit,replacement=plan))
                audit=execution_audit(planning_input,plan,predictions,config)
            timing['execution_audit_s']=time.perf_counter()-stage_started
            timing_samples['execution_audit'].append(timing['execution_audit_s'])
            summary['safe_braking_steps']+=int(plan.diagnostics.get('execution_mode')=='safe_brake')
            summary['emergency_steps']+=int(plan.diagnostics.get('emergency',False))
            summary['execution_alarms']+=int(audit.get('alarm',False))
            writer.append('execution_audit',dict(frame_id=scene.frame_id,time=scene.time,**audit))
            if newly_received_goal is not None:
                summary['async_goal_passed_on_arrival']+=int(any(
                    event.get('goal_id')==newly_received_goal and event.get('status')=='passed'
                    for event in plan.diagnostics.get('goal_events',[])))
                newly_received_goal=None
            summary['soft_safety_violation_steps']+=int(plan.diagnostics.get('safety_feasible') is False)
            summary['initialization_candidate_steps']+=int(plan.diagnostics.get('candidate_source')=='validated_current_initialization')
            timing['mpc_cold_start']=bool(plan.diagnostics.get('cold_start',False))
            timing_samples['mpc_cold' if timing['mpc_cold_start'] else 'mpc_warm'].append(timing['mpc_s'])
            timing_samples['prediction'].append(timing['prediction_s'])
            mpc_overruns+=int(timing['mpc_s']>planning_period)
            optimizer_budget_overruns+=int(plan.diagnostics.get('deadline_missed',False))
            stage_started=time.perf_counter()
            adapter.set_planned_trajectory(plan.states if plan.valid else None)
            timing['world_plan_s']=time.perf_counter()-stage_started
            timing_samples['world_plan'].append(timing['world_plan_s'])
            goal_overlay_path=None
            if source_frame and async_fast is None:
                stage_started=time.perf_counter()
                try:
                    overlay_path=capture_dir/'goal_overlay.png'
                    render_goal_overlay(scene,plan).save(overlay_path)
                except (OSError,ValueError) as exc:
                    # 目标诊断图不是控制验收条件；可诊断的绘制或文件错误不取消有效规划。
                    logger.warning('仿真 %.1fs：目标诊断图生成失败：%s',scene.time,str(exc))
                    writer.append('rendering',dict(frame_id=scene.frame_id,time=scene.time,
                        status='目标诊断图生成失败，规划照常验收执行',error=f'{type(exc).__name__}: {exc}'))
                else:
                    goal_overlay_path=overlay_path
                    scene.assets['goal_overlay_png']=str(goal_overlay_path.resolve())
                timing['goal_overlay_s']=time.perf_counter()-stage_started
                timing_samples['goal_overlay'].append(timing['goal_overlay_s'])
            writer.append('plans',dict(frame_id=scene.frame_id,time=scene.time,answer_source=active_answer.frame_id if active_answer else None,
                plan=plan,goal_overlay_path=str(goal_overlay_path.resolve()) if goal_overlay_path is not None else None))
            if 'goal_state' in plan.diagnostics or 'goal_events' in plan.diagnostics:
                writer.append('goals',dict(frame_id=scene.frame_id,time=scene.time,
                    answer_protocol_version=ANSWER_PROTOCOL_VERSION,
                    goal_state=plan.diagnostics.get('goal_state'),goal_events=plan.diagnostics.get('goal_events',[])))
            # 仅快模型原帧评估反思；参考规划实例不输出任何执行控制。
            if source_frame and async_fast is None:
                stage_started=time.perf_counter()
                base_scene,base_predictions=prepare_planning_context(scene,config,existing=predictions)
                reference=reference_planner.solve(base_scene,base_predictions,None)
                timing['reference_s']=time.perf_counter()-stage_started
                timing_samples['reference'].append(timing['reference_s'])
                stage_started=time.perf_counter()
                quality=evaluate(planning_input,plan,reference,predictions,config)
                timing['quality_s']=time.perf_counter()-stage_started
                timing_samples['quality'].append(timing['quality_s'])
                logger.info('仿真 %.1fs：轨迹可执行=%s，安全条件满足=%s，候选=%s，主求解=%.1fms（%s），预测=%.1fms，表示=%.1fms，参照=%.1fms，评分=%.1fms，质量=%s，预碰撞=%s，完整场景对象=%d，实际规划对象=%d',
                    scene.time,plan.valid,plan.diagnostics.get('safety_feasible'),plan.diagnostics.get('candidate_source'),
                    timing['mpc_s']*1000,'首次' if timing['mpc_cold_start'] else '后续',
                    timing['prediction_s']*1000,timing['representations_s']*1000,timing['reference_s']*1000,
                    timing['quality_s']*1000,quality.score,quality.precollision,len(scene.actors),len(planning_input.actors))
                initial_failures=plan.diagnostics.get('optimizer_skipped_initial_failures',[])
                if initial_failures:
                    logger.warning('初态不合格项=%s；跳过无法改写初态的QP迭代，制动仍独立验收；速度证据=%s',
                        initial_failures,plan.diagnostics.get('speed_limit_evidence'))
                summary['quality_total']+=1
                summary['quality_valid']+=int(quality.valid)
                summary['safety_alarms']+=int(quality.precollision or quality.collision)
                writer.append('quality',dict(time=scene.time,quality=quality,reference_plan=reference))
                # 保存实际使用的预测和参照，不能仅凭低频历史重新估计 EKF 后冒充原帧回放。
                scene.assets['planning_evidence_json']=writer.save(f'captures/{step:06d}/planning_evidence.json',
                    dict(frame_id=scene.frame_id,time=scene.time,version=scene.version,
                         predictions=predictions,plan=plan,reference=reference,quality=quality,
                         selected_answer=active_answer,raw_fast_answer=raw,retrieval=active_retrieval.scores,
                         answer_protocol_version=ANSWER_PROTOCOL_VERSION,goal_reference=scene.diagnostics['goal_reference']))
                triggers=reflection_reasons(quality,bool(answer_errors),config['quality']['minimum_score'])
                if raw is not None and triggers and worker is not None:
                    task=ReflectionTask(copy.deepcopy(scene),reps,copy.deepcopy(predictions),active_answer,raw,triggers+answer_errors,
                                        plan,reference,quality,active_retrieval)
                    worker.submit(task)
                elif write_enabled and memory_mode=='collect' and not triggers and admissible(active_answer,plan,quality,config['quality']['minimum_score']):
                    try:
                        record=make_record(scene,reps,active_answer,quality,scene.assets['scene_json'])
                        store.write(record,reps)
                        summary['written_fast']+=1
                        writer.append('memory_admission',dict(frame_id=scene.frame_id,status='快答案验收通过',record_id=record.id))
                    except (ValueError,OSError) as exc:
                        writer.append('memory_admission',dict(frame_id=scene.frame_id,status='拒收',reason=str(exc)))
            timing['decision_pipeline_s']=time.perf_counter()-cycle_started
            timing_samples['decision_pipeline'].append(timing['decision_pipeline_s'])
            def record_cycle_timing(executed):
                # 终点在本条timing记录写入前，避免把写入自身递归计入；其他本周期落盘均包含。
                timing['control_cycle_s']=time.perf_counter()-cycle_started
                timing_samples['control_cycle'].append(timing['control_cycle_s'])
                writer.append('timing',dict(frame_id=scene.frame_id,time=scene.time,**timing,
                    planner_backend=planner_backend,planning_s=timing.get('mpc_s'),
                    execution_mode=execution_mode,runtime_mode=runtime_mode,source_frame=bool(source_frame),executed=executed,
                    planning_period_s=planning_period,control_period_s=float(config.get('mpc',{}).get('step_s',.1)),
                    planned_execution_controls=execution_count,source_period_s=source_period,
                    simulation_cycle_s=timing_samples['simulation_cycle'][-1] if timing_samples['simulation_cycle'] else None,
                    wall_cycle_interval_s=timing_samples['wall_cycle_interval'][-1] if timing_samples['wall_cycle_interval'] else None,
                    definition='snapshot_s包含观测等待与状态构建，observation_wait_s为刻意等待下一真实源帧，snapshot_build_s为实际状态构建；decision_pipeline_s不含执行审计/物理步进/显示录像；control_cycle_s包含这些阶段及此前落盘，不含本条timing写入、轮次初始化与收尾'))
            if not plan.valid:
                summary['planning_failures']+=1
                summary['termination_reason']='planning_failure'
                last_constraints=plan.diagnostics.get('last_optimizer_constraints') or {}
                logger.error('仿真 %.1fs：规划器未生成物理有效候选，状态=%s，超预算=%s，末候选约束余量=%s',
                    scene.time,plan.diagnostics.get('solver_status'),plan.diagnostics.get('deadline_missed'),
                    last_constraints.get('minimum_by_group'))
                writer.save('failure_scene.json',scene)
                record_cycle_timing(False)
                break
            used.append(dict(time=scene.time,frame_id=scene.frame_id,answer_source=active_answer.frame_id if active_answer else None,
                             retrieved=[r['id'] for r in active_answer_retrieval.records] if active_answer is not None else []))
            # 仅正常安全解/安全制动可正常执行；无安全解按已确认策略单列紧急制动。
            stage_started=time.perf_counter()
            result=adapter.step_plan(plan) if hasattr(adapter,'step_plan') else adapter.step(plan.controls[0])
            timing['physics_step_s']=time.perf_counter()-stage_started
            timing_samples['physics_step'].append(timing['physics_step_s'])
            record_bridge_observation(result,'step_plan',scene.frame_id)
            used=[u for u in used if result['time']-u['time']<=config.get('memory',{}).get('post_accident_association_s',3)+1e-8]
            summary.update(simulated_seconds=result['time'],steps=step+1,collision=result['collision'],arrival=result['arrival'],
                           progress_m=result['progress']-first_s,termination_reason=result['reason'])
            summary['offroad_steps']+=int(result['offroad'])
            applied_control,actual_control=_actual_control_evidence(result)
            if config.get('platform','highway')!='carla' and not hasattr(adapter,'step_plan'):
                # highway旧同步接口明确执行当前唯一控制，其返回值不附桥证据。
                applied_control=plan.controls[0]
                actual_control={'control':applied_control,'source':'synchronous_step_argument'}
            if applied_control is not None:
                previous_control=np.asarray(applied_control,float).tolist()
            writer.append('frames',dict(frame_id=scene.frame_id,time=scene.time,ego=scene.ego,actors=scene.actors,
                planned_first_control=plan.controls[0],applied_control=applied_control,actual_control=actual_control,
                applied_control_semantics='最后一次真实执行控制；完整前缀非恒定控制，无执行证据时为空',
                planned_control_prefix=plan.controls[:execution_count],actual=result,
                answer_source=active_answer.frame_id if active_answer else None))
            stage_started=time.perf_counter()
            if video_recorder is not None:
                video_recorder._capture_frame()
            elif live:
                adapter.render()
            timing['video_display_s']=time.perf_counter()-stage_started
            if video_recorder is not None or live:
                timing_samples['video_display'].append(timing['video_display_s'])
            if pure_base and source_due:
                logger.info('仿真 %.1fs：纯基础规划，轨迹可执行=%s，规划=%.1fms，预测=%.1fms，场景构建=%.1fms，等待观测=%.1fms，步进=%.1fms，显示录像=%.1fms',
                    scene.time,plan.valid,timing['mpc_s']*1000,timing['prediction_s']*1000,
                    timing['snapshot_build_s']*1000,timing['observation_wait_s']*1000,
                    timing['physics_step_s']*1000,timing['video_display_s']*1000)
            if result['collision']:
                writer.save('accident_review.json',dict(status='待人工复查，不自动停用经验',actual=result,associated=used))
            record_cycle_timing(True)
            step+=1
            if result['terminated']:
                break
    except Exception as exc:
        summary.update(termination_reason='runtime_error',error=f'{type(exc).__name__}: {exc}')
        if adapter is not None and hasattr(adapter,'runtime_failure_report'):
            summary['runtime_failure']=adapter.runtime_failure_report()
        logger.exception('本轮运行失败，保留现场')
    finally:
        cleanup_errors=[]
        def close_resource(name,callback):
            try:
                callback()
            except Exception as exc:
                cleanup_errors.append({'resource':name,'error':type(exc).__name__})
                logger.error('收尾失败：%s（%s）；继续释放其他资源',name,type(exc).__name__)
        if adapter is not None and hasattr(adapter,'pause_runtime'):
            close_resource('停止本轮世界推进',adapter.pause_runtime)
        if async_fast is not None:
            close_resource('后台快模型',async_fast.close)
            close_resource('后台快模型最终结果接纳',lambda:consume_async_results(drain_only=True))
            summary['async_fast']=async_fast.counts
        if admission_worker is not None:
            close_resource('异步原帧验收',admission_worker.close)
            summary['async_admission']=admission_worker.counts
            for key in ('quality_total','quality_valid','safety_alarms','written_fast'):
                summary[key]+=admission_worker.counts[key]
            for key,values in admission_worker.timings.items():
                timing_samples['async_source_'+key]=values
        if worker is not None:
            close_resource('反思队列',worker.close)
            summary['reflection']=worker.counts
        if video_recorder is not None:
            close_resource('录像',video_recorder.close)
            close_resource('录像统计',lambda: summary.update(video=video_recorder.metadata()))
        if adapter is not None:
            close_resource('仿真适配器',adapter.close)
            if hasattr(adapter,'evaluation_report'):
                def record_physical_evaluation():
                    measured=adapter.evaluation_report()
                    if measured is not None:
                        summary['evaluation']=measured
                        summary['driving_outcome']=measured.get('driving_outcome')
                        summary['route_completed']=bool(measured.get('route_completed'))
                        summary['strict_success']=bool(measured.get('strict_success'))
                        summary['strict_success_status']=measured.get('strict_success_status')
                        summary['measurement_complete']=bool(measured.get('measurement_complete'))
                        summary['infrastructure_valid']=bool(measured.get('infrastructure_valid'))
                        summary['collision']=summary['collision'] or bool(measured.get('collision'))
                        summary['arrival']=bool(measured.get('arrival'))
                close_resource('独立物理评价',record_physical_evaluation)
            if hasattr(adapter,'runtime_failure_report'):
                # SDK在EOF/finally完成销毁核验，必须在等待其退出之后读取。
                def record_cleanup_evidence():
                    report=adapter.runtime_failure_report()
                    if 'cleanup' in report:
                        summary['world_cleanup']=report['cleanup']
                    if cleanup_errors or summary['termination_reason']=='runtime_error':
                        summary['runtime_failure']=report
                close_resource('轮间清理记录',record_cleanup_evidence)
        if cleanup_errors:
            summary['cleanup_errors']=cleanup_errors
            summary['termination_reason_before_cleanup']=summary['termination_reason']
            summary['termination_reason']='runtime_error'
        summary['wall_seconds']=time.monotonic()-start_wall
        summary['timing']={name:_timing_summary(values) for name,values in timing_samples.items()}
        summary['timing']['planning_cold']=summary['timing']['mpc_cold']
        summary['timing']['planning_warm']=summary['timing']['mpc_warm']
        summary['timing']['planner_backend']=planner_backend
        summary['timing']['legacy_mpc_keys_note']='mpc_*保留旧统计读取兼容；实际后端见planner_backend，不能把PDM耗时称MPC求解耗时'
        summary['timing']['mpc_deadline_misses']=mpc_overruns
        summary['timing']['mpc_deadline_s']=planning_period
        summary['timing']['mpc_optimizer_budget_misses']=optimizer_budget_overruns
        summary['timing']['control_cycle_deadline_s']=planning_period
        summary['timing']['control_cycle_deadline_misses']=sum(value>planning_period for value in timing_samples['control_cycle'])
        summary['timing']['control_cycle_deadline_miss_fraction']=(
            sum(value>planning_period for value in timing_samples['control_cycle'])/len(timing_samples['control_cycle'])
            if timing_samples['control_cycle'] else None)
        summary['timing']['legacy_100ms_note']='以下100ms统计仅供旧实验对比，当前重规划截止时间见control_cycle_deadline_s'
        summary['timing']['control_cycle_over_100ms']=sum(value>.1 for value in timing_samples['control_cycle'])
        summary['timing']['control_cycle_over_100ms_fraction']=(
            sum(value>.1 for value in timing_samples['control_cycle'])/len(timing_samples['control_cycle'])
            if timing_samples['control_cycle'] else None)
        try:
            writer.save('summary.json',summary)
        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
    logger.info('本轮结束：%s',json.dumps(summary,ensure_ascii=False))
    return summary
