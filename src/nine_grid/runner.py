"""10Hz 物理规划与 1Hz 检索调用的同步实验入口。"""
from __future__ import annotations
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import numpy as np
from shapely.geometry import box, Polygon
from .contracts import ANSWER_PROTOCOL_VERSION, ReflectionTask, RetrievalResult, plain
from .goals import build_goal_reference
from .highway import HighwayAdapter
from .memory import MemoryStore
from .models import VisionClient, CommunicationFailure, parse_answer, build_prompt, local_line
from .planner import Planner
from .prediction import Predictor
from .quality import evaluate, admissible, reflection_reasons, execution_audit
from .representations import build_representations, DINOExtractor, NineGridBuilder, render_goal_overlay
from .retrieval import retrieve
from .runtime import EpisodeWriter, ReflectionWorker, make_record

logger=logging.getLogger(__name__)


def _timing_summary(values):
    if not values:return {'count':0}
    return dict(count=len(values),p50_ms=float(np.percentile(values,50))*1000,
                p95_ms=float(np.percentile(values,95))*1000,max_ms=float(max(values))*1000)


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
        global_map_asset=scene.assets.get('global_map_json'),map_crop='局部显示为自车中心前后左右各150米；完整地图另存为不可变资产')


def run_episode(config,model_settings,episode_id,episodes_dir,memory_read_dir=None,memory_write_dir=None,
                retrieve_enabled=False,write_enabled=False,memory_mode='collect',seed=1,live=False,video=False,
                fast_client=None,reflection_client=None,extractor=None):
    fast=fast_client if fast_client is not None else VisionClient('fast',model_settings['fast'])
    reflection=reflection_client if reflection_client is not None else VisionClient('reflection',model_settings['reflection'])
    # 本地模型导入/配置先于图像解码，且绝不以随机特征替代真实权重。
    if (retrieve_enabled or write_enabled) and extractor is None:
        extractor=DINOExtractor(config.get('dino',{}).get('model','facebook/dinov2-base'))
    store=MemoryStore(memory_read_dir if retrieve_enabled else None,memory_write_dir if write_enabled else None)
    records=store.snapshot()
    writer=EpisodeWriter(episodes_dir,episode_id)
    file_handler=logging.FileHandler(writer.path/'run.log',encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logging.getLogger().addHandler(file_handler)
    writer.save('configuration.json',dict(parameters=config,
        models={role:getattr(client,'settings',model_settings[role])
                for role,client in (('fast',fast),('reflection',reflection))},seed=seed,
        effective_parameters_sha256=hashlib.sha256(json.dumps(plain(config),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest(),
        numerical_threads={name:os.environ.get(name) for name in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        answer_protocol_version=ANSWER_PROTOCOL_VERSION,
        retrieval_enabled=retrieve_enabled,write_enabled=write_enabled,memory_mode=memory_mode,
        memory_read_dir=str(memory_read_dir),memory_write_dir=str(memory_write_dir),snapshot_id=store.snapshot_id,
        snapshot_records=[{'id':r['id'],'revision':r.get('revision',1)} for r in records],excluded_records=store.audit))
    adapter=None
    worker=None
    video_recorder=None
    summary=dict(episode_id=episode_id,seed=seed,scenario=config.get('scenario','highway'),
        simulated_seconds=0.,steps=0,collision=False,arrival=False,planning_failures=0,
        safety_alarms=0,execution_alarms=0,quality_valid=0,quality_total=0,fast_calls=0,communication_failures=0,
        invalid_answers=0,retrieval_hits=0,written_fast=0,offroad_steps=0,
        fallback_controls=0,termination_reason='initialization_failure',progress_m=0.)
    start_wall=time.monotonic()
    logger.info('开始 %s，场景=%s，seed=%s，检索=%s，写入=%s',episode_id,config.get('scenario'),seed,retrieve_enabled,write_enabled)
    used=[]
    timing_samples={'mpc_cold':[],'mpc_warm':[],'prediction':[],'decision_pipeline':[]}
    mpc_overruns=0
    try:
        adapter=HighwayAdapter(config,seed,episode_id,live)
        if video:video_recorder=adapter.start_recording(writer.path/'video.mp4')
        planner,reference_planner=Planner(config),Planner(config)
        predictor=Predictor(config)
        grids=NineGridBuilder(config)
        worker=ReflectionWorker(reflection,lambda:Planner(config),config,store,writer,write_enabled)
        active_answer=None
        active_retrieval=RetrievalResult('',snapshot_id=store.snapshot_id)
        first_s=None
        fast_every=round(config['clock']['planner_hz']/config['clock']['fast_hz'])
        limit=round(config.get('duration',config['clock']['episode_seconds'])*config['clock']['planner_hz'])
        for step in range(limit):
            cycle_started=time.perf_counter()
            timing=dict(representations_s=0.,model_s=0.,reference_s=0.,quality_s=0.)
            scene=adapter.snapshot()
            # 10Hz推进参考迟滞与投影；完整实体归格仅由1Hz表示阶段消费。
            grids.update(scene,build=False)
            if first_s is None:
                first_s=scene.diagnostics['route_s']
                writer.save('map.json',dict(lanes=scene.lanes,route=scene.route,route_lane_ids=scene.route_lane_ids,drivable=scene.drivable,
                    bev_drivable=scene.assets.get('bev_drivable'),physical_lane_centers=scene.assets.get('physical_lane_centers'),
                    ))
            scene.assets['global_map_json']=str((writer.path/'map.json').resolve())
            scene.assets['configuration_json']=str((writer.path/'configuration.json').resolve())
            source_frame=step%fast_every==0
            raw,answer_errors=None,[]
            reps=None
            if source_frame:
                # 原帧的固定参考同时绑定模型输入、答案、反思和记忆，不能中途重解释。
                scene.diagnostics['goal_reference']=build_goal_reference(
                    scene,config,request_version=summary['fast_calls']+1)
                stage_started=time.perf_counter()
                capture_dir=writer.path/'captures'/f'{step:06d}'
                reps=build_representations(scene,capture_dir,extractor,grid_builder=grids)
                timing['representations_s']=time.perf_counter()-stage_started
                scene.assets['scene_json']=writer.save(f'captures/{step:06d}/scene.json',capture_payload(scene))
                if retrieve_enabled:
                    active_retrieval=retrieve(reps,records,top_k=config['retrieval']['final_k'],min_score=config['retrieval']['minimum_score'],snapshot_id=store.snapshot_id)
                else:
                    active_retrieval=RetrievalResult(scene.frame_id,snapshot_id=store.snapshot_id,time=scene.time)
                summary['retrieval_hits']+=len(active_retrieval.records)
                writer.append('retrieval',dict(frame_id=scene.frame_id,time=scene.time,snapshot_id=store.snapshot_id,scores=active_retrieval.scores,
                    records=[{'id':r['id'],'revision':r.get('revision',1)} for r in active_retrieval.records]))
                active_answer=None
                if fast.enabled:
                    prompt=build_prompt(scene,reps,active_retrieval)
                    writer.append('model_inputs',dict(frame_id=scene.frame_id,time=scene.time,prompt=prompt,images=[reps.bev_path],
                        answer_protocol_version=ANSWER_PROTOCOL_VERSION,
                        goal_reference_id=scene.diagnostics['goal_reference']['reference_id']))
                    summary['fast_calls']+=1
                    call_start=time.monotonic()
                    try:
                        raw=fast.call(prompt,[reps.bev_path])
                        try:
                            active_answer=parse_answer(raw,scene,max_characters=config['models']['answer_max_characters'],
                                reference=scene.diagnostics['goal_reference'],config=config)
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
            stage_started=time.perf_counter()
            predictions=predictor.predict(scene)
            timing['prediction_s']=time.perf_counter()-stage_started
            timing['prediction_preparation_s']=getattr(predictor,'preparation_seconds',0.)
            writer.append('predictions',predictions)
            stage_started=time.perf_counter()
            plan=planner.solve(scene,predictions,active_answer)
            timing['mpc_s']=time.perf_counter()-stage_started
            timing['mpc_cold_start']=bool(plan.diagnostics.get('cold_start',False))
            timing_samples['mpc_cold' if timing['mpc_cold_start'] else 'mpc_warm'].append(timing['mpc_s'])
            timing_samples['prediction'].append(timing['prediction_s'])
            mpc_overruns+=int(plan.diagnostics.get('deadline_missed',False))
            goal_overlay_path=None
            if source_frame:
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
            writer.append('plans',dict(frame_id=scene.frame_id,time=scene.time,answer_source=active_answer.frame_id if active_answer else None,
                plan=plan,goal_overlay_path=str(goal_overlay_path.resolve()) if goal_overlay_path is not None else None))
            if 'goal_state' in plan.diagnostics or 'goal_events' in plan.diagnostics:
                writer.append('goals',dict(frame_id=scene.frame_id,time=scene.time,
                    answer_protocol_version=ANSWER_PROTOCOL_VERSION,
                    goal_state=plan.diagnostics.get('goal_state'),goal_events=plan.diagnostics.get('goal_events',[])))
            # 仅快模型原帧评估反思；参考规划实例不输出任何执行控制。
            if source_frame:
                stage_started=time.perf_counter()
                reference=reference_planner.solve(scene,predictions,None)
                timing['reference_s']=time.perf_counter()-stage_started
                stage_started=time.perf_counter()
                quality=evaluate(scene,plan,reference,predictions,config)
                timing['quality_s']=time.perf_counter()-stage_started
                logger.info('仿真 %.1fs：MPC=%s，主求解=%.1fms（%s），预测=%.1fms，表示=%.1fms，参照=%.1fms，评分=%.1fms，质量=%s，预碰撞=%s，可见对象=%d',
                    scene.time,plan.valid,timing['mpc_s']*1000,'首次' if timing['mpc_cold_start'] else '后续',
                    timing['prediction_s']*1000,timing['representations_s']*1000,timing['reference_s']*1000,
                    timing['quality_s']*1000,quality.score,quality.precollision,len(scene.actors))
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
                if raw is not None and triggers:
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
            writer.append('timing',dict(frame_id=scene.frame_id,time=scene.time,**timing,
                definition='决策管线含同步表示、模型等待、预测、主规划、参照评分和对应落盘；不含随后执行审计、物理步进、显示和录像'))
            if not plan.valid:
                summary['planning_failures']+=1
                summary['termination_reason']='planning_failure'
                last_constraints=plan.diagnostics.get('last_optimizer_constraints') or {}
                logger.error('仿真 %.1fs：MPC未生成有效候选，状态=%s，超预算=%s，末候选约束余量=%s',
                    scene.time,plan.diagnostics.get('solver_status'),plan.diagnostics.get('deadline_missed'),
                    last_constraints.get('minimum_by_group'))
                writer.save('failure_scene.json',scene)
                break
            used.append(dict(time=scene.time,frame_id=scene.frame_id,answer_source=active_answer.frame_id if active_answer else None,
                             retrieved=[r['id'] for r in active_retrieval.records] if active_answer is not None else []))
            # 有解时即使独立风险报警也执行当前控制；不存在旧控制/备用控制器分支。
            audit=execution_audit(scene,plan,predictions,config)
            summary['execution_alarms']+=int(audit['alarm'])
            writer.append('execution_audit',dict(frame_id=scene.frame_id,time=scene.time,**audit))
            result=adapter.step(plan.controls[0])
            used=[u for u in used if result['time']-u['time']<=config.get('memory',{}).get('post_accident_association_s',3)+1e-8]
            summary.update(simulated_seconds=result['time'],steps=step+1,collision=result['collision'],arrival=result['arrival'],
                           progress_m=result['progress']-first_s,termination_reason=result['reason'])
            summary['offroad_steps']+=int(result['offroad'])
            writer.append('frames',dict(frame_id=scene.frame_id,time=scene.time,ego=scene.ego,actors=scene.actors,
                applied_control=plan.controls[0],actual=result,answer_source=active_answer.frame_id if active_answer else None))
            if video_recorder is not None:
                video_recorder._capture_frame()
            elif live:
                adapter.render()
            if result['collision']:
                writer.save('accident_review.json',dict(status='待人工复查，不自动停用经验',actual=result,associated=used))
            if result['terminated']:
                break
        else:
            summary['termination_reason']='timeout'
    except Exception as exc:
        summary.update(termination_reason='runtime_error',error=f'{type(exc).__name__}: {exc}')
        logger.exception('本轮运行失败，保留现场')
    finally:
        if worker is not None:
            worker.close()
            summary['reflection']=worker.counts
        if video_recorder is not None:
            video_recorder.close()
            summary['video']=video_recorder.metadata()
        if adapter is not None:
            adapter.close()
        summary['wall_seconds']=time.monotonic()-start_wall
        summary['timing']={name:_timing_summary(values) for name,values in timing_samples.items()}
        summary['timing']['mpc_deadline_misses']=mpc_overruns
        writer.save('summary.json',summary)
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
    logger.info('本轮结束：%s',json.dumps(summary,ensure_ascii=False))
    return summary
