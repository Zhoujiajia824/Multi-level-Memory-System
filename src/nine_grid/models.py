"""两个独立视觉模型的协议、显式重试与校验。"""
from __future__ import annotations
import base64
import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path
import numpy as np
import yaml
from shapely.geometry import LineString,box
from .contracts import ModelAnswer, plain
from .decision_protocol import normalize_model_output
from .object_scope import scope_scene
from .perception_scope import perception_scene

logger=logging.getLogger(__name__)

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2]/'config/prompts.yaml'


class _PromptLoader(yaml.SafeLoader):
    """提示词重复字段直接报错，避免修改了被后一个同名字段覆盖的内容。"""


def _unique_prompt_mapping(loader,node):
    mapping={}
    for key_node,value_node in node.value:
        key=loader.construct_object(key_node)
        if not isinstance(key,str) or key in mapping:
            raise ValueError('提示词YAML字段必须为不重复的字符串')
        mapping[key]=loader.construct_object(value_node)
    return mapping


_PromptLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,_unique_prompt_mapping)


def load_system_prompts(path=DEFAULT_PROMPTS_PATH):
    """从UTF-8 YAML加载两份独立系统提示词，保留原文及换行，不做格式插值。"""
    path=Path(path).resolve()
    try:
        data=yaml.load(path.read_text(encoding='utf-8-sig'),Loader=_PromptLoader)
    except (OSError,UnicodeError,yaml.YAMLError) as exc:
        raise ValueError(f'无法读取提示词YAML：{path}（{type(exc).__name__}）') from None
    if not isinstance(data,dict) or set(data)!={'fast','reflection'}:
        raise ValueError('提示词YAML必须且仅包含 fast、reflection')
    for role in ('fast','reflection'):
        entry=data[role]
        if not isinstance(entry,dict) or set(entry)!={'system'}:
            raise ValueError(f'提示词 {role} 必须且仅包含 system')
        if not isinstance(entry['system'],str) or not entry['system'].strip():
            raise ValueError(f'提示词 {role}.system 必须为非空文本')
    return {role:data[role]['system'] for role in ('fast','reflection')}


class CommunicationFailure(RuntimeError):
    pass


class VisionClient:
    def __init__(self,role,settings,transport=None,sleeper=time.sleep):
        if role not in ('fast','reflection'):
            raise ValueError('模型角色必须为 fast 或 reflection')
        self.role=role
        self.settings=dict(settings)
        self.backend=settings.get('backend','off')
        self.sleeper=sleeper
        self.client=None
        self.attempts=0
        self.system_prompt=None
        if self.backend not in ('off','api','server'):
            raise ValueError('模型 backend 必须为 off、api 或 server')
        if self.backend=='off':
            return
        system=settings.get('system_prompt')
        if system is None:
            system=load_system_prompts()[role]
            self.settings['prompts_config']=str(DEFAULT_PROMPTS_PATH)
        if not isinstance(system,str) or not system.strip():
            raise ValueError(f'{role} 系统提示词必须为非空文本')
        self.system_prompt=system
        self.settings['system_prompt']=system
        self.settings['system_prompt_sha256']=hashlib.sha256(system.encode('utf-8')).hexdigest()
        if not settings.get('model') or not settings.get('base_url'):
            raise ValueError(f'{role} 模型必须配置 model 与 base_url')
        if transport is not None:
            self.client=transport
            return
        key=os.environ.get(settings.get('api_key_env',''), '')
        if self.backend=='api' and not key:
            raise ValueError(f'{role} 模型 API 密钥环境变量未设置')
        from openai import OpenAI
        self.client=OpenAI(api_key=key or 'local-server',base_url=settings['base_url'],
                           timeout=settings.get('timeout_s',120 if role=='fast' else 300),max_retries=0)

    @property
    def enabled(self):
        return self.backend!='off'

    def call(self,prompt,image_paths):
        if not self.enabled:
            return None
        content=[{'type':'text','text':prompt}]
        for path in image_paths:
            path=Path(path)
            raw=base64.b64encode(path.read_bytes()).decode('ascii')
            mime='image/jpeg' if path.suffix.lower() in ('.jpg','.jpeg') else 'image/png'
            content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,{raw}'}})
        system=self.system_prompt
        logger.info('%s 模型输入\n系统：%s\n场景：%s\n图像：%s',self.role,system,prompt,[str(x) for x in image_paths])
        for attempt in range(6):
            self.attempts+=1
            started=time.monotonic()
            try:
                result=self.client.chat.completions.create(model=self.settings['model'],
                    messages=[{'role':'system','content':system},{'role':'user','content':content}],
                    max_tokens=self.settings.get('max_tokens',4096))
                choice=result.choices[0]
                if choice.finish_reason=='length':
                    raise CommunicationFailure('输出被截断')
                raw=choice.message.content or ''
                logger.info('%s 模型输出（第%d次，%.2f秒）\n%s',self.role,attempt+1,time.monotonic()-started,raw)
                return raw
            except Exception as exc:
                from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
                retryable=isinstance(exc,(CommunicationFailure,TimeoutError,ConnectionError,APIConnectionError,APITimeoutError,InternalServerError,RateLimitError))
                # 不输出远端异常文本：其中可能含 URL 认证信息或请求头。
                logger.warning('%s 调用失败：%s，尝试 %d/6',self.role,type(exc).__name__,attempt+1)
                if not retryable or attempt==5:
                    raise CommunicationFailure(f'{self.role} 调用失败，类型 {type(exc).__name__}，尝试 {attempt+1} 次') from None
                self.sleeper(2**attempt)
        raise AssertionError('不可达分支')


def attention_ids(scene,config=None):
    # 候选是车道范围对象与本帧BEV内动态对象的并集，不限八向代表。
    scoped, _ = perception_scene(scene,config)
    return {a.id for a in scoped.actors}


def _camera_entries(scene):
    """平台只提供与当前冻结帧对齐的图像；不在核心层猜测最近相机帧。"""
    entries=scene.assets.get('camera_images',[])
    if not isinstance(entries,list):
        raise ValueError('camera_images 必须为当前帧图像列表')
    seen=set()
    for item in entries:
        if (not isinstance(item,dict) or not {'view','path','frame_id','time'}.issubset(item)
            or not isinstance(item['view'],str) or not item['view'].strip()
            or item['view'] in seen or not isinstance(item['path'],str) or not item['path'].strip()):
            raise ValueError('相机图像视角、路径或对齐字段缺失/重复')
        stamp=item['time']
        if (item['frame_id']!=scene.frame_id or type(stamp) not in (int,float)
            or not math.isfinite(stamp) or not math.isclose(stamp,scene.time,rel_tol=0.,abs_tol=1e-6)):
            raise ValueError('相机图像不属于当前冻结帧，禁止使用错帧或过期图像')
        seen.add(item['view'])
    return entries


def model_image_paths(scene,reps):
    """快模型与原帧反思共用同一顺序：已命名相机视图在前，当前BEV在后。"""
    if reps.frame_id!=scene.frame_id or reps.time!=scene.time:
        raise ValueError('BEV表示与模型原帧身份或时间不一致')
    paths=[item['path'] for item in _camera_entries(scene)]+[reps.bev_path]
    for path in paths:
        if not isinstance(path,str) or not Path(path).is_file():
            raise ValueError('模型当前帧图像缺失；不能静默丢弃相机或BEV')
    return paths


def _unique_answer_object(pairs):
    """重复JSON键会产生多个目标解释，整份回答拒收，不采用最后一个值。"""
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError('模型回答含重复JSON字段')
        result[key]=value
    return result


def _goal_reference(scene,config=None):
    """一个原帧只建立一份目标参考，提示词、解析和反思共同使用。"""
    reference=scene.diagnostics.get('goal_reference')
    if reference is None:
        from .goals import build_goal_reference
        scoped, _ = scope_scene(scene, config)
        reference=build_goal_reference(scoped,config)
        scene.diagnostics['goal_reference']=reference
    return reference


def parse_answer(raw,scene,source='fast',max_characters=6000,reference=None,config=None):
    if not isinstance(raw,str) or len(raw)>max_characters:
        raise ValueError('回答缺失或超出完整模板长度')
    serialized=raw.strip()
    if serialized.startswith('```'):
        # 仅容忍包住完整JSON的单一外壳，不从杂文或多个代码块中猜测答案。
        wrapped=re.fullmatch(r'```(?:json)?[ \t]*\r?\n((?:(?!```).)*?)\r?\n```',serialized,re.DOTALL|re.IGNORECASE)
        if wrapped is None:
            raise ValueError('回答不是单个有效 JSON')
        serialized=wrapped.group(1)
    try:
        data=json.loads(serialized,object_pairs_hook=_unique_answer_object)
    except (ValueError,TypeError):
        raise ValueError('回答不是单个有效 JSON') from None
    attention,goal,explanation,output_protocol=normalize_model_output(data)
    scene, _ = perception_scene(scene, config)
    ids={actor.id for actor in scene.actors}
    seen=set()
    for item in attention:
        if item['id'] not in ids or item['id'] in seen:
            raise ValueError('关注 ID 不存在、不可用、自车或重复')
        seen.add(item['id'])
    from .goals import validate_local_goal
    selected=validate_local_goal(scene,_goal_reference(scene,config) if reference is None else reference,goal)
    return ModelAnswer(scene.frame_id,scene.time,attention,selected['local_goal'],explanation,
                       selected['world_goal'],selected['goal_reference'],selected['goal_id'],raw,source,
                       target_lane_id=selected['target_lane_id'],
                       output_protocol_version=output_protocol,model_output=data)


def local_line(points,origin,radius=120.):
    points=np.asarray(points)
    if len(points)<2:
        return points
    x,y=origin
    clipped=LineString(points).intersection(box(x-radius,y-radius,x+radius,y+radius))
    if clipped.is_empty:
        return np.empty((0,2))
    if hasattr(clipped,'geoms'):
        sections=[np.asarray(g.coords) for g in clipped.geoms if g.geom_type=='LineString']
        return np.vstack(sections) if sections else np.empty((0,2))
    return np.asarray(clipped.coords)


def _current_actor(actor):
    """模型只接收对象当前物理状态，内部车道、可见性和历史记录不外发。"""
    return {key:getattr(actor,key) for key in
            ('id','kind','position','velocity','acceleration','heading','length','width','radius')}


def scene_text(scene,representations):
    """模型只接收统一候选列表、必要图像说明与合法目标参考。"""
    scene, _ = perception_scene(scene)
    from .goals import goal_reference_context
    data={'自车':_current_actor(scene.ego),
        '候选对象':[_current_actor(actor) for actor in scene.actors],
        '规划参考':goal_reference_context(scene,_goal_reference(scene)),
        '观测说明':'候选对象均为本帧提供的当前状态；只从列表ID选择关注对象，观察到对象不代表其所在空间允许自车驶入。',
        '图像说明':'BEV自车朝上，x前y左，刻度为米；红为自车、蓝为车辆、橙为自行车、紫为行人、黄为道路障碍。细灰虚线是虚拟车道中心参考，不是真实标线；绿色虚线是固定目标参考，s+表示道路弧长。历史轨迹只表示已观测运动。'}
    if scene.diagnostics.get('observation_policy')=='range_limited_ground_truth_no_occlusion-1.0':
        data['观测说明']+='本帧候选真值不判断遮挡，不因RGB遮挡删除给定状态。'
    cameras=_camera_entries(scene)
    if any(item['view']=='highway_env_topdown' for item in cameras):
        data['图像说明']+=' highway_env_topdown为原生俯视图：世界x右y下，自车不固定朝上；不套用BEV坐标和配色。'
    if any(item['view']=='multi_camera_mosaic' for item in cameras):
        data['图像说明']+=' multi_camera_mosaic六相机拼图：上排左前、正前、右前，下排左后、正后、右后；不同视角可包含同一对象，透视图不按BEV刻度读距。'
    if cameras:
        data['图像说明']+='当前图像顺序：'+'、'.join(item['view'] for item in cameras)+'、BEV；均绑定本帧状态。'
    return json.dumps(plain(data),ensure_ascii=False,allow_nan=False)


def _historical_situation(situation):
    """历史经验保留情境与当前态示例，不附带帧标识、时间或坐标元数据。"""
    result={key:situation[key] for key in ('scenario','情境','自车速度','可见对象','观测来源') if key in situation}
    if '可见对象' in result:
        fields=('id','类型','世界位置','速度','尺寸')
        result['可见对象']=[{key:obj[key] for key in fields if key in obj} for obj in result['可见对象']]
    return result


def _reflection_diagnostics(value):
    """原帧身份留在任务与日志中，不通过诊断嵌套字段重新进入模型请求。"""
    value=plain(value)
    excluded={'visible','history','observed','object_scope','perception_scope','prediction_scope','planning_object_selection',
              'planning_scope','road_rule_observations','excluded','actor_classification',
              'potential_field','rule_observation_scope',
              '仿真时间秒','坐标','导航路线世界坐标','导航路线当前自车坐标','规则','地图边界',
              'historical_only_objects','只供历史回顾且不可选为当前关注对象',
              'rules','rule_evidence','desired_speed','must_wait','wait_reason','stop_s','speed_limit',
              'crosswalks','traffic_lights','map_available','road_priority_available','goal_s',
              'route','route_lane_ids','reference','observed_lane_id','stable_navigation_lane_id',
              'reference_state_before','local_reference','goal_reference',
              'goal_state','goal_state_before','goal_events'}
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if key in excluded:
                continue
            if key=='goal_evaluation':
                result[key]=({name:scalar for name,scalar in item.items()
                              if scalar is None or isinstance(scalar,(str,int,float,bool))}
                             if isinstance(item,dict) else None)
            elif key=='active_goal':
                result[key]=item if item is None or isinstance(item,(str,int,float,bool)) else None
            else:
                result[key]=_reflection_diagnostics(item)
        return result
    if isinstance(value,list):
        return [_reflection_diagnostics(item) for item in value]
    return value


def build_prompt(scene,reps,retrieval=None,reflection=None):
    prompt='当前帧\n'+scene_text(scene,reps)
    if retrieval is not None and retrieval.records:
        examples=[]
        origins={score['id']:score.get('matched_channels',[]) for score in retrieval.scores}
        channel_labels={'bev':'BEV视觉布局','occupancy':'OCC面积占据','grid':'交通交互相似'}
        for record in retrieval.records:
            if record.get('record_origin')=='offline_retrospective':
                example={'经验来源':'离线回顾经验，已人工审核；模型整理内容不是因果或最优驾驶真值',
                         '完整历史经验':record['offline_text'],
                         '使用边界':'其中后续过程是历史片段实际记录，不是当前已知事实或当前未来。不得将历史对象描述作为当前ID，也不得复制历史坐标；须依据当前场景独立生成关注列表和单个合法局部目标。'}
            else:
                example={'历史情境':_historical_situation(record.get('situation',{})),'历史参考答案':record.get('answer',{}),
                         '坐标警告':'历史local_goal仅是历史参考中的示例；须依据当前规划参考独立生成单个目标，不可复制历史坐标'}
            channels=origins.get(record.get('id'),[])
            if channels:
                example['检索来源通道']=[channel_labels[c] for c in channels if c in channel_labels]
            examples.append(example)
        prompt+='\n历史经验（不包含历史图像，不代表当前坐标）\n'+json.dumps(plain(examples),ensure_ascii=False)
    if reflection is not None:
        prompt+='\n原始快答案\n'+reflection.raw_answer
        prompt+='\n原帧诊断\n'+json.dumps(_reflection_diagnostics({'errors':reflection.errors,'plan':reflection.plan,'quality':reflection.quality}),ensure_ascii=False,allow_nan=False)
    return prompt
