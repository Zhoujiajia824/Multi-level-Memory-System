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


def attention_ids(scene):
    # 只接受实际提供给模型的当前对象ID；地图边界不再进入模型输入。
    scoped, _ = scope_scene(scene)
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
    scene, _ = scope_scene(scene, config)
    ids=attention_ids(scene)
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


_REGION_NAMES = {
    'left_front':'自车左侧相邻车道前方区域',
    'front':'自车车道前方区域',
    'right_front':'自车右侧相邻车道前方区域',
    'left_side':'自车左侧相邻车道侧方区域',
    'right_side':'自车右侧相邻车道侧方区域',
    'left_rear':'自车左侧相邻车道后方区域',
    'rear':'自车车道后方区域',
    'right_rear':'自车右侧相邻车道后方区域',
}


def _eight_direction_context(scene,grid):
    """内部格子仅提供区域归属；给模型的数值全部取自当前世界状态。"""
    from .representations import build_grid
    scene, scope = scope_scene(scene)
    # 检索八向仍描述较宽的图像上下文，不能直接将其代表对象塞入模型真值表。
    # 模型角色按同一车道拓扑、当前物理车道和本次限定弧长重新确定。
    grid = build_grid(scene, object_range=scope['settings'], include_static=True)
    actors={a.id:a for a in scene.actors if a.visible}
    available=bool(grid.get('valid',False))
    regions={}
    result={'区域描述可用':available,'对象':[],
            '未发现可见对象的区域':[],'不存在相邻同向车道的区域':[],
            '相邻车道存在但当前位置禁止跨越的区域':[],'未知区域':[]}
    for slot,label in _REGION_NAMES.items():
        info=grid.get('slots',{}).get(slot,{}) if available else {}
        state=info.get('state')
        identity=info.get('id')
        candidates=[]
        for key,row in scope['objects'].items():
            member=row['memberships'].get(slot)
            if key in actors and member is not None:
                gap=member['side_gap_m'] if slot.endswith('side') else member['longitudinal_gap_m']
                candidates.append((gap,abs(member['delta_s_m']),key))
        if candidates:
            identity=min(candidates)[2]
            state='occupied'
        elif state=='occupied':
            state='unknown'
        if state=='occupied' and identity in actors:
            regions.setdefault(identity,[]).append(label)
        elif state=='empty':
            result['未发现可见对象的区域'].append(label)
        elif (state=='not_applicable' and info.get('reason')=='no_adjacent_lane'
              and slot not in ('front','rear')):
            result['不存在相邻同向车道的区域'].append(label)
        elif (state=='not_applicable' and info.get('reason')=='noncrossable_boundary'
              and slot not in ('front','rear')):
            result['相邻车道存在但当前位置禁止跨越的区域'].append(label)
        else:
            # 缺证据或对象已不在可见集合时，不把未知格描述成没有车辆。
            result['未知区域'].append(label)
    for identity,labels in regions.items():
        result['对象'].append({**_current_actor(actors[identity]),'所在区域':labels})
    others=[_current_actor(actor) for actor in scene.actors if actor.visible and actor.id not in regions]
    return result,others


def scene_text(scene,representations):
    """只构造模型所需的当前状态；完整快照仍用于物理规划与审计。"""
    scene, scope = scope_scene(scene)
    context,others=_eight_direction_context(scene,representations.grid)
    from .goals import goal_reference_context
    data={'自车':_current_actor(scene.ego),'八向图':context,'范围内其他对象':others,
        '对象输入范围':{'version':scope['version'], **scope['settings'],
            '距离定义':'对象中心沿各允许车道连续中心线的弧长差；不是圆形距离',
            '类型':['vehicle','bicycle','pedestrian','static_obstacle'],
            '障碍物定义':'明确类别的道路阻塞物且实际进入准入车道；不含围栏、护栏、墙、杆和泛化Static/Other',
            '对象数量':len(scene.actors),
            '车道':{side:None if chain is None else chain['lane_ids'] for side,chain in scope['chains'].items()},
            '说明':'范围之外未提供当前对象真值；图像可能显示更宽背景，不能据此编造可关注ID或精确运动状态'},
        '规划参考':goal_reference_context(scene,_goal_reference(scene)),
        '图像解释':'自车固定朝上，横纵轴为米。非行驶区白色，道路为NAVSIM浅灰，人行横道为30%透明紫色；车道边界线和分割线不绘制，细深灰短虚线表示虚拟车道中心线及连接参考路径，不得将其视作交通标线。模型图的绿色虚线是本次固定Frenet参考，s+距离标签表示从原帧自车投影开始的道路弧长，不等于BEV前向x距离；这些参考标签不进入DINO特征图。颜色不是风险或关注度。自车红色、其他车辆蓝色、自行车橙色、行人紫色、静态障碍黄色；图中历史轨迹保持类别色相和高亮度，速度越高饱和度越高，固定范围0至120km/h。历史只含已观测样本。范围内未被八向图选中的对象列于范围内其他对象，同样可以被关注；图像范围和结构化对象准入范围不同。'}
    if scene.diagnostics.get('observation_policy')=='range_limited_ground_truth_no_occlusion-1.0':
        data['输入权限']='仅允许车道及前后弧长范围内的指定交通对象真值，不判断遮挡；不等同于实际传感器可见性。不得依据图像遮挡编造或删除给定状态，也不得恢复范围外对象。'
    cameras=_camera_entries(scene)
    if any(item['view']=='highway_env_topdown' for item in cameras):
        data['图像解释']+=' highway_env_topdown是原生环境俯视图：世界x向右、y向下，自车不固定朝上，配色由环境定义；不含规划线，不按BEV刻度读数。'
    if cameras:
        data['图像解释']+=' 当前输入图像依次为：'+ '、'.join(item['view'] for item in cameras)+'、当前BEV。相机图像与上述状态属于同一冻结帧；摄像头视角不能按BEV坐标直接读数。'
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
    excluded={'visible','history','observed','object_scope','planning_object_selection',
              'planning_scope','road_rule_observations','excluded','actor_classification',
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
        channel_labels={'bev':'BEV视觉布局','occupancy':'OCC面积占据','grid':'车道八向交互'}
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
