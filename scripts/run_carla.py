"""CARLA独立世界的同步实验入口；dry-run不启动SDK、模型或特征加载。"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import yaml

os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def project_path(value):
    """配置、产物及本地模型路径不随终端工作目录变化。"""
    path=Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


class UniqueSafeLoader(yaml.SafeLoader):
    """拒绝重复字段和合并覆盖，避免实验参数被最后一个同名键悄悄替换。"""


def unique_mapping(loader,node,deep=False):
    loader.flatten_mapping(node)
    result={}
    for key,value in loader.construct_pairs(node,deep=deep):
        try:
            if key in result:
                raise yaml.constructor.ConstructorError(None,None,'YAML字段重复',node.start_mark)
            result[key]=value
        except TypeError:
            raise yaml.constructor.ConstructorError(None,None,'YAML字段必须可用作键',node.start_mark) from None
    return result


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,unique_mapping)


def load_yaml(path,raw=None):
    """同一份原始字节用于解析和审计；错误不回显可能包含认证信息的源行。"""
    raw=path.read_bytes() if raw is None else raw
    try:
        data=yaml.load(raw.decode('utf-8-sig'),Loader=UniqueSafeLoader)
    except (yaml.YAMLError,UnicodeDecodeError):
        raise ValueError('YAML语法错误或字段重复，请核查配置文件：'+str(path)) from None
    if not isinstance(data,dict):
        raise ValueError('配置文件必须为YAML对象：'+str(path))
    return data


def positive(value,name,allow_zero=False):
    if (type(value) not in (int,float) or not math.isfinite(value)
        or value<0 or (value==0 and not allow_zero)):
        raise ValueError(name+'必须为有限'+('非负数' if allow_zero else '正数'))


SCENE_SCHEMA_VERSION='carla-experiment-2.0'
SCENE_DIRECTORY=Path('config/carla_scenarios')


def expand_routes(data, base):
    """Resolve route files beside the declaring YAML; freeze contents, never their path."""
    data=copy.deepcopy(data)
    def expand(route):
        if isinstance(route,dict) and 'file' in route:
            if set(route)!={'file'}:raise ValueError('route.file不能与内嵌路线字段混用')
            route_path=(base/route['file']).resolve()
            value=json.loads(route_path.read_text(encoding='utf-8-sig'))
            return value
        return route
    data['experiment']['route']=expand(data['experiment']['route'])
    for actor in data['experiment']['actors']:
        if 'route' in actor['placement']:
            actor['placement']['route']=expand(actor['placement']['route'])
    return data


def scene_configuration(path,seed):
    from nine_grid.carla_experiment_spec import resolve_spec,settings_from_manifest
    raw=path.read_bytes()
    data=load_yaml(path,raw)
    if data.get('version')=='carla-instance-2.0':
        experiment=settings_from_manifest(data)
        manifest=data
        scene={'name':data['scenario_name'],'map':data['map'],
            'desired_speed_mps':data['desired_speed_mps'],'duration_s':data['duration_s'],
            'description':'冻结实例 '+data['instance_hash'],'status':data['status'],'split':data['split']}
        frozen=True
    else:
        scene=expand_routes(data,path.parent)
        experiment,manifest=resolve_spec(scene,seed)
        frozen=False
    source={'schema_version':SCENE_SCHEMA_VERSION,'effective_schema_version':SCENE_SCHEMA_VERSION,
        'route_selection_version':'locked-opendrive-route-2.0','path':str(path),
        'sha256':hashlib.sha256(raw).hexdigest(),'yaml':raw.decode('utf-8-sig'),
        'name':scene['name'],'template':'experiment_v2','requested_map':scene['map'],
        'requested_desired_speed_mps':scene['desired_speed_mps'],'description':scene.get('description'),
        'duration_s':scene['duration_s'],'instance_hash':manifest['instance_hash'],
        'manifest':manifest,'frozen_input':frozen,'status':manifest['status'],'split':manifest['split']}
    return scene,experiment,source


def validate_trial_contract(args,cfg):
    method=args.method
    if method=='A' and (args.fast!='off' or args.retrieve):raise ValueError('A组必须关闭模型和检索')
    if method in ('B','C') and args.fast=='off':raise ValueError('B/C组必须通过--fast api或server启用同一模型')
    if method=='B' and args.retrieve:raise ValueError('B组必须关闭检索')
    if method=='C' and not args.retrieve:raise ValueError('C组必须显式--retrieve')
    if method and (args.reflection!='off' or args.write_memory):
        raise ValueError('A/B/C配对比较关闭反思及在线记忆写入')
    if args.experiment_phase=='formal':
        if not args.instance_config:raise ValueError('正式实验必须使用--instance-config冻结实例')
        if cfg['scenario_source']['split']!='test':raise ValueError('正式实例必须显式属于test划分')
        if args.duration is not None:raise ValueError('正式实验时限来自冻结实例，不允许命令覆盖')
        if cfg['runtime_mode']!='sync':raise ValueError('正式主表使用sync；async请单列development实验')
        if cfg['world_plan_debug']:raise ValueError('正式实验禁止世界debug线污染模型RGB；录像副本仍可叠加')
        if not method:raise ValueError('正式对比必须指定--method A/B/C')
        if method=='C' and not args.memory_read_dir:raise ValueError('C组正式实验必须显式指定冻结的--memory-read-dir')
        if not args.validation_report:raise ValueError('正式实验需要--validation-report真实服务器验收报告')
        report=json.loads(project_path(args.validation_report).read_text(encoding='utf-8-sig'))
        from nine_grid.carla_experiment_validation import verify_report,sdk_supported,infrastructure_contract
        if (report.get('version')!='carla-instance-validation-2.0' or report.get('instance_hash')!=cfg['scenario_source']['instance_hash']
            or report.get('passed') is not True or not sdk_supported(report.get('server_version'))):
            raise ValueError('验收报告版本、实例、服务器版本或结论不匹配')
        checks=('route','actors','events','source_alignment','repeatability','cleanup','video')
        if any(report.get('checks',{}).get(k) is not True for k in checks):
            raise ValueError('验收报告缺少完整物理闭环检查')
        cfg['validation_report']=verify_report(project_path(args.validation_report),cfg['scenario_source']['instance_hash'])
        if cfg['validation_report'].get('implementation_sha256')!=cfg['implementation_sha256']:
            raise ValueError('场景/桥接/评价实现已变更，需要重新技术验收')
        if cfg['validation_report'].get('infrastructure_contract')!=infrastructure_contract(cfg):
            raise ValueError('相机或地图几何配置已变更，需要重新技术验收；规划器参数不属于该门禁')
    cfg['experiment_phase']=args.experiment_phase
    cfg['comparison_method']=method


def memory_fingerprint(directory):
    if not directory.is_dir():raise ValueError('记忆库目录不存在：'+str(directory))
    rows=[]
    for path in sorted(p for p in directory.rglob('*') if p.is_file()):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        rows.append([path.relative_to(directory).as_posix(),h.hexdigest()])
    if not rows:raise ValueError('正式C组记忆库为空')
    return {'sha256':hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),'files':len(rows)}


def refresh_scene_audit(cfg):
    """每轮完整复制实际实验设置，避免下一轮seed更新污染前轮来源审计。"""
    source=cfg['scenario_source']
    source['effective_map']=cfg['map']
    source['effective_desired_speed_mps']=cfg['desired_speed_mps']
    source['effective_experiment']=copy.deepcopy(cfg['experiment'])


def platform_configuration(data,args,scene_path):
    if set(data)!={'carla'} or not isinstance(data['carla'],dict):
        raise ValueError('平台YAML必须且仅包含carla对象；实验参数请放到独立场景YAML')
    allowed={'python','host','port','timeout_s','allow_world_reset','camera_width','camera_height',
        'camera_fov','camera_height_m','arrival_margin_m','map_resolution_m',
        'map_speed_cap_mps','same_level_tolerance_m','runtime_mode','planner_hz','camera_hz',
        'world_plan_debug','display_plan_overlay','camera_save_hz','rgb_postprocess_effects','display_queue_frames',
        'longitudinal_control'}
    if set(data['carla'])-allowed:
        raise ValueError('carla包含未实现字段：'+str(sorted(set(data['carla'])-allowed)))
    scene,experiment,source=scene_configuration(scene_path,args.seed)
    if args.map is not None and args.map!=scene['map']:
        raise ValueError('锁定路线禁止--map换图；请选择对应地图的场景文件')
    cfg=copy.deepcopy(data['carla'])
    from nine_grid.carla_execution import SignedAccelerationController
    longitudinal=cfg.get('longitudinal_control',{})
    if not isinstance(longitudinal,dict):raise ValueError('carla.longitudinal_control必须为参数对象')
    # Only validates the SDK-independent controller and expands effective defaults.
    # The measured steering geometry is installed by the bridge during setup.
    cfg['longitudinal_control']=copy.deepcopy(SignedAccelerationController(.6,settings=longitudinal).settings)
    cfg.setdefault('runtime_mode','sync')
    if getattr(args,'runtime_mode',None) is not None:
        cfg['runtime_mode']=args.runtime_mode
    if cfg['runtime_mode'] not in ('sync','async'):
        raise ValueError('carla.runtime_mode必须为sync或async')
    cfg.setdefault('planner_hz',10)
    cfg.setdefault('camera_hz',10)
    if type(cfg['planner_hz']) is not int or cfg['planner_hz'] not in (1,2,10):
        raise ValueError('carla.planner_hz必须为1、2或10Hz整数')
    if type(cfg['camera_hz']) not in (int,float) or cfg['camera_hz']!=10:
        raise ValueError('carla.camera_hz必须为10Hz；原始图片按camera_save_hz=1归档')
    for key,default in (('world_plan_debug',False),('display_plan_overlay',True),
                        ('rgb_postprocess_effects',True)):
        cfg.setdefault(key,default)
        if type(cfg[key]) is not bool:raise ValueError('carla.'+key+'必须为布尔值')
    cfg.setdefault('camera_save_hz',1)
    cfg.setdefault('display_queue_frames',50)
    if type(cfg['display_queue_frames']) is not int or cfg['display_queue_frames']<10:
        raise ValueError('carla.display_queue_frames必须为至少10的整数')
    if type(cfg['camera_save_hz']) not in (int,float) or cfg['camera_save_hz']!=1:
        raise ValueError('carla.camera_save_hz当前固定为1Hz，与1Hz模型源帧同步')
    if getattr(args,'world_plan',None) is not None:
        cfg['world_plan_debug']=args.world_plan
    cfg['map']=scene['map']
    cfg['desired_speed_mps']=scene['desired_speed_mps']
    for key in ('python','host','port','map'):
        value=getattr(args,'carla_python' if key=='python' else key)
        if value is not None:cfg[key]=value
    cfg['exclusive_world']=bool(args.exclusive_world)
    cfg['allow_world_reset']=bool(args.allow_world_reset)
    for key in ('host','map'):
        if not isinstance(cfg.get(key),str) or not cfg[key].strip():
            raise ValueError('carla.'+key+'必须为非空字符串')
    if type(cfg.get('port')) is not int or not 1<=cfg['port']<=65535:
        raise ValueError('CARLA端口必须为1至65535的整数')
    for key in ('camera_width','camera_height'):
        if type(cfg.get(key)) is not int or cfg[key]<64 or cfg[key]%2:
            raise ValueError(key+'必须为不小于64的偶数')
    for key in ('timeout_s','camera_fov','camera_height_m','arrival_margin_m','map_resolution_m','map_speed_cap_mps',
                'same_level_tolerance_m'):
        positive(cfg.get(key),'carla.'+key)
    positive(cfg.get('desired_speed_mps'),'carla.desired_speed_mps',True)
    if not 20<cfg['camera_fov']<150:
        raise ValueError('相机FOV须位于(20,150)度')
    if cfg.get('python') is not None:
        python=Path(cfg['python']).expanduser()
        if not python.is_absolute():raise ValueError('--carla-python必须是SDK解释器的绝对路径')
        cfg['python']=str(python)
    source['map_overridden']=args.map is not None
    source['command_overrides']={key:getattr(args,'carla_python' if key=='python' else key)
        for key in ('python','host','port','map')
        if getattr(args,'carla_python' if key=='python' else key) is not None}
    source['command_overrides'].update(exclusive_world=bool(args.exclusive_world),
        allow_world_reset=bool(args.allow_world_reset))
    if getattr(args,'world_plan',None) is not None:
        source['command_overrides']['world_plan_debug']=args.world_plan
    cfg['experiment']=experiment
    cfg['scenario_source']=source
    refresh_scene_audit(cfg)
    names=('carla_bridge.py','carla_adapter.py','carla_execution.py','carla_experiments.py',
           'carla_experiment_spec.py','carla_experiment_routes.py','carla_evaluation.py','carla_task_criteria.py',
           'carla_initialization.py','carla_public_priority.py','carla_sensor_policy.py','carla_experiment_validation.py',
           'carla_world_ownership.py','object_scope.py','lane_roles.py','planning_objects.py',
           'models.py','representations.py','runner.py','highway.py','runtime.py')
    files={name:hashlib.sha256((ROOT/'src/nine_grid'/name).read_text(encoding='utf-8-sig').replace('\r\n','\n').encode('utf-8')).hexdigest() for name in names}
    files['scripts/run_carla.py']=hashlib.sha256(Path(__file__).read_text(encoding='utf-8-sig').replace('\r\n','\n').encode('utf-8')).hexdigest()
    cfg['implementation_sha256']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    return cfg


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='EP-Driver CARLA 0.9.15独立世界实验')
    parser.add_argument('--config',default='config/parameters.json')
    parser.add_argument('--carla-config',default='config/carla.yaml')
    parser.add_argument('--carla-python',help='CARLA SDK Python 3.7/3.8解释器绝对路径')
    parser.add_argument('--host');parser.add_argument('--port',type=int);parser.add_argument('--map')
    parser.add_argument('--runtime-mode',choices=['sync','async'],help='同步步进或世界与模型均异步；默认读取CARLA平台YAML')
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--scenario',choices=sorted(p.stem for p in (ROOT/SCENE_DIRECTORY).glob('*.yaml')),help='选择内置v2场景，默认town01_follow_stop_go')
    selection.add_argument('--scenario-config',help='指定独立v2场景YAML；相对路径以项目根为准')
    selection.add_argument('--instance-config',help='加载已冻结实例JSON；不重新采样参数')
    parser.add_argument('--freeze-instance',help='离线解析单个实例并保存JSON，然后退出')
    parser.add_argument('--method',choices=['A','B','C'],help='严格基础/模型/模型加检索三组开关检查')
    parser.add_argument('--experiment-phase',choices=['development','formal'],default='development')
    parser.add_argument('--validation-report',help='formal所需、绑定实例哈希的服务器验收报告')
    parser.add_argument('--exclusive-world',action='store_true',help='确认此端口为本实验专用CARLA世界')
    parser.add_argument('--allow-world-reset',action='store_true',help='只在专用空世界允许按配置切换地图')
    parser.add_argument('--list-scenarios',action='store_true')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true',help='只检查配置并显示运行清单，不连接任何服务')
    mode.add_argument('--inspect-scene',action='store_true',help='只读连接当前地图检查所有候选起点，不换地图、不生成演员、不推进仿真')
    parser.add_argument('--models-config',default='config/models.yaml')
    parser.add_argument('--prompts-config',default='config/prompts.yaml')
    parser.add_argument('--fast',choices=['off','api','server'],default='off')
    parser.add_argument('--reflection',choices=['off','api','server'],default='off')
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--seed-mode',choices=['fixed','increment'],default='increment')
    parser.add_argument('--duration',type=float)
    parser.add_argument('--prefix',default=datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--episodes-dir',default='outputs/carla/episodes')
    parser.add_argument('--retrieve',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--write-memory',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--memory-mode',choices=['collect','formal'],default='collect')
    parser.add_argument('--memory-dir',default='outputs/memory_db/carla')
    parser.add_argument('--memory-read-dir');parser.add_argument('--memory-write-dir')
    parser.add_argument('--dino-model')
    parser.add_argument('--live',action='store_true',help='在有桌面的本机显示相机拼图；无桌面服务器请使用外部旁观端')
    parser.add_argument('--video',action='store_true',help='录制每轮六相机RGB显示拼图，退出时检查可播放性')
    parser.add_argument('--world-plan',action=argparse.BooleanOptionalAction,default=None,
        help='在CARLA世界绘制有限寿命规划线；相机可能拍到，严格原图采集使用--no-world-plan')
    args=parser.parse_args(argv)
    try:
        if args.episodes<1 or args.seed<0:raise ValueError('轮数必须为正整数，seed必须非负')
        if args.duration is not None:positive(args.duration,'duration')
        if not args.prefix or Path(args.prefix).name!=args.prefix or args.prefix in ('.','..'):
            raise ValueError('prefix必须为单个目录名片段')
        scene_path=project_path(args.instance_config or args.scenario_config) if (args.instance_config or args.scenario_config) else project_path(SCENE_DIRECTORY/((args.scenario or 'town01_follow_stop_go')+'.yaml'))
        platform=load_yaml(project_path(args.carla_config))
        cfg=platform_configuration(platform,args,scene_path)
        if args.freeze_instance:
            if args.episodes!=1:raise ValueError('--freeze-instance每次保存一个seed实例')
            out=project_path(args.freeze_instance)
            if out.exists():raise ValueError('冻结实例文件已存在，请换路径，避免覆盖已配对实例')
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(cfg['scenario_source']['manifest'],ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            print(json.dumps({'冻结实例':str(out),'instance_hash':cfg['scenario_source']['instance_hash'],'状态':cfg['scenario_source']['status'],'说明':'仅冻结参数和路线，不代表服务器动态验收'},ensure_ascii=False))
            return 0
        if args.list_scenarios:
            files=[scene_path] if args.scenario_config or args.scenario else sorted(project_path(SCENE_DIRECTORY).glob('*.yaml'))
            listed={}
            for path in files:
                _,experiment,source=scene_configuration(path,args.seed)
                if source['name'] in listed:raise ValueError('内置场景name重复：'+source['name'])
                listed[source['name']]={'template':source['template'],'map':source['requested_map'],
                    'config':str(path),'description':source['description'],'duration_s':source['duration_s'],
                    'family':experiment['family'],'status':source['status'],'split':source['split'],
                    'actors':[a['role'] for a in experiment['actors']],
                    'route_m':experiment['route'].get('length_m'),'sha256':source['sha256']}
            print(json.dumps({'场景清单':listed,'说明':'仅列出独立场景配置；未连接CARLA，也不证明地图适配成功'},ensure_ascii=False,indent=2))
            return 0
        validate_trial_contract(args,cfg)
        config=json.loads(project_path(args.config).read_text(encoding='utf-8-sig'))
        config.update(platform='carla',scenario=cfg['scenario_source']['name'],carla=cfg)
        # 重规划间隔与优化轨迹离散步长分离，保留0.1秒×30段的MPC。
        config['clock']['planner_hz']=cfg['planner_hz']
        cfg['runtime_version']='carla-range-clock-3.0'
        cfg['observation_policy']='range_limited_ground_truth_no_occlusion-1.0'
        config['perception'].update(fully_occluded_in_current_planning=True,
            occlusion_tested=False,input_policy=cfg['observation_policy'])
        if cfg['scenario_source']['duration_s'] is not None:config['duration']=cfg['scenario_source']['duration_s']
        if args.duration is not None:config['duration']=args.duration
        if args.dino_model:
            local=project_path(args.dino_model)
            config['dino']['model']=str(local) if local.is_dir() else args.dino_model
        elif not Path(config['dino']['model']).is_absolute():
            local=project_path(config['dino']['model'])
            if local.is_dir():config['dino']['model']=str(local)
        model_file=project_path(args.models_config)
        if not model_file.is_file():
            if args.fast!='off' or args.reflection!='off':
                raise ValueError('启用模型必须先提供实际models配置，不能使用模板假装已完成配置')
            model_file=project_path('config/models.example.yaml')
        model_settings=load_yaml(model_file)
        from nine_grid.models import load_system_prompts
        prompts=load_system_prompts(project_path(args.prompts_config))
        for role in ('fast','reflection'):
            if not isinstance(model_settings.get(role),dict):raise ValueError('模型配置缺少'+role+'对象')
            settings=model_settings[role]
            if set(settings)-{'backend','model','base_url','api_key_env','timeout_s','max_tokens'}:
                raise ValueError(role+'含未支持的模型配置字段；密钥仅通过api_key_env指定的环境变量读取')
            if 'max_tokens' in settings and (type(settings['max_tokens']) is not int or settings['max_tokens']<1):
                raise ValueError(role+'.max_tokens必须为正整数')
            settings.update(backend=getattr(args,role),system_prompt=prompts[role],
                system_prompt_sha256=hashlib.sha256(prompts[role].encode('utf-8')).hexdigest(),
                prompts_config=str(project_path(args.prompts_config)))
            if settings['backend']!='off' and any(not isinstance(settings.get(k),str) or not settings[k].strip() for k in ('model','base_url')):
                raise ValueError(role+'必须配置非空model与base_url')
            settings.setdefault('timeout_s',config['models'][role+'_timeout_s'])
            positive(settings['timeout_s'],role+'.timeout_s')
            config['models'].update({role+'_backend':settings['backend'],role+'_timeout_s':settings['timeout_s']})
        config['memory'].update(read_enabled=args.retrieve,write_enabled=args.write_memory,mode=args.memory_mode)
        from nine_grid.configuration import validate_config
        config=validate_config(config)
        from nine_grid.carla_bridge import validate_camera_source_clock
        validate_camera_source_clock(config.get('clock', {}))
        paths={'episodes':project_path(args.episodes_dir),'read':project_path(args.memory_read_dir or args.memory_dir),
               'write':project_path(args.memory_write_dir or args.memory_dir)}
        seeds=([cfg['experiment']['seed']]*args.episodes if cfg['scenario_source']['frozen_input'] else
               [args.seed if args.seed_mode=='fixed' else args.seed+i for i in range(args.episodes)])
        if args.dry_run:
            from nine_grid.carla_sensor_policy import sensor_policy
            print(json.dumps({'模式':'只检查配置；未连接CARLA、模型服务或DINO',
                '平台':config['platform'],'场景实例':cfg['scenario_source']['name'],'场景文件':str(scene_path),
                '冻结实例':{'instance_hash':cfg['scenario_source']['instance_hash'],'地图':cfg['map'],
                    'status':cfg['scenario_source']['status'],'split':cfg['scenario_source']['split'],
                    '路线米':cfg['experiment']['route'].get('length_m'),
                    '演员数':len(cfg['experiment']['actors'])},
                '平台配置':{k:v for k,v in cfg.items() if k not in ('experiment','scenario_source')},
                'seeds':seeds,'最长仿真秒':config.get('duration',config['clock']['episode_seconds']),
                'fast':args.fast,'reflection':args.reflection,'检索':args.retrieve,'写入':args.write_memory,
                '运行时钟':{'runtime_mode':cfg['runtime_mode'],'physics_hz':config['clock']['physics_hz'],
                    'planner_hz':config['clock']['planner_hz'],'mpc_step_s':config['mpc']['step_s'],
                    'mpc_horizon_s':config['mpc']['step_s']*config['mpc']['horizon_steps']},
                '传感器策略':sensor_policy(fast_enabled=args.fast!='off',reflection_enabled=args.reflection!='off',video=args.video,live=args.live),
                '输出与库路径':{k:str(v) for k,v in paths.items()},'模型配置文件':str(model_file),
                '提示词SHA256':{role:model_settings[role]['system_prompt_sha256'] for role in ('fast','reflection')},
                '真实运行待满足':['已启动CARLA 0.9.15专用空世界','--exclusive-world','有效--carla-python及SDK','启用模型时需可用服务']},ensure_ascii=False,indent=2))
            return 0
        if not args.inspect_scene and not args.exclusive_world:raise ValueError('真实运行须显式指定--exclusive-world确认专用CARLA世界')
        if not cfg.get('python') or not Path(cfg['python']).is_file():
            raise ValueError('真实运行需要--carla-python指向已安装CARLA 0.9.15 SDK的解释器文件')
    except (OSError,ValueError,TypeError,KeyError,yaml.YAMLError) as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    if args.inspect_scene:
        from nine_grid.carla_adapter import BridgeProcess
        paths['episodes'].mkdir(parents=True,exist_ok=True)
        output=paths['episodes']/(args.prefix+'_scene_selection.json')
        bridge=BridgeProcess(cfg['python'],paths['episodes'],timeout_s=float(cfg['timeout_s'])*2+10)
        try:
            result=bridge.call('inspect_scene',config=cfg,seed=cfg['experiment']['seed'])
        finally:
            bridge.close()
        temporary=output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(output)
        print(json.dumps({'模式':'只读锁定路线与演员布置检查','报告':str(output),'检查':result},ensure_ascii=False))
        return 0 if result['success'] else 2
    from nine_grid.models import VisionClient
    from nine_grid.representations import DINOExtractor
    from nine_grid.runner import run_episode
    try:
        fast=VisionClient('fast',model_settings['fast']);reflection=VisionClient('reflection',model_settings['reflection'])
        diagnostics=config.get('diagnostics',{})
        extractor=(DINOExtractor(config['dino']['model'],device=config['dino'].get('device'),
                   local_files_only=config['dino'].get('local_files_only',True))
                   if args.retrieve or args.write_memory or (diagnostics.get('enabled',True) and diagnostics.get('dino',True)) else None)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.error(str(exc))
    frozen_memory=None
    if args.experiment_phase=='formal' and args.method=='C':
        frozen_memory=memory_fingerprint(paths['read'])
        config['carla']['memory_snapshot']=frozen_memory
    summaries=[]
    for index,seed in enumerate(seeds):
        episode_config=copy.deepcopy(config)
        _,instance,source=scene_configuration(scene_path,seed)
        episode_config['carla']['experiment']=instance
        episode_config['carla']['scenario_source']=source
        refresh_scene_audit(episode_config['carla'])
        identity=f"{args.prefix}_{cfg['scenario_source']['name']}_{index:03d}_seed{seed}"
        summaries.append(run_episode(episode_config,model_settings,identity,paths['episodes'],paths['read'],paths['write'],
            args.retrieve,args.write_memory,args.memory_mode,seed,args.live,args.video,fast,reflection,extractor))
    memory_unchanged=frozen_memory is None or memory_fingerprint(paths['read'])==frozen_memory
    for result in summaries:
        result['comparison_protocol']={'method':args.method,'phase':args.experiment_phase,
            'memory_snapshot':frozen_memory,'memory_unchanged':memory_unchanged,
            'valid':memory_unchanged}
        if not memory_unchanged:result['comparison_protocol']['error']='frozen_memory_changed_during_batch'
    paths['episodes'].mkdir(parents=True,exist_ok=True)
    report=paths['episodes']/f'{args.prefix}_batch.json'
    temporary=report.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(summaries,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(report)
    print(json.dumps({'批量报告':str(report),'轮数':len(summaries),'冻结记忆未改变':memory_unchanged,
        '到达':sum(bool(x.get('arrival')) for x in summaries),
        '合规完成':sum(bool(x.get('evaluation',{}).get('strict_success')) for x in summaries),
        '超时':sum(x['termination_reason']=='timeout' for x in summaries),
        '运行异常':sum(x['termination_reason'] in ('runtime_error','initialization_failure') for x in summaries)},ensure_ascii=False))
    return int(not memory_unchanged or any(x['termination_reason'] in ('runtime_error','planning_failure','initialization_failure') for x in summaries))


if __name__=='__main__':
    raise SystemExit(main())
