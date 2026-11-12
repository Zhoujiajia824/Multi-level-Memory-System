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


SCENE_SCHEMA_VERSION='carla-scene-1.2'
SCENE_DIRECTORY=Path('config/carla_scenarios')


def scene_configuration(path,seed):
    """每份场景文件独立声明地图、模板和完整实验参数，不继承其他场景。"""
    from nine_grid.carla_scenarios import (SCENARIOS,EXPERIMENT_FIELDS,REQUIRED_EXPERIMENT_FIELDS,
        OPTIONAL_EXPERIMENT_DEFAULTS,ROUTE_SELECTION_VERSION,validate_experiment)
    raw=path.read_bytes()
    data=load_yaml(path,raw)
    required={'schema_version','name','template','map','desired_speed_mps','experiment'}
    optional={'description','duration_s'} if data.get('schema_version')==SCENE_SCHEMA_VERSION else set()
    if required-set(data) or set(data)-required-optional:
        raise ValueError('场景YAML缺少必需字段或包含未知字段；1.2版另支持description和duration_s')
    if data['schema_version'] not in ('carla-scene-1.0','carla-scene-1.1',SCENE_SCHEMA_VERSION):
        raise ValueError('场景schema_version不兼容：'+str(data['schema_version']))
    if 'duration_s' in data:positive(data['duration_s'],'场景duration_s')
    if 'description' in data and (not isinstance(data['description'],str) or not data['description'].strip()):
        raise ValueError('场景description必须为非空文字')
    name=data['name']
    if (not isinstance(name,str) or not name or
        any(not (c.isascii() and (c.isalnum() or c in '_-')) for c in name) or
        not name[0].isalnum()):
        raise ValueError('场景name必须是字母或数字开头的ASCII字母、数字、下划线、连字符名称')
    if data['template'] not in SCENARIOS:
        raise ValueError('场景template不是已实现的交互模板')
    if not isinstance(data['map'],str) or not data['map'].strip():
        raise ValueError('场景map必须为非空CARLA地图名称或完整资源路径')
    positive(data['desired_speed_mps'],'场景desired_speed_mps',True)
    experiment=data['experiment']
    if not isinstance(experiment,dict):raise ValueError('场景experiment必须为对象')
    scripted=data['template']=='scripted_traffic'
    if scripted:
        if data['schema_version']!=SCENE_SCHEMA_VERSION:
            raise ValueError('scripted_traffic需要carla-scene-1.2')
        from nine_grid.carla_scripted_scenarios import (SCRIPTED_EXPERIMENT_FIELDS,
            SCRIPTED_REQUIRED_FIELDS,SCRIPTED_DEFAULTS,validate_scripted_experiment)
        experiment_fields=SCRIPTED_EXPERIMENT_FIELDS
        required_fields=SCRIPTED_REQUIRED_FIELDS
        defaults=SCRIPTED_DEFAULTS
    else:
        experiment_fields=EXPERIMENT_FIELDS
        required_fields=REQUIRED_EXPERIMENT_FIELDS
        defaults=OPTIONAL_EXPERIMENT_DEFAULTS
    fields=set(experiment_fields)-{'name','seed'}
    unknown=set(experiment)-fields
    missing=(set(required_fields)-{'name','seed'})-set(experiment)
    if unknown:raise ValueError('CARLA场景experiment包含未实现字段：'+str(sorted(unknown)))
    if missing:raise ValueError('CARLA场景experiment缺少显式参数：'+str(sorted(missing)))
    experiment=copy.deepcopy(experiment)
    for key,value in defaults.items():
        experiment.setdefault(key,copy.deepcopy(value))
    experiment.update(name=data['template'],seed=seed)
    (validate_scripted_experiment if scripted else validate_experiment)(experiment)
    index=experiment.get('spawn_index')
    if index is not None and (type(index) is not int or index<0):
        raise ValueError('spawn_index必须为null或非负整数')
    for field in (('ego_blueprint',) if scripted else ('ego_blueprint','lead_blueprint','occluder_blueprint')):
        if not isinstance(experiment.get(field),str) or not experiment[field].strip():
            raise ValueError(field+'必须为非空字符串')
    if scripted:
        from nine_grid.carla_extended_routes import ROUTE_SELECTION_VERSION
    return data,experiment,{'schema_version':data['schema_version'],
        'effective_schema_version':SCENE_SCHEMA_VERSION,'route_selection_version':ROUTE_SELECTION_VERSION,'path':str(path),
        'sha256':hashlib.sha256(raw).hexdigest(),'yaml':raw.decode('utf-8-sig'),
        'name':name,'template':data['template'],'requested_map':data['map'],
        'requested_desired_speed_mps':data['desired_speed_mps'],
        'description':data.get('description'), 'duration_s':data.get('duration_s')}


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
        'map_speed_cap_mps','same_level_tolerance_m','observation_resolution_m','observation_heights_m',
        'depth_empty_tolerance_m','visibility_surface_tolerance_m','visibility_min_samples',
        'world_plan_debug','display_plan_overlay','camera_save_hz','rgb_postprocess_effects'}
    if set(data['carla'])-allowed:
        raise ValueError('carla包含未实现字段：'+str(sorted(set(data['carla'])-allowed)))
    scene,experiment,source=scene_configuration(scene_path,args.seed)
    cfg=copy.deepcopy(data['carla'])
    for key,default in (('world_plan_debug',False),('display_plan_overlay',True),
                        ('rgb_postprocess_effects',True)):
        cfg.setdefault(key,default)
        if type(cfg[key]) is not bool:raise ValueError('carla.'+key+'必须为布尔值')
    cfg.setdefault('camera_save_hz',1)
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
    if type(cfg.get('visibility_min_samples')) is not int or not 1<=cfg['visibility_min_samples']<=36:
        raise ValueError('visibility_min_samples必须为1至36的整数')
    for key in ('timeout_s','camera_fov','camera_height_m','arrival_margin_m','map_resolution_m','map_speed_cap_mps',
                'same_level_tolerance_m','observation_resolution_m','depth_empty_tolerance_m','visibility_surface_tolerance_m'):
        positive(cfg.get(key),'carla.'+key)
    positive(cfg.get('desired_speed_mps'),'carla.desired_speed_mps',True)
    if not 20<cfg['camera_fov']<150 or cfg['observation_resolution_m']>2:
        raise ValueError('相机FOV须位于(20,150)度，观测网格边长须位于(0,2]米')
    heights=cfg.get('observation_heights_m')
    if not isinstance(heights,list) or not heights:
        raise ValueError('observation_heights_m必须为非空高度列表')
    for height in heights:positive(height,'观测高度',True)
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
    return cfg


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='EP-Driver CARLA 0.9.15独立世界实验')
    parser.add_argument('--config',default='config/parameters.json')
    parser.add_argument('--carla-config',default='config/carla.yaml')
    parser.add_argument('--carla-python',help='CARLA SDK Python 3.7/3.8解释器绝对路径')
    parser.add_argument('--host');parser.add_argument('--port',type=int);parser.add_argument('--map')
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--scenario',choices=sorted(p.stem for p in (ROOT/SCENE_DIRECTORY).glob('*.yaml')),help='选择内置独立场景YAML，默认slow_lead')
    selection.add_argument('--scenario-config',help='指定独立场景YAML；相对路径以项目根为准')
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
        scene_path=project_path(args.scenario_config) if args.scenario_config else project_path(SCENE_DIRECTORY/((args.scenario or 'slow_lead')+'.yaml'))
        platform=load_yaml(project_path(args.carla_config))
        cfg=platform_configuration(platform,args,scene_path)
        if args.list_scenarios:
            files=[scene_path] if args.scenario_config or args.scenario else sorted(project_path(SCENE_DIRECTORY).glob('*.yaml'))
            listed={}
            for path in files:
                _,experiment,source=scene_configuration(path,args.seed)
                if source['name'] in listed:raise ValueError('内置场景name重复：'+source['name'])
                listed[source['name']]={'template':source['template'],'map':source['requested_map'],
                    'config':str(path),'description':source['description'],'duration_s':source['duration_s'],
                    'experiment':experiment,'sha256':source['sha256']}
            print(json.dumps({'场景清单':listed,'说明':'仅列出独立场景配置；未连接CARLA，也不证明地图适配成功'},ensure_ascii=False,indent=2))
            return 0
        config=json.loads(project_path(args.config).read_text(encoding='utf-8-sig'))
        config.update(platform='carla',scenario=cfg['scenario_source']['name'],carla=cfg)
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
        seeds=[args.seed if args.seed_mode=='fixed' else args.seed+i for i in range(args.episodes)]
        if args.dry_run:
            print(json.dumps({'模式':'只检查配置；未连接CARLA、模型服务或DINO',
                '平台':config['platform'],'场景':cfg['experiment'],'场景实例':cfg['scenario_source']['name'],'场景文件':str(scene_path),'平台配置':cfg,
                'seeds':seeds,'最长仿真秒':config.get('duration',config['clock']['episode_seconds']),
                'fast':args.fast,'reflection':args.reflection,'检索':args.retrieve,'写入':args.write_memory,
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
            result=bridge.call('inspect_scene',config=cfg,seed=args.seed)
        finally:
            bridge.close()
        temporary=output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(output)
        print(json.dumps({'模式':'只读选址检查','报告':str(output),'地图':result['map'],
            '成功':result['success'],'选定起点':result['selected_spawn_index'],
            '可用起点':[row['spawn_index'] for row in result['candidates'] if row['reason']=='accepted'],
            '拒绝原因':result['rejection_counts'],'交互段m':result['interaction_length_m'],
            '导航长度m':result['route_length_m']},ensure_ascii=False))
        return 0 if result['success'] else 2
    from nine_grid.models import VisionClient
    from nine_grid.representations import DINOExtractor
    from nine_grid.runner import run_episode
    try:
        fast=VisionClient('fast',model_settings['fast']);reflection=VisionClient('reflection',model_settings['reflection'])
        extractor=DINOExtractor(config['dino']['model']) if args.retrieve or args.write_memory else None
    except (ValueError,RuntimeError,OSError) as exc:
        parser.error(str(exc))
    summaries=[]
    for index,seed in enumerate(seeds):
        episode_config=copy.deepcopy(config)
        episode_config['carla']['experiment']['seed']=seed
        refresh_scene_audit(episode_config['carla'])
        identity=f"{args.prefix}_{cfg['scenario_source']['name']}_{index:03d}_seed{seed}"
        summaries.append(run_episode(episode_config,model_settings,identity,paths['episodes'],paths['read'],paths['write'],
            args.retrieve,args.write_memory,args.memory_mode,seed,args.live,args.video,fast,reflection,extractor))
    paths['episodes'].mkdir(parents=True,exist_ok=True)
    report=paths['episodes']/f'{args.prefix}_batch.json'
    temporary=report.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(summaries,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(report)
    print(json.dumps({'批量报告':str(report),'轮数':len(summaries),
        '完成或到达':sum(x['termination_reason'] in ('timeout','arrival') for x in summaries)},ensure_ascii=False))
    return int(any(x['termination_reason'] in ('runtime_error','planning_failure','initialization_failure') for x in summaries))


if __name__=='__main__':
    raise SystemExit(main())
