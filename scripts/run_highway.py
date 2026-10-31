"""在项目根目录执行四场景单轮或批量实验。"""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import yaml

# 小规模 SQP 矩阵避免多线程调度开销；已有用户设置优先。
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='九宫格检索增强 MPC highway-env 实验')
    parser.add_argument('--scenario',choices=['highway','merge','roundabout','intersection'],default='highway')
    parser.add_argument('--config',default='config/parameters.json')
    parser.add_argument('--scenarios-config',default='config/scenarios.yaml')
    parser.add_argument('--models-config',default='config/models.yaml')
    parser.add_argument('--prompts-config',default='config/prompts.yaml',help='快模型和反思模型系统提示词YAML')
    parser.add_argument('--fast',choices=['off','api','server'])
    parser.add_argument('--reflection',choices=['off','api','server'])
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--seed-mode',choices=['fixed','increment'],default='increment')
    parser.add_argument('--duration',type=float)
    parser.add_argument('--vehicles',type=int)
    parser.add_argument('--spawn-probability',type=float,help='路口每仿真秒生成车辆概率，0 表示不生成')
    parser.add_argument('--prefix',default=datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--episodes-dir',default='outputs/episodes')
    parser.add_argument('--retrieve',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--write-memory',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--memory-dir',default='outputs/memory_db/default')
    parser.add_argument('--memory-read-dir')
    parser.add_argument('--memory-write-dir')
    parser.add_argument('--memory-mode',choices=['collect','formal'],default='collect')
    parser.add_argument('--dino-model',help='本地 DINOv2 模型目录或 Hugging Face 标识')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--video',action='store_true')
    args=parser.parse_args()
    def path(value):
        p=Path(value)
        return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
    if args.episodes<1 or args.seed<0 or (args.duration is not None and args.duration<=0) or (args.vehicles is not None and args.vehicles<0):
        parser.error('轮数、种子、时长或车辆数不合法')
    config=json.loads(path(args.config).read_text(encoding='utf-8'))
    scenarios=yaml.safe_load(path(args.scenarios_config).read_text(encoding='utf-8'))
    for name,settings in scenarios.items():
        config.setdefault('scenarios',{}).setdefault(name,{}).update(settings)
    config['scenario']=args.scenario
    for key in ('duration','vehicles'):
        if getattr(args,key) is not None:
            config[key]=getattr(args,key)
    if args.spawn_probability is not None:
        if not 0<=args.spawn_probability<=1:
            parser.error('交通生成概率须在 0 至 1 范围内')
        config.setdefault('environment',{})['spawn_probability']=args.spawn_probability
    if args.dino_model:
        config['dino']['model']=str(path(args.dino_model)) if path(args.dino_model).is_dir() else args.dino_model
    model_settings=yaml.safe_load(path(args.models_config).read_text(encoding='utf-8'))
    from nine_grid.models import load_system_prompts
    try:
        prompts=load_system_prompts(path(args.prompts_config))
    except ValueError as exc:
        parser.error(str(exc))
    for role in ('fast','reflection'):
        model_settings[role].update(system_prompt=prompts[role],
            system_prompt_sha256=hashlib.sha256(prompts[role].encode('utf-8')).hexdigest(),
            prompts_config=str(path(args.prompts_config)))
        if getattr(args,role) is not None:
            model_settings[role]['backend']=getattr(args,role)
        model_settings[role].setdefault('timeout_s',config['models'][f'{role}_timeout_s'])
        # 服务配置的显式覆盖必须回填有效参数，保证落盘哈希与实际调用一致。
        config['models'][f'{role}_timeout_s']=model_settings[role]['timeout_s']
        config['models'][f'{role}_backend']=model_settings[role]['backend']
    config['memory'].update(read_enabled=args.retrieve,write_enabled=args.write_memory,mode=args.memory_mode)
    from nine_grid.configuration import validate_config
    try:
        config=validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    from nine_grid.models import VisionClient
    from nine_grid.representations import DINOExtractor
    from nine_grid.runner import run_episode
    try:
        fast=VisionClient('fast',model_settings['fast'])
        reflection=VisionClient('reflection',model_settings['reflection'])
        extractor=DINOExtractor(config['dino']['model']) if args.retrieve or args.write_memory else None
    except (ValueError,RuntimeError) as exc:
        parser.error(str(exc))
    summaries=[]
    for index in range(args.episodes):
        seed=args.seed if args.seed_mode=='fixed' else args.seed+index
        identity=f'{args.prefix}_{args.scenario}_{index:03d}_seed{seed}'
        summaries.append(run_episode(config,model_settings,identity,path(args.episodes_dir),
            path(args.memory_read_dir or args.memory_dir),path(args.memory_write_dir or args.memory_dir),
            args.retrieve,args.write_memory,args.memory_mode,seed,args.live,args.video,fast,reflection,extractor))
    report=path(args.episodes_dir)/f'{args.prefix}_batch.json'
    report.write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'批量报告':str(report),'轮数':len(summaries),'完成或到达':sum(x['termination_reason'] in ('timeout','arrival') for x in summaries)},ensure_ascii=False))
    return int(any(x['termination_reason'] in ('runtime_error','planning_failure','initialization_failure') for x in summaries))


if __name__=='__main__':
    raise SystemExit(main())
