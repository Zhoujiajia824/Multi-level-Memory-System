"""只读环境自检，不下载权重、不调用 API。"""
import importlib.metadata
import json
import sys
from pathlib import Path

if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
result={'python':sys.version,'工作区':str(Path(__file__).resolve().parents[1]),'依赖':{}}
for name in ('highway-env','gymnasium','numpy','scipy','casadi','shapely','Pillow','matplotlib','PyYAML','openai','torch','transformers','opencv-python','pytest'):
    try:
        result['依赖'][name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result['依赖'][name]='缺失'
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(int(any(result['依赖'][p]=='缺失' for p in ('highway-env','gymnasium','numpy','scipy','casadi','shapely','Pillow','PyYAML'))))
