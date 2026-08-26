#!/usr/bin/env bash
# =============================================================================
# CARLA 闭环集成环境搭建：mulmem_carla (Python 3.9)
# =============================================================================
# 在一个 Python 3.9 环境里同时装记忆系统依赖 + faiss + carla 绑定，
# 单进程单环境跑闭环。mulmem(3.12) 保持不动。
#
# 依据：carla_0915(3.9) 的 carla 是 pip 装的（pip show carla = 0.9.15），
# 故此处直接 pip install carla==0.9.15，无需 egg。
#
# 用法（任意目录）：
#   bash carla_bridge/setup_env.sh
# =============================================================================
set -e
cd "$(dirname "$0")/.."   # carla_bridge/ -> 项目根

ENV=mulmem_carla
PY=3.9

echo "[1/5] 创建 conda 环境 $ENV (python=$PY)..."
conda create -n "$ENV" "python=$PY" -y

echo "[2/5] 升级 pip..."
conda run -n "$ENV" python -m pip install --upgrade pip

echo "[3/5] 安装记忆系统依赖 (requirements.txt)..."
conda run -n "$ENV" pip install -r requirements.txt

echo "[4/5] 安装 faiss-cpu==1.9.0 (勿用 1.14.x，与 torch 冲突)..."
conda run -n "$ENV" pip install --force-reinstall --no-cache-dir "faiss-cpu==1.9.0"

echo "[5/5] 安装 carla==0.9.15..."
conda run -n "$ENV" pip install "carla==0.9.15"

echo "[验证] import 关键包 + 记忆系统入口..."
conda run -n "$ENV" python -c "import sys; sys.path.insert(0,'.'); \
import carla, faiss, torch, transformers, pydantic; \
from src.vla_memory.pipeline.online_loop import OnlineDrivingLoop; \
print('ENV_OK | carla+faiss+torch+transformers+pydantic+OnlineDrivingLoop 全部可用')"

echo ""
echo "完成。激活环境： conda activate $ENV"
echo "启动 CARLA 服务器（另一终端）： D:\\software\\carla\\WindowsNoEditor\\CarlaUE4.exe"
echo "跑闭环： python -m carla_bridge.run_carla_demo --scenario straight_traffic --mode memory_on"
