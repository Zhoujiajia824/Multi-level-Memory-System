# Multi-level-Memory-System
用于智驾的外挂式记忆系统开发，逐步适用不同智驾数据集（通用数据集与VLA数据集）与智驾仿真平台

# 智能驾驶 VLA 路线的分层记忆系统 Demo

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ⚠️ **重要声明**：本项目是**离线研究 Demo**，不是实车控制系统。所有输出仅用于学术研究，**不可用于实际驾驶决策**。

---

## 🚗 CARLA 闭环集成（v1.0 新增）

在离线 nuScenes 记忆系统之上，新增 **CARLA 0.9.15 闭环驾驶**集成（`carla_bridge/`，
**不改 `src/vla_memory/` 一行**）：用 CARLA 实时感知（六视角图像 / GT 障碍物 / 天气 / 导航）替代
离线 nuScenes，决策轨迹实时回控 CARLA，实现自定义环境下的闭环驾驶，并支持驾驶视频录制回看。

- **3s 周期全量认知** + **每周期 5 次 raw 捕获**（短期记忆时刻最新）+ **10Hz Pure Pursuit/PID 回控**
- **同步模式**：VLM 思考期间 CARLA 冻结，wall-clock 慢但 sim 时间连续正确
- **自定义场景**：地图 / 天气 / 交通流 / 车辆 / 路由，全 Python API
- **驾驶视频录制**：跳过 VLM 冻结期，视频时长 = 仿真实际驾驶时间
- 独立环境 `mulmem_carla`(Python 3.9)，与 nuScenes 的 `mulmem`(3.12) 互不干扰

> 完整框架 / 模块介绍 / 操作指令 / 故障排查见 **[carla_bridge/README.md](carla_bridge/README.md)**。

快速跑闭环（先手动启动 CARLA 服务器 `CarlaUE4.exe`）：
```bash
conda activate mulmem_carla
export DASHSCOPE_API_KEY=你的key    # Git Bash(MINGW64) 用 export，不是 set
python -m carla_bridge.run_carla_demo --scenario straight_traffic --mode memory_on
```

> 下文 §1–§14 为**离线 nuScenes 记忆系统**的完整文档（v0.1，环境 `mulmem`/Python 3.12）。

---

## 1. 项目简介

本项目是一个 **VLA（Vision-Language-Action）驾驶 Agent 多层记忆系统**的 v0.1 离线 demo。
核心命题：让视觉语言模型（VLM）在驾驶决策时**复用历史经验**与**显式驾驶知识**，
通过对比有/无记忆两种模式，量化记忆系统对规划质量的增益。

### 1.1 核心特性

| 特性 | 说明 |
|---|---|
| **逐帧在线循环** | `OnlineDrivingLoop` 严格按时间顺序：感知→检索→决策→更新记忆。第 *i* 帧检索时索引里**只含 [0, i-1] 帧**，彻底消除批处理 demo 常见的 data leakage |
| **三层记忆** | 短期（deque 滑窗）+ 中期（FAISS 6 路融合检索）+ 长期（YAML 规则按 scene_id 严格匹配） |
| **价值门控事件记忆** | 中期记忆从"逐帧全存"升级为价值门控的事件经验库：6 信号准入门控（拒巡航/稳定停车/冗余帧，留变道/急停/路口/cut-in/行人交互等 17 类高价值事件）+ 容量淘汰（soft delete + FAISS rebuild）+ 价值感知检索重排 + 连续高价值帧→事件级记忆 + 冲突感知软更新 + 离线沉淀长期候选（不覆盖正式库）。✅ 已端到端验证，详见 [§9.2](#92-中期记忆-mid-term-memory) |
| **CAN bus 真值自车状态** | 优先用 nuScenes `pose.json` + `vehicle_monitor.json`（含 yaw_rate / steering / throttle / brake），失败回退到 ego_pose 差分 |
| **结构化场景理解** | VLM 输出 lanes / vehicles / pedestrians / traffic_lights / intersections 等独立列表，不再是混合 JSON |
| **多图决策上下文** | 决策 VLM 收到当前帧 + N-1 张历史图（默认 3 张可配） |
| **六视角环视拼接** | 可选 `surround_mosaic` 模式：6 相机拼成 2×3 单张图替代前视图，进入 VLM/特征/记忆全流程（向后兼容 `single_front`） |
| **Oracle 感知对象** | 可选注入 nuScenes GT 标注投影的 `perception_objects`（检测框/语义/速度/加速度），明确标注为 oracle 真值（非模型预测，研究/评测用） |
| **完整提示词模板化** | 所有 VLM 提示词集中在 `config/prompts.yaml`，改提示词无需改 Python |
| **中断恢复** | jsonl append + fsync；重启时扫已写记录自动跳过；中期记忆可选磁盘持久化跨次累积 |
| **智驾标准评测** | ADE / FDE / **L2@1s/2s/3s** / 轨迹有效率 / 行为准确率 |
| **CARLA 闭环驾驶** | `carla_bridge/` 接入 CARLA 0.9.15：3s 认知 + 10Hz 回控 + raw 捕获 + 驾驶视频录制，自定义场景闭环（v1.0，不改 src/） |
| **预留接口** | `DynamicsPlanner`（轨迹→控制量，给 CARLA 闭环）+ `TrajectorySampler`（多模态轨迹选择，给扩散/AR 模型） |

### 1.2 第一版边界

| 维度 | 第一版 | 后续规划 |
|---|---|---|
| 数据集 | nuScenes v1.0-mini / v1.0-trainval | ✅ CARLA 0.9.15（carla_bridge）、自录数据 |
| 摄像头 | 默认 CAM_FRONT，可选六视角环视拼接（surround_mosaic） | 真 BEV / 特征级多摄像头融合 |
| 图像特征 | DINOv2-base (768 维) | 支持其他 backbone |
| VLM | OpenAI 兼容 API（Qwen-VL 默认） | 本地模型 / 更多 provider |
| 向量检索 | FAISS-CPU IndexFlatIP | IndexIVF / HNSW；GPU FAISS |
| 输出 | ego-centric 轨迹点列 | ✅ 控制量回控 CARLA（carla_bridge）/ 多模态轨迹簇 |
| 评测 | 内部 memory_on vs memory_off 对比 | 接入官方 nuScenes planning benchmark |

---

## 2. 架构概览

### 2.1 高层数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ scripts/07_run_full_demo.py  --mode {memory_on|memory_off}      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ run_full_demo(config, mode, resume) ← full_demo_pipeline.py     │
│                                                                 │
│   [1/3] run_prepare_nuscenes                                    │
│         - NuScenesAdapter (含 CanBusLoader)                     │
│         - NuScenesKeyframeSampler (1Hz)                         │
│                                                                 │
│   [2/3] enrich_keyframes_with_state (per-frame，纯数据准备)     │
│         - EgoStateBuilder.build (CAN bus 优先 / 差分回退)       │
│         - TrajectoryBuilder (history + GT future)               │
│         - RouteInfer (伪导航语义)                                │
│                                                                 │
│   [3/3] OnlineDrivingLoop.run(keyframes)                        │
│         for kf in keyframes: step(kf)                           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        outputs/decisions_<mode>_<run_id>.jsonl
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ scripts/06_run_evaluation.py  --decisions <jsonl>               │
│                              --compare <on.jsonl> <off.jsonl>   │
│                                                                 │
│   eval_pipeline.run_eval_pipeline / run_eval_compare            │
│   - 用 nuScenes ego_pose 构造未来真值轨迹                       │
│   - 计算 ADE / FDE / L2@1s/2s/3s / 行为准确率                   │
│   - ReportWriter 输出 CSV / JSONL / Markdown                    │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
              outputs/reports/{eval_summary.csv,
                               eval_detail.jsonl,
                               eval_report.md}
```

### 2.2 单帧 `step()` 内部 11 步时序（核心正确性保证位置）

```
OnlineDrivingLoop.step(kf):
  ① [resume 短路] sample_token 在 resume_set 中? → return None
  ② [感知] SceneUnderstandingPipeline.process_frame
         ├─ DINOv2.extract → 768-d 特征 → 保存 .npy
         └─ VLM.understand_scene → 结构化 JSON
  ③ [检索] MemoryRetriever.retrieve(use_*=use_memory)
         ├─ short_term : 滑窗中过去 [0, i-1] 帧
         ├─ mid_term   : FAISS 6 路融合检索 [0, i-1] 条记忆   ★
         └─ long_term  : 按 scene_id 严格匹配 YAML 规则
                                                            │
   ★ 关键正确性：mid_term.add_record 在末尾⑩，所以第 i 帧检索时索引里只有过去帧
                                                            │
  ④ [组装 image_paths] 短期图像窗口 + 当前帧（memory_off 仅当前帧）
  ⑤ [决策] DecisionClient.decide(prompt, image_paths)
         → 日志 [DECISION] 打印 raw_response
  ⑥ [parse] parse_decision_output → fallback 兜底
  ⑦ [组装 record] 含 scene/decision/image_paths/记忆 ID 等审计字段
  ⑧ [持久化] append_decision_record + flush + fsync (防中断丢数据)
  ⑨ [审计日志] _log_frame_audit → outputs/logs/online_loop_*.log
  ⑩ [push 短期 + add 中期]  ← 写：仅 memory_on 模式
         short_term.add(item)
         mid_term.add_record(record 含 behavior/reason/trajectory, feature)
  ⑪ return record
```

### 2.3 `memory_on` vs `memory_off` 模式差别（**全文唯一权威表**）

| 维度 | `memory_on` | `memory_off`（对照基线） |
|---|---|---|
| `use_short/mid/long_term` | True | **False** |
| 图像数量 | 1-N 张（含历史） | 仅当前帧 1 张 |
| prompt 中是否含 `## 短期记忆` 段 | 是 | **否** |
| prompt 中是否含 `## 中期相似经验` 段 | 是 | **否** |
| prompt 中是否含 `## 长期规则` 段 | 是 | **否** |
| `step()` 末尾是否更新短期 / 中期 | 是 | **否（保持空窗，对照纯净）** |
| 当前场景理解 / 自车状态 / 历史轨迹 / 导航 | 都传 | 都传 |

### 2.4 中断恢复（Resume）

* 每帧记录 append 到 jsonl 后立即 `f.flush() + os.fsync()`，进程崩溃不丢已写帧
* `OnlineDrivingLoop.setup()` 启动时扫已有 jsonl，把所有 `sample_token` 加入 `_resume_set`
* `step(kf)` 开头检查 `_resume_set` 命中 → `return None` 跳过（不调 VLM，不更新记忆）
* 配合 `config/memory.yaml -> mid_term.persistence.enabled=true`，中期记忆 FAISS 索引也可跨次会话持续累积

---

## 3. 环境准备

### 3.1 环境要求

| 项 | 推荐 |
|---|---|
| 操作系统 | Windows 11（已测试）/ Linux / macOS |
| Conda | Miniconda 或 Anaconda |
| Python | **3.12** |
| GPU | 可选（CPU 可运行；DINOv2 + VLM API 不强依赖 GPU） |
| 网络 | 需访问 HuggingFace（下 DINOv2）+ VLM API（决策） |

### 3.2 5 步配齐环境

```bash
# 步骤 1：创建 conda 环境（环境名必须为 mulmem，与脚本里的路径匹配）
conda create -n mulmem python=3.12 -y
conda activate mulmem
python -m pip install --upgrade pip

# 步骤 2：装 Python 依赖
pip install -r requirements.txt

# 步骤 3：装 FAISS（推荐 pip 1.9.0；不要装 1.14.x，会与 torch 冲突）
pip install --force-reinstall --no-cache-dir "faiss-cpu==1.9.0"

# 步骤 4：环境自检（一行验证关键依赖）
python -c "import nuscenes, pydantic, torch, transformers, openai; \
import faiss; print('OK | faiss', faiss.__version__, \
'| torch', torch.__version__, '| pydantic', pydantic.VERSION)"
# 期望：OK | faiss 1.9.0 | torch 2.x.x+cpu | pydantic 2.x.x

# 步骤 5：设 VLM API Key（以 Qwen-VL 为例）
# Windows PowerShell:
$env:DASHSCOPE_API_KEY = "sk-bad8c24f654042ddbb506eb4dd9bfbd9"
# Windows CMD:
set DASHSCOPE_API_KEY=sk-bad8c24f654042ddbb506eb4dd9bfbd9
# GIT bash:
export DASHSCOPE_API_KEY=sk-bad8c24f654042ddbb506eb4dd9bfbd9
# Linux / macOS:
export DASHSCOPE_API_KEY="your-api-key-here"
```

API Key 申请：[阿里云百炼平台](https://dashscope.console.aliyuncs.com/)。

切换其他 VLM provider（OpenAI / 智谱 / DeepSeek-VL 等）：改
[config/api_models.yaml](config/api_models.yaml) 中的 `base_url` / `model_name` / `api_key_env`。

### 3.3 IDE 中使用 mulmem 环境

VSCode / PyCharm 必须把解释器切到 `mulmem` 环境的 `python.exe`：

```
Windows: D:\software\ANACONDA3_2022.10\envs\mulmem\python.exe
Linux/macOS: ~/anaconda3/envs/mulmem/bin/python
```

否则命令行 `conda activate mulmem` 成功但 IDE 仍报 `ModuleNotFoundError`。

### 3.4 常见环境问题

| 错误 | 解决 |
|---|---|
| `ModuleNotFoundError: nuscenes` | `pip install nuscenes-devkit` |
| `ImportError: ... _swigfaiss ...`（Windows DLL 问题） | 走 `pip install --force-reinstall --no-cache-dir "faiss-cpu==1.9.0"` |
| `NameError: SuperKMeans is not defined`（import torch 后 import faiss） | 同上，**faiss-cpu 1.14.x 与 torch 冲突，固定到 1.9.0** |
| `EnvironmentError: VLM API Key 未设置` | 确认环境变量已 export 且当前 shell 可见 |
| `RuntimeError: DINOv2 模型加载失败` | 跑 `python scripts/00_prepare_models.py` 重新下载 |

### 3.5 准备数据

#### nuScenes v1.0-mini

从 [nuScenes 官网](https://www.nuscenes.org/nuscenes#download) 下载 v1.0-mini，解压到 `data/nuscenes/raw/`：

```
data/nuscenes/raw/
├── v1.0-mini/        # 13 个元数据 JSON
├── samples/          # 关键帧图像（每个传感器一个子目录）
├── sweeps/           # 非关键帧
├── maps/             # 4 张地图 PNG
└── can_bus/          # CAN bus expansion（强烈推荐：含 pose.json + vehicle_monitor.json）
```

可选下载：[CAN bus expansion](https://www.nuscenes.org/nuscenes#download)（解压到 `data/nuscenes/raw/can_bus/`），
启用后自车状态用真值替代差分估算，质量大幅提升。

#### DINOv2 模型权重（首次跑前）

```bash
python scripts/00_prepare_models.py
# 自动从 HuggingFace 下载 facebook/dinov2-base 到 .cache/huggingface
```

---

## 4. 快速开始（5 步跑通完整 Demo）

前置：完成第 3 节环境准备 + 数据集放置。

```bash
conda activate mulmem
cd "<your_path>/vla_memory_demo"

# ── 步骤 1：（可选）环境检查 ──
python -c "import nuscenes, faiss; print('OK')"

# ── 步骤 2：跑 memory_off 基线（约 1-2 分钟，5 帧）──
python scripts/07_run_full_demo.py --mode memory_off --max-scenes 1 --max-frames 5
# 输出: outputs/decisions_memory_off_<run_id>.jsonl

# ── 步骤 3：跑 memory_on（含三层记忆，约 1-2 分钟）──
python scripts/07_run_full_demo.py --mode memory_on --max-scenes 1 --max-frames 5
# 输出: outputs/decisions_memory_on_<run_id>.jsonl

# ── 步骤 4：双 mode 对比评测 ──
python scripts/06_run_evaluation.py --compare \
    outputs/decisions_memory_on_<run_id>.jsonl \
    outputs/decisions_memory_off_<run_id>.jsonl
# 输出: outputs/reports/{eval_summary.csv, eval_detail.jsonl, eval_report.md}

# ── 步骤 5：查看报告 ──
cat outputs/reports/eval_summary.csv
# 或在 IDE 打开 outputs/reports/eval_report.md
```

预期日志（步骤 2/3 每帧 1 段）：

```
[1/5] mode=memory_on frame=ca9a282c9e77460f
[SCENE_UNDERSTANDING] image=... raw_response=
  {"scene_description":"...", "scene_id":"straight_road", ...}
[DECISION] frame=... images=1 (...) raw_response=
  {"behavior":"KEEP_LANE", "trajectory":[...], ...}
=========== AUDIT frame=... mode=memory_on ===========
📷 图片  ...
🚗 自车状态  位置/航向角/速度/...
🧭 导航指令: straight
📊 历史轨迹: 10 个点
🧠 场景理解 (lanes/vehicles/pedestrians/traffic_lights/intersections)
🗂️ 记忆检索 [短期/中期/长期]
📝 决策模型 行为/原因/轨迹
=========== AUDIT END ===========
```

### Resume 中断恢复（自动）

`07_run_full_demo.py` 默认 `--resume`，重跑同样命令会跳过已写入 jsonl 的 sample_token。
强制重跑加 `--no-resume`（覆盖原 jsonl）。

```bash
# 第一次只跑了 3 帧（被 Ctrl+C）
python scripts/07_run_full_demo.py --mode memory_off --max-scenes 1 --max-frames 5
# jsonl 已写 3 条

# 重新执行同样命令
python scripts/07_run_full_demo.py --mode memory_off --max-scenes 1 --max-frames 5
# 日志: "resume 跳过已处理帧: ..." × 3，只跑后 2 帧
```

### 让中期记忆跨次会话累积

```yaml
# config/memory.yaml
mid_term:
  persistence:
    enabled: true              # ← 改成 true
    save_on_close: true
    auto_load_on_init: true
```

下次跑 demo 时，第 1 帧检索就能命中上次跑的所有记忆。

---

## 5. 脚本与 CLI 参考

所有脚本都在 `scripts/` 目录。**`07_run_full_demo.py` 是主入口**，其他脚本为可选/调试用。

### 5.1 完整脚本表

| 脚本 | 用途 | 何时跑 |
|---|---|---|
| `00_check_environment.py` | 自检（依赖、API Key、FAISS、模型权重、数据集） | 配完环境后 |
| `00_prepare_models.py` | 下载 + 验证 DINOv2 权重 | 首次跑前 |
| `01_prepare_nuscenes_index.py` | 生成 nuScenes 索引 JSONL | 调试时单独跑 |
| `02_extract_keyframes.py` | 1Hz 关键帧抽样 + 增强 | 调试时单独跑 |
| `03_run_scene_understanding.py` | **独立**跑 DINOv2 + 场景理解 | 仅调试用；主流程已被 `07` 内部调用 |
| `06_run_evaluation.py` | 评测决策结果 | 跑完 07 之后 |
| `07_run_full_demo.py` | **主入口**：跑在线循环 demo | 主流程 |
| `08_consolidate_long_term_candidates.py` | 中期记忆→长期记忆候选沉淀（Phase 7；离线，不需 VLM/FAISS） | 跑完 07 之后 |

### 5.2 CLI 参数全集

#### `scripts/07_run_full_demo.py`（主入口）

```
--mode {memory_on,memory_off}   ← 必填
--config CONFIG                 配置文件路径（可选）
--dataroot DATAROOT             覆盖 config 的 dataroot
--version VERSION               覆盖 nuScenes 版本（默认 v1.0-mini）
--max-scenes MAX_SCENES         调试用，最多处理几个场景
--max-frames MAX_FRAMES         调试用，每场景最多几帧
--output OUTPUT                 决策 jsonl 输出路径
                                默认: outputs/decisions_<mode>_<run_id>.jsonl
--resume                        启动扫已写 jsonl 跳过已处理帧（默认开启）
--no-resume                     强制重跑，删掉已有 jsonl
```

> **感知输入模式**（`single_front` / `surround_mosaic`）与 oracle 感知对象由 `config/data_nuscenes.yaml`
> 的 `perception` 块控制，**不是 CLI 参数**。详见 [§6.5](#65-感知输入模式六视角环视拼接--oracle-感知对象)。
> ⚠️ 该脚本**没有 `--run-id`** 参数；要避免覆盖既有 jsonl，请用 `--output` 指定独立输出路径。

#### `scripts/06_run_evaluation.py`（评测）

```
--decisions DECISIONS                 单 mode 评测的 jsonl 路径
--mode MODE                           覆盖文件名推断的 mode 标签
--compare JSONL_A JSONL_B             双 mode 对比
--config CONFIG                       配置文件路径
--max-frames MAX_FRAMES               限制评测帧数
--output-dir OUTPUT_DIR               报告输出目录（覆盖配置）
```

`--decisions` 与 `--compare` 互斥；必填其一。

#### 其他脚本

| 脚本 | 主要参数 |
|---|---|
| `01_prepare_nuscenes_index.py` | `--max-scenes` `--version` `--dataroot` `--camera-name` |
| `02_extract_keyframes.py` | `--max-scenes` `--max-frames` `--keyframe-step` `--camera-name` |
| `03_run_scene_understanding.py` | `--max-frames` `--resume` `--force` |

### 5.3 常用命令组合

```bash
# 快速冒烟（1 个场景 5 帧）
python scripts/07_run_full_demo.py --mode memory_off --max-scenes 1 --max-frames 5

# 中等规模（5 个场景，每个 20 帧）
python scripts/07_run_full_demo.py --mode memory_on --max-scenes 5 --max-frames 20

# 不 resume，强制重跑覆盖
python scripts/07_run_full_demo.py --mode memory_on --no-resume

# 指定自定义输出路径（不用默认 run_id 命名）
python scripts/07_run_full_demo.py --mode memory_on --output outputs/my_exp_v1.jsonl

# 单 mode 评测（mode 从文件名推断）
python scripts/06_run_evaluation.py --decisions outputs/decisions_memory_off_20260607.jsonl

# 双 mode 对比 + 限制评测帧数
python scripts/06_run_evaluation.py --max-frames 10 --compare \
    outputs/decisions_memory_on_20260607.jsonl \
    outputs/decisions_memory_off_20260607.jsonl
```

### 5.4 Hard-fail 场景

下列条件不满足时脚本会 hard fail 并提示中文错误信息，**不允许 mock**：

| 条件 | 触发脚本 | 错误信息 |
|---|---|---|
| `DASHSCOPE_API_KEY` 环境变量未设置 | 07, 03 | `EnvironmentError: VLM API Key 未设置` |
| DINOv2 模型权重缺失 | 07, 03 | `RuntimeError: DINOv2 模型加载失败，请运行 00_prepare_models.py` |
| FAISS 未安装 | 07 | `ImportError: FAISS 未安装` |
| nuScenes 数据集不存在 | 07, 06, 01, 02 | `FileNotFoundError: nuScenes 数据集目录不存在` |
| 评测无法构建真值轨迹且 jsonl 无内置真值 | 06 | `RuntimeError: 无法构建任何真值轨迹` |

---

## 6. 配置文件参考

所有 YAML 在 `config/` 目录，启动时自动深合并（[common/config.py](src/vla_memory/common/config.py)）。

| 文件 | 用途 | 主要配置项 |
|---|---|---|
| `default.yaml` | 项目级默认 | `project_name`, `run_id`, `seed`, `device`, `output_dir` |
| `api_models.yaml` | VLM API 模型 | `scene_understanding` / `decision` 的 `provider`, `model_name`, `api_key_env`, `base_url`, `system_prompt` |
| **`prompts.yaml`** | **所有 VLM 提示词模板** | `scene_understanding.user`, `decision.user`, `memory_integration.*`；详见 [docs/prompts.md](docs/prompts.md) |
| `data_nuscenes.yaml` | nuScenes 数据 | `dataroot`, `version`, `camera_name`, `keyframe.step`, `history_seconds`, `future_seconds`, `ego_state.use_can_bus`, `can_bus.{enabled, root, tolerance_us, fallback_to_pose_diff}`, `perception.{mode, cameras, mosaic, oracle_objects, oracle}`（六视角+oracle，详见 [§6.5](#65-感知输入模式六视角环视拼接--oracle-感知对象) 与 [docs/perception_upgrade.md](docs/perception_upgrade.md)） |
| `memory.yaml` | 记忆系统 | `short_term.{capacity, store_image_paths}`, `mid_term.{top_k, weights, persistence.*}`, `long_term.{strict_scene_match, strict_weather_match, rules_file}` |
| `decision.yaml` | 决策模块 | `trajectory.{waypoint_min_num, waypoint_max_num, horizon_seconds, dt}`, `vlm_inputs.{image_context_size, include_current_frame, max_images_per_call}`, `fallback.allow_rule_fallback` |
| `evaluation.yaml` | 评测 | `prediction_horizon_seconds`, `displacement_metrics.resample_num`, `l2_per_horizon.horizons_seconds`, `behavior_accuracy.{nav_to_behavior_map, normalize_case}`, `output.report_dir` |

### 6.1 常见任务速查

| 想做什么 | 改哪个 yaml 哪个 key |
|---|---|
| 改提示词文本 / 加新检测类 | [`config/prompts.yaml`](config/prompts.yaml)（详见 [docs/prompts.md](docs/prompts.md)） |
| 改决策轨迹路点数量约束 | `decision.yaml -> trajectory.waypoint_min_num / max_num`（prompt + parser + schema 自动同步） |
| 改决策图片张数 | `decision.yaml -> vlm_inputs.image_context_size`（默认 3：当前 + 2 历史） |
| 开/关中期记忆磁盘持久化 | `memory.yaml -> mid_term.persistence.enabled` |
| 开/关长期记忆严格 scene 匹配 | `memory.yaml -> long_term.strict_scene_match`（True 时屏蔽 `scene_id="all"` 通配） |
| 开/关 CAN bus 真值 | `data_nuscenes.yaml -> can_bus.enabled` 和 `ego_state.use_can_bus`（两者都 True 才启用） |
| 改 L2 评测时间点 | `evaluation.yaml -> l2_per_horizon.horizons_seconds` |
| 改导航 → 行为映射 | `evaluation.yaml -> behavior_accuracy.nav_to_behavior_map` |
| 切换 VLM provider/模型 | `api_models.yaml -> scene_understanding.* / decision.*` |
| 改短期记忆滑窗大小 | `memory.yaml -> short_term.capacity`（默认 10） |
| 改中期记忆 6 路融合权重 | `memory.yaml -> mid_term.weights.{visual,text,scene,weather,nav,state}_weight` |
| 切换单前视 / 六视角环视 | `data_nuscenes.yaml -> perception.mode`（`single_front` / `surround_mosaic`） |
| 开/关 oracle 感知对象 | `data_nuscenes.yaml -> perception.oracle_objects`（GT 投影，非模型预测） |

---

## 6.5 感知输入模式（六视角环视拼接 + Oracle 感知对象）

> 详细设计见 [docs/perception_upgrade.md](docs/perception_upgrade.md)。

本 demo 支持**两种感知输入模式**，由 `config/data_nuscenes.yaml` 的 `perception` 块切换，**无需改代码**：

| 模式 | `perception.mode` | 主感知图像 | 说明 |
|---|---|---|---|
| **单前视**（默认，向后兼容） | `single_front` | 单张 CAM_FRONT 图 | 原有行为 |
| **六视角环视拼接** | `surround_mosaic` | 6 相机拼成的 2×3 surround-view mosaic | 替代前视图进入 VLM 场景理解/决策/DINOv2/记忆全流程 |

**2×3 布局**（每个子图左上角标注相机名，便于 VLM 识别视角）：

```
上排：CAM_FRONT_LEFT | CAM_FRONT | CAM_FRONT_RIGHT
下排：CAM_BACK_LEFT  | CAM_BACK  | CAM_BACK_RIGHT
```

**Oracle 感知对象**（`perception.oracle_objects: true` 时开启）：基于 nuScenes **GT 标注
（`sample_annotation`）投影**到 6 相机，为每帧生成结构化 `perception_objects`（检测框/类别/
位置/速度/加速度），喂给决策 VLM 并写入 jsonl。⚠️ 这些是 **nuScenes 真值标注投影**，
**不是检测模型预测**——用于研究/评测下为决策提供准确感知先验，每个对象 `is_oracle=true`。
速度/加速度严格满足**在线因果性**（仅用当前+历史帧，缺历史置空标记不可用，禁止假值）。

运行（**务必用 `--output` 指定独立 jsonl，避免覆盖 `default` 基线**）：
```bash
# 六视角环视 + oracle（先在 config 设 perception.mode=surround_mosaic, oracle_objects=true）
python scripts/07_run_full_demo.py --mode memory_on --max-scenes 1 --max-frames 1 \
  --output outputs/decisions_memory_on_mosaic_test.jsonl
```

每帧 jsonl 新增字段：`perception_mode`、`perception_objects`；新产物目录 `outputs/mosaic/`。

---

## 7. 项目结构

```
vla_memory_demo/
├── README.md                           # 本文件
├── LICENSE                             # MIT 许可证
├── requirements.txt                    # Python 依赖
├── environment.yml                     # Conda 环境定义
├── pyproject.toml                      # 项目构建配置
│
├── config/                             # YAML 配置（启动时自动深合并）
│   ├── default.yaml                    # 项目级默认
│   ├── api_models.yaml                 # VLM provider + system prompt
│   ├── prompts.yaml                    # 【R1+ 新增】所有 VLM 提示词模板
│   ├── data_nuscenes.yaml              # nuScenes + CAN bus 配置
│   ├── memory.yaml                     # 三层记忆配置
│   ├── decision.yaml                   # 决策约束 + 图片上下文配置
│   └── evaluation.yaml                 # 评测指标 + L2 horizon 配置
│
├── data/
│   ├── README.md
│   ├── nuscenes/
│   │   ├── raw/                        # 用户手动放置：nuScenes v1.0-mini + can_bus
│   │   └── processed/                  # 预处理产物 (index.jsonl 等)
│   └── knowledge/                      # 长期记忆 YAML 知识库
│       ├── long_term_rules.yaml        # 17 条驾驶规则
│       ├── driving_strategies.yaml     # 13 条驾驶策略
│       └── knowledge_graph/            # 预留：知识图谱
│
├── scripts/                            # 入口脚本（全部支持 --help）
│   ├── 00_check_environment.py         # 环境自检
│   ├── 00_prepare_models.py            # 下载 DINOv2 权重
│   ├── 01_prepare_nuscenes_index.py    # 构建数据集索引
│   ├── 02_extract_keyframes.py         # 1Hz 关键帧抽样
│   ├── 03_run_scene_understanding.py   # 独立场景理解（调试用）
│   ├── 06_run_evaluation.py            # 评测 (--decisions / --compare)
│   └── 07_run_full_demo.py             # 【主入口】OnlineDrivingLoop
│
├── src/vla_memory/                     # 核心包（src layout）
│   ├── __init__.py
│   │
│   ├── common/                         # 通用工具
│   │   ├── config.py                   # YAML 配置加载 + 深合并
│   │   ├── logging_utils.py            # 中文日志 + 文件 handler
│   │   ├── path_utils.py
│   │   ├── json_utils.py               # 含 extract_json_from_text (VLM 输出解析)
│   │   ├── image_io.py
│   │   ├── prompt_loader.py            # 【R1】prompts.yaml 加载器
│   │   └── decision_record_io.py       # 【R1】jsonl append + resume 扫描
│   │
│   ├── schemas/                        # Pydantic v2 数据模型
│   │   ├── frame.py                    # FrameMeta
│   │   ├── ego_state.py                # EgoState（含 CAN bus 字段）
│   │   ├── trajectory.py               # Trajectory + TrajectoryPoint
│   │   ├── scene.py                    # SceneUnderstandingResult + LaneInfo/VehicleInfo/...
│   │   ├── memory.py                   # ShortTermMemoryItem + MemoryRecord + LongTermRule
│   │   ├── decision.py                 # DecisionOutput + VALID_BEHAVIORS
│   │   └── evaluation.py               # EvalSampleResult + EvalSummary (含 L2 字段)
│   │
│   ├── data/                           # 数据适配器
│   │   ├── base_dataset.py             # 抽象基类
│   │   ├── nuscenes_adapter.py         # nuscenes-devkit 封装
│   │   ├── can_bus_loader.py           # 【R3】CAN bus pose.json + vehicle_monitor.json
│   │   ├── ego_state_builder.py        # 【R3】build()/from_can_bus/from_poses
│   │   ├── trajectory_builder.py       # 历史 + 未来 ego-centric 轨迹
│   │   └── route_infer.py              # 伪导航语义推断
│   │
│   ├── keyframes/                      # 关键帧抽样
│   │   ├── base.py
│   │   ├── periodic_sampler.py         # 按 step 抽样
│   │   └── nuscenes_keyframe_sampler.py
│   │
│   ├── perception/                     # 感知（DINOv2 + VLM 场景理解）
│   │   ├── image_feature_extractor.py  # 抽象基类
│   │   ├── dinov2_extractor.py         # facebook/dinov2-base
│   │   ├── vlm_client.py               # 抽象基类
│   │   ├── openai_compatible_client.py # OpenAI 兼容 API 客户端
│   │   └── scene_understanding.py      # DINOv2 + VLM 组合 pipeline
│   │
│   ├── memory/                         # 三层记忆
│   │   ├── short_term_memory.py        # deque 滑窗 (capacity=10)
│   │   ├── mid_term_memory.py          # FAISS 6 路融合 + 可选持久化
│   │   ├── long_term_memory.py         # 【P2】strict_scene_match
│   │   ├── retrieval.py                # MemoryRetriever (三层统一入口)
│   │   ├── vector_store.py             # 抽象基类
│   │   ├── faiss_store.py              # FAISS IndexFlatIP
│   │   └── memory_record_io.py         # JSON / pickle 序列化
│   │
│   ├── decision/                       # 决策
│   │   ├── prompt_builder.py           # 用 prompts.yaml 渲染决策 prompt
│   │   ├── decision_client.py          # 调 VLM + 日志 [DECISION]
│   │   ├── output_parser.py            # 解析 + 校验（waypoint 数量等）
│   │   ├── rule_fallback.py            # parse 失败兜底
│   │   ├── config_access.py            # 【P1】统一访问 decision.yaml
│   │   └── dynamics_adapter.py         # 旧版动力学 stub（保留）
│   │
│   ├── evaluation/                     # 评测
│   │   ├── metrics.py                  # ADE / FDE / L2@horizon / 轨迹有效性
│   │   ├── evaluator.py                # Evaluator (修复行为准确率 bug)
│   │   └── report_writer.py            # CSV / JSONL / Markdown 报告
│   │
│   ├── pipeline/                       # 流水线编排
│   │   ├── prepare_nuscenes.py         # 数据加载 + 关键帧抽样
│   │   ├── online_loop.py              # 【R1 核心】OnlineDrivingLoop
│   │   ├── full_demo_pipeline.py       # 薄壳：prepare → enrich → loop
│   │   └── eval_pipeline.py            # 【R3】单 mode + 对比 双入口
│   │
│   └── planning/                       # 【R4 新增】预留接口
│       ├── __init__.py
│       ├── dynamics_planner.py         # 轨迹 → 控制量（CARLA 闭环预留）
│       └── trajectory_sampler.py       # 多模态轨迹选择（扩散/AR 预留）
│
├── carla_bridge/                       # 【v1.0 新增】CARLA 0.9.15 闭环集成（不改 src/）
│   ├── README.md                       #   CARLA 集成完整文档（框架/模块/指令/排查）
│   ├── setup_env.sh                    #   一键建 mulmem_carla(Python 3.9) 环境
│   ├── closed_loop.py                  #   主驱动：3s 认知 + 10Hz 回控 + raw 捕获 + 视频
│   ├── run_carla_demo.py               #   入口（python -m carla_bridge.run_carla_demo）
│   ├── config/{carla.yaml, scenarios/} #   连接/相机/控制器/视频 + 场景 YAML
│   ├── env/                            #   连接+同步+场景(ego/交通流/天气/路由)+spectator
│   ├── sensors/                        #   6 相机 mosaic + GT 感知对象
│   ├── state/                          #   坐标变换(手性) + 自车状态 + 历史位姿
│   ├── control/                        #   Pure Pursuit(横向) + PID(纵向)
│   ├── memory_adapter/                 #   kf dict 构建 + 封装 OnlineDrivingLoop
│   ├── metrics/                        #   碰撞/违规/路线/舒适度 + 报告
│   └── video/                          #   驾驶视频录制（跳过 VLM 冻结期）
│
├── tests/                              # 28 个测试模块（pytest）
│   ├── test_can_bus_loader.py
│   ├── test_phase_refactor_online_loop.py
│   ├── test_phase{2,4,5,6,7}_*.py
│   └── ... (其他模块单元测试)
│
├── docs/                               # 详细设计文档
│   ├── architecture.md                 # 完整架构 + 时序图
│   ├── prompts.md                      # 提示词编辑指南
│   ├── memory_design.md
│   ├── data_flow.md
│   ├── evaluation.md
│   ├── api_config.md
│   ├── future_work.md
│   └── perception_upgrade.md           # 【新增】六视角环视 + Oracle 感知特性
│
└── outputs/                            # 运行产物（gitignored）
    ├── decisions_<mode>_<run_id>.jsonl # 决策结果（每帧 append）
    ├── features/                       # DINOv2 特征 .npy
    ├── memory_db/                      # 中期记忆持久化（可选）
    ├── mosaic/                         # 【新增】surround_mosaic 拼接图（六视角模式）
    ├── logs/                           # 各模块日志 + 单帧审计 dump
    └── reports/                        # 评测报告 (CSV / JSONL / MD)
```

---

## 8. 核心模块（src/vla_memory/）

按子目录分组，列出关键类/函数。

### 8.1 `common/` — 通用工具

| 名称 | 文件 | 关键 API | 用途 |
|---|---|---|---|
| `Config` / `load_config` | `config.py` | `.get_nested(*keys)`, `.ensure_output_dirs()` | YAML 自动深合并 |
| `get_prompt_loader()` | `prompt_loader.py` | `.render(key, **vars)`, `.get(key)`, `.get_list(key)` | prompts.yaml 加载 + str.format_map 渲染 |
| `append_decision_record` / `load_processed_sample_tokens` | `decision_record_io.py` | — | jsonl append + fsync + resume 扫描 |
| `extract_json_from_text` | `json_utils.py` | — | 从 VLM 自由文本中抠出 JSON |
| `get_logger` | `logging_utils.py` | — | 双写控制台 + 文件的中文日志 |

### 8.2 `schemas/` — Pydantic 数据模型

| 模型 | 文件 | 关键字段 |
|---|---|---|
| `FrameMeta` | `frame.py` | `sample_token`, `image_path`, `timestamp`, `scene_token` |
| `EgoState` | `ego_state.py` | `x/y/z, yaw, speed, acceleration` + CAN bus 字段 `yaw_rate / steering_angle / throttle / brake / gear / source` |
| `Trajectory` | `trajectory.py` | `points: List[TrajectoryPoint(t,x,y,optional_v)]` |
| `SceneUnderstandingResult` | `scene.py` | `scene_id / weather_id / lanes / vehicles / pedestrians / traffic_lights / intersections / risk_factors` |
| `PerceptionObject` | `perception.py` | oracle 感知对象：`category / position_ego / distance_to_ego / boxes_2d / velocity / acceleration / kinematics_source / is_oracle` |
| `ShortTermMemoryItem` | `memory.py` | 11 字段（含 image_path / scene_understanding_result） |
| `MemoryRecord` (= `MidTermMemoryRecord`) | `memory.py` | 12 字段（含 behavior / decision_reason / trajectory） |
| `LongTermRule` | `memory.py` | `rule_id / scene_id / weather_id / title / content / priority` |
| `DecisionOutput` | `decision.py` | `behavior / behavior_reason / target_speed / risk_level / trajectory / safety_notes` |
| `EvalSampleResult` / `EvalSummary` | `evaluation.py` | 含 `l2_per_horizon` / `l2_mean_per_horizon` |

### 8.3 `data/` — 数据适配器

| 类 | 文件 | 关键方法 | 用途 |
|---|---|---|---|
| `BaseDrivingDataset` | `base_dataset.py` | 抽象接口 | 数据源抽象 |
| `NuScenesAdapter` | `nuscenes_adapter.py` | `load()`, `iter_frames()`, `get_frame_image_paths()`, `get_ego_pose()`, `get_future_ego_trajectory()`, `get_perception_objects()` | nuScenes 数据访问（多相机 + oracle） |
| `CanBusLoader` | `can_bus_loader.py` | `load_scene()`, `query_at(scene, utime, tolerance_us)` | CAN bus 真值加载，含单位换算 |
| `EgoStateBuilder` | `ego_state_builder.py` | `build()` (dispatcher), `build_from_can_bus()`, `build_from_poses()` | 自车状态构造（CAN bus 优先，差分回退） |
| `TrajectoryBuilder` | `trajectory_builder.py` | `build_history_trajectory()`, `build_future_trajectory()` | ego-centric 历史 / 未来轨迹 |
| `RouteInfer` | `route_infer.py` | `infer(future_poses, current_speed)` | 伪导航语义（straight / left_turn / 等） |
| `get_perception_objects()` | `oracle_perception.py` | 6 相机 GT 投影 + 因果运动学 | 生成 oracle `perception_objects`（nuScenes 真值，非模型预测） |

### 8.4 `perception/` — 感知

| 类 | 文件 | 关键方法 | 用途 |
|---|---|---|---|
| `ImageFeatureExtractor` | `image_feature_extractor.py` | 抽象 | 图像特征抽象 |
| `DINOv2Extractor` | `dinov2_extractor.py` | `load_model()`, `extract(image_path)`, `save_feature()` | DINOv2 768-d 特征 + L2 归一 |
| `VLMClient` | `vlm_client.py` | 抽象 | VLM 抽象 |
| `OpenAICompatibleVLMClient` | `openai_compatible_client.py` | `understand_scene()`, `decide(prompt, image_paths)` | OpenAI 兼容 API（base64 图像编码） |
| `SceneUnderstandingPipeline` | `scene_understanding.py` | `process_frame(sample_token, image_path, image_layout)` | DINOv2 + VLM 组合，输出结构化 JSON |
| `build_surround_mosaic` | `surround_mosaic.py` | `build_surround_mosaic(image_paths, cameras, ...)` | 六视角 2×3 环视拼接（surround_mosaic 模式替代前视图） |

### 8.5 `memory/` — 三层记忆

| 类 | 文件 | 关键方法 | 用途 |
|---|---|---|---|
| `ShortTermMemory` | `short_term_memory.py` | `add()`, `get_recent_image_paths(n)`, `generate_summary()` | deque 滑窗 + 文本摘要 |
| `MidTermMemory` | `mid_term_memory.py` | `add_record(record, feature)`, `search(...)`, `save_full()`, `close()` | FAISS 6 路融合 + 可选持久化 |
| `LongTermMemory` | `long_term_memory.py` | `load()`, `search_rules(scene_id, weather_id)`, `format_rules_text()` | YAML 规则严格 scene_id 匹配 |
| `MemoryRetriever` | `retrieval.py` | `retrieve(query_feature, scene_id, ..., use_short/mid/long_term)` | 三层统一入口 |
| `FAISSVectorStore` | `faiss_store.py` | `add(vectors, ids)`, `search(query, top_k)`, `save()/load()` | IndexFlatIP 封装 |
| `MemoryAdmissionController` | `admission.py` | `decide(ctx) -> AdmissionResult` | Phase 2 价值门控（6 信号 + 17 高价值事件 + 3 低价值过滤） |
| `MemoryValueScorer` | `value_scorer.py` | `score(record)` | Phase 3 存量价值评分（admission+事件+近期性+检索效用+冗余+置信+冲突） |
| `MemoryEvictionManager` / `MemoryCompactionManager` | `eviction.py` | `evict()`, `rebuild_index(reconstruct_n)` | Phase 3 容量淘汰（soft delete）+ FAISS 物理压缩 |
| `EventMemoryManager` | `event_memory.py` | `on_frame_admitted()`, `finalize_event()` | Phase 5 连续高价值帧→事件级记忆（关键帧+摘要） |
| `MemoryUpdateManager` | `update.py` | `process(new, candidates)` | Phase 6 冲突感知软更新（5 类冲突+版本链+unsafe 保护） |
| `MemoryConsolidationManager` | `consolidation.py` | `consolidate(records)` | Phase 7 离线沉淀长期记忆候选（pending_review，不覆盖正式库） |

### 8.6 `decision/` — 决策

| 类 | 文件 | 关键方法 | 用途 |
|---|---|---|---|
| `DecisionPromptBuilder` | `prompt_builder.py` | `build(scene, ego, ..., short_term_summary, mid_term_memories, long_term_rules_text)` | 用 prompts.yaml 渲染决策 prompt |
| `DecisionClient` | `decision_client.py` | `decide(image_paths, scene_understanding, ...)` | 调 VLM + 日志 `[DECISION]` |
| `parse_decision_output` | `output_parser.py` | `→ (parsed, errors)` | 校验 behavior 枚举 + waypoint 数量 + JSON 格式 |
| `generate_fallback_decision` | `rule_fallback.py` | → DecisionOutput | parse 失败兜底 |
| `get_waypoint_bounds` / `get_valid_behaviors` | `config_access.py` | — | 统一访问 decision.yaml |

### 8.7 `evaluation/` — 评测

| 类 / 函数 | 文件 | 关键 API |
|---|---|---|
| `compute_ade` / `compute_fde` / `is_valid_trajectory` / `resample_trajectory` | `metrics.py` | — |
| `compute_l2_at_horizon(pred, gt, horizon_s)` / `compute_l2_per_horizon(...)` | `metrics.py` | 【P6】L2@1s/2s/3s |
| `Evaluator` | `evaluator.py` | `evaluate_sample()`, `aggregate_results()`, `from_config()` |
| `ReportWriter` | `report_writer.py` | `write_csv()`, `write_jsonl()`, `write_markdown()` |

### 8.8 `pipeline/` — 流水线编排

| 函数 / 类 | 文件 | 关键 API |
|---|---|---|
| `run_prepare_nuscenes(config)` | `prepare_nuscenes.py` | → `{adapter, keyframe_index, total_keyframes}` |
| `enrich_keyframes_with_state(adapter, keyframe_index, config)` | `full_demo_pipeline.py` | per-frame 数据准备（ego_state / history / nav）；surround_mosaic 模式下在此拼 mosaic 落盘并改写 image_path；oracle_objects 在此注入 keyframe |
| `OnlineDrivingLoop` | `online_loop.py` | `setup()`, `step(kf)`, `run(keyframes)`, `close()` |
| `run_full_demo(config, mode, resume)` | `full_demo_pipeline.py` | 薄壳：prepare → enrich → OnlineDrivingLoop |
| `run_eval_pipeline(results, mode, ...)` | `eval_pipeline.py` | 单 mode 评测 |
| `run_eval_compare(results_by_mode, ...)` | `eval_pipeline.py` | 多 mode 对比 |

### 8.9 `planning/` — 预留接口（v0.1 不实现）

| 类 | 文件 | 用途 | 当前行为 |
|---|---|---|---|
| `DynamicsPlanner` | `dynamics_planner.py` | 轨迹 → 低层控制量（CARLA 闭环用） | `.plan()` 抛 NotImplementedError + 详细指引 |
| `TrajectorySampler` | `trajectory_sampler.py` | N 条候选 → 最优轨迹（扩散/AR 模型用） | `.select()` 抛 NotImplementedError + 详细指引 |

---

## 9. 分层记忆系统设计

> ⚠️ **第一版必须使用 FAISS**，中期记忆不允许降级到 numpy。FAISS 未安装时脚本会 hard fail。

三层记忆遵循同一条贯穿原则：**在 `OnlineDrivingLoop.step()` 内"先读后写"**——
第 *i* 帧的检索发生在 `mid_term.add_record` 和 `short_term.add` **之前**，
保证当前帧检索时只能看到 [0, i-1] 帧的历史，**彻底消除 data leakage**。

### 9.1 短期记忆 (ShortTermMemory)

#### 存储

* **容器**：`collections.deque(maxlen=capacity)`，纯内存
* **容量**：默认 10 帧（`config/memory.yaml -> short_term.capacity`），FIFO 自动淘汰
* **不持久化**：进程退出即丢失

#### 每条记录的 11 个字段（`ShortTermMemoryItem`）

| 字段 | 类型 | 内容 |
|---|---|---|
| `frame_id` | str | sample_token |
| `timestamp` | int | 微秒级时间戳 |
| `image_path` | str | **图像绝对路径**（surround_mosaic 模式下为拼接图 mosaic 路径；不存字节，节省内存） |
| `image_feature_path` | str | `.npy` 文件路径 |
| `scene_description` | str | VLM 自由文本描述 |
| `scene_id` / `weather_id` | str | 场景 / 天气枚举 |
| `nav_instruction` | str | 导航语义 |
| `ego_state` | dict | 完整自车状态（含 CAN bus 字段） |
| `history_trajectory` | list | 该帧的历史轨迹 |
| `scene_understanding_result` | dict | 场景理解完整结构化 JSON |

#### 更新时机

仅在 `step()` 末尾、决策完成之后 push 1 次（仅 `memory_on` 模式）。memory_off 不更新。

#### 输入给决策模型的两类内容

1. **图像列表**：`get_recent_image_paths(N-1)` 取最近 N-1 张图，base64 后塞 `image_url` block
2. **文本摘要**：`generate_summary()` 渲染 `## 短期记忆` 段（5 字段：scene_id / weather_id / nav / speed / description）

### 9.2 中期记忆 (MidTermMemory)

#### 存储：3 个并行容器（轻量级文件数据库）

```
MidTermMemory（内存）
    ├── _records         : Dict[record_id → MemoryRecord]   ← 元数据
    ├── _text_corpus     : Dict[record_id → scene_text]     ← 文本检索加速
    └── faiss_store      : FAISSVectorStore (IndexFlatIP)   ← 视觉向量
                              ├── _index : C++ 二进制 (768-d × N)
                              └── _ids   : List[record_id]
```

三者通过 `record_id`（= sample_token）关联。

#### 每条记录的 12 个字段（存「决策后的完整经验」）

`record_id` / `scene_id` / `weather_id` / `nav_instruction` / `ego_state` / `scene_text` /
`history_trajectory` / `image_feature_path` / **`decision_reason`** / **`behavior`** / **`trajectory`** / `frame_meta`。

> ✅ **价值门控事件记忆系统（阶段 1-7）已端到端验证**：memory_on 烟雾测试中，连续高价值帧正确
> 合并为 `event_memory`（含 start/peak/end 关键帧 + ego/perception/decision 摘要），准入门控识别
> `intersection` / `ghost_probing_risk` 等高价值事件（score 0.6-0.8，`admission_policy_version=value_gated_v0.1`），
> 阶段 1-7 流程全跑通；阶段 3 淘汰/6 冲突/7 沉淀因数据量小未触发（属预期，需更多场景）。
> 快速验证（需 `DASHSCOPE_API_KEY`）：
> ```bash
> python scripts/07_run_full_demo.py --mode memory_on --max-scenes 1 --max-frames 5 --output outputs/decisions_smoke.jsonl
> ```

#### Phase 1 metadata 扩展（schema v0.2，为价值门控做准备）

每条记录在上述 12 字段基础上新增 **8 类共 37 个 metadata 字段**（基础状态 / 来源 / 视觉输入 /
场景标签 / 写入价值 / 记忆价值 / 使用统计 / 更新与删除），全部带默认值，**旧 `mid_term_meta.json`
可正常加载**（缺字段自动补默认）。Phase 1 不改写入触发逻辑（memory_on 仍逐帧全存），价值类字段
保持 `None` 绝不伪造，准入标记为 `legacy`。每帧 jsonl 同步新增 `mid_term_memory_added` /
`mid_term_memory_id` / `memory_admission_score` / `memory_admission_reasons` /
`memory_record_status` 5 个字段。字段清单与配置见 [`docs/memory_design.md §3.7`](docs/memory_design.md)，
价值门控改造路线见 [`docs/mid_term_memory_value_gating_plan.md`](docs/mid_term_memory_value_gating_plan.md)。

#### Phase 2 价值门控写入（MemoryAdmissionController）

中期记忆不再逐帧全存：每帧决策后、写入前由 `MemoryAdmissionController` 判断是否入库。低价值帧
（普通巡航 / 稳定停车 / 冗余帧）拒绝，高价值事件（变道 / 起步 / 急停 / 避障 / 路口 / cut-in / 行人交互 /
长尾等 17 类）强制写入。综合价值分由 6 信号加权得到（dynamics_surprise / scene_salience /
perception_change / decision_change / memory_novelty / posthoc_outcome_value）。`enabled=false` 时
退化为逐帧全存（回归基线）。先读后写与 memory_on/off 公平性不变（门控只在 memory_on 写入路径生效，
短期记忆不受门控）。每帧 jsonl 记录 `memory_admission_should_store` / `memory_admission_reasons` /
`memory_admission_reject_reasons` / `memory_event_type` 等。设计与配置见
[`docs/stage2_admission_design.md`](docs/stage2_admission_design.md) 与
[`docs/memory_design.md §3.8`](docs/memory_design.md)。

**写入率统计**（开启门控后应显著 < 100%）：
```bash
# 记录数 / 帧数
python -c "import json; m=json.load(open('outputs/memory_db/mid_term_meta.json',encoding='utf-8')); print('memories:', len(m))"
wc -l outputs/decisions_memory_on_*.jsonl | awk '{print "frames:", $1}'
```

#### Phase 3 容量上限与价值淘汰（MemoryEvictionManager）

中期记忆设容量上限（`capacity.max_records` / `max_disk_mb`）；接近上限时按 `memory_value_score` 淘汰
低价值记忆（soft delete：`is_active=False` / `status=deleted` / `deleted_reason`，元数据保留），高价值/长尾/
高风险记忆受保护（`protect_*` + `min_keep_per_event_type`）。检索过滤 inactive；inactive 比例过高时 rebuild
FAISS 物理压缩（`IndexFlatIP` 用 `reconstruct_n` 重建）。存量价值由 `MemoryValueScorer` 综合 admission/事件/
近期性/检索效用/冗余/置信/冲突计算。`capacity.enabled=false` 时无容量上限（回归阶段 2）。先读后写与
memory_on/off 公平性不变（淘汰只在 `add_record` 写入路径触发）。设计与配置见
[`docs/stage3_eviction_design.md`](docs/stage3_eviction_design.md) 与 [`docs/memory_design.md §3.9`](docs/memory_design.md)。

**小规模测试**（人为设小上限验证淘汰）：
```bash
# 编辑 config/memory.yaml: mid_term.capacity.max_records: 5, eviction_trigger_ratio: 0.8, eviction_target_ratio: 0.7
python scripts/07_run_full_demo.py --mode memory_on --max-frames 20
# 检查淘汰：inactive 记录数 > 0，active 记录数 ≤ 5×0.7≈3，高价值事件未被淘汰
python -c "import json; d=json.load(open('outputs/memory_db/mid_term_meta.json',encoding='utf-8')); \
a=[r for r in d.values() if r.get('is_active',True)]; i=[r for r in d.values() if not r.get('is_active',True)]; \
print('active:',len(a),'inactive(soft-deleted):',len(i)); \
print('evicted reasons:', set(r.get('deleted_reason') for r in i))"
```

#### Phase 4 价值感知检索重排与多样性（检索增强）

中期记忆检索在原 6 路相似度基础上升级：过滤 inactive/deleted/deprecated/低置信 → 取候选池 top-N →
价值重排（`value_aware_score = 0.8·相似度 + 0.2·memory_value_score`）→ 多样性约束（同 event_type /
同 scene_token 限流 + 近重复抑制）→ top-K → 更新命中统计。inactive 始终不返回。每帧 jsonl 记录
`retrieval_candidate_count` / `retrieval_active_candidate_count` / `retrieved_memory_value_scores` 等。
`enable_value_rerank=false` 退化为仅相似度。设计与配置见 [`docs/memory_design.md §3.10`](docs/memory_design.md)。

#### Phase 5 事件级记忆（EventMemory）

中期记忆从帧级升级为事件级：连续高价值帧合并为一个 `event_memory`（只存 start/peak/end 关键帧 +
结构化摘要 `ego/perception/decision/admission_summary`），显著降低 memory_db 规模。事件结束条件：
高价值信号消失 / 达最大长度 / scene 切换 / run 结束。检索优先返回 event_memory（`prefer_event_memory` 加成）。
`event_memory.enabled=false` 退化为阶段 2 逐帧 frame_memory；旧 frame_memory 记录仍兼容（可混存）。
事件 `event_id`/关键帧/摘要见 `outputs/memory_db/mid_term_meta.json`（`memory_type=event_memory`）。
设计与配置见 [`docs/stage5_event_memory_design.md`](docs/stage5_event_memory_design.md) 与
[`docs/memory_design.md §3.11`](docs/memory_design.md)。

**查看事件级记忆**：
```bash
python -c "import json; d=json.load(open('outputs/memory_db/mid_term_meta.json',encoding='utf-8')); \
ev=[r for r in d.values() if r.get('memory_type')=='event_memory']; fm=[r for r in d.values() if r.get('memory_type','frame_memory')=='frame_memory']; \
print('event_memory:',len(ev),'frame_memory:',len(fm)); \
[print(' ',e.get('event_id'),'|',e.get('event_type'),'|',e.get('admission_summary')) for e in ev[:5]]"
```

#### Phase 6 冲突感知更新（MemoryUpdateManager）

中期记忆支持"改"：检索到相似记忆但当前决策/安全评价冲突时，对旧记忆软更新（降权 / 标记
deprecated/superseded / 增 conflict_count / 版本化），新记忆作为新版本或替代。冲突分类：policy_conflict
（跨类别策略不同）、style_conflict（同类别风格不同，两条都保留）、context_mismatch（情境不同，不冲突）、
unsafe_old_memory（旧涉险）、unsafe_new_evidence（新不安全，**不覆盖旧**）。**不物理删除**；
unsafe 新证据绝不覆盖安全旧记忆。更新记录于旧记忆 `update_history`（`mid_term_meta.json` 可查）+ 日志。
`update.enabled=false` 退化为阶段 5 行为。设计与配置见
[`docs/stage6_update_design.md`](docs/stage6_update_design.md) 与 [`docs/memory_design.md §3.12`](docs/memory_design.md)。

**查看冲突更新**：
```bash
python -c "import json; d=json.load(open('outputs/memory_db/mid_term_meta.json',encoding='utf-8')); \
upd=[r for r in d.values() if r.get('update_history')]; \
print('memories with update_history:', len(upd)); \
[print(' ',r.get('memory_id'),'| status=',r.get('status'),'| conflicts=',r.get('conflict_count'),'| history=',len(r.get('update_history',[]))) for r in upd[:5]]"
```

#### Phase 7 中期记忆沉淀为长期记忆候选（MemoryConsolidationManager）

离线从中期记忆库挖掘高价值、稳定、可泛化经验，总结为长期记忆**候选**规则（`status=pending_review`），
写入 `outputs/long_term_candidates/candidate_rules.yaml`，**不自动覆盖**正式长期记忆库
`data/knowledge/long_term_rules.yaml`（需人工审核晋升）。按 `(event_type, risk_tags)` 分组，找多次出现
（≥ `min_evidence_count`）且价值高的模式，生成 safety/strategy/style 三类候选，带 evidence memory_ids、
confidence、safety_guard。危险偏好（高风险变道）剔除；风格候选不得覆盖安全规则。设计与配置见
[`docs/memory_design.md §3.13`](docs/memory_design.md)。

**生成候选长期记忆**（跑完 memory_on demo 后）：
```bash
python scripts/08_consolidate_long_term_candidates.py
# 查看候选
cat outputs/long_term_candidates/candidate_rules.yaml
```

#### 持久化：3 个文件（yaml 开关控制）

| 文件 | 格式 |
|---|---|
| `outputs/memory_db/mid_term_faiss.index` | FAISS 原生二进制 |
| `outputs/memory_db/mid_term_faiss.ids.json` | JSON 数组 |
| `outputs/memory_db/mid_term_meta.json` | JSON 字典 |

#### 数据库形态：可跨次会话累积

```yaml
# config/memory.yaml
mid_term:
  persistence:
    enabled: false              # 默认关
    save_on_close: true         # close() 时全量写
    auto_load_on_init: true     # setup() 时自动加载
    strict_load: false
```

开启后跑多次 demo：

```
第 1 次：    setup load → 索引 0 → step×5 add → close save → 磁盘 5 条
第 2 次：    setup load → 索引 5 → step×5 add → close save → 磁盘 10 条
第 N 次：    库无界增长（受磁盘 / RAM 限制）
```

#### 6 路加权融合检索

```
final_score = 0.40 × visual_score    # FAISS 余弦相似度
            + 0.15 × text_score      # Jaccard 词汇重叠
            + 0.15 × scene_score     # scene_id 精确匹配 (1/0)
            + 0.05 × weather_score   # weather_id 精确匹配
            + 0.15 × nav_score       # nav 精确匹配
            + 0.10 × state_score     # 自车状态相似度
                                     # = 0.4×speed_sim + 0.3×acc_sim + 0.3×yaw_sim
```

权重在 `config/memory.yaml -> mid_term.weights`。返回 top-3，渲染到 prompt 只取 4 字段
（scene_id / weather_id / decision_reason / behavior）。

### 9.3 长期记忆 (LongTermMemory)

* **完全静态**：YAML 加载，运行期不变
* **不持久化输出**：输入本身就是磁盘文件
* **内容**：
  - `data/knowledge/long_term_rules.yaml` — 17 条规则
  - `data/knowledge/driving_strategies.yaml` — 13 条策略
* **严格 scene 匹配（P2 默认行为）**：

```yaml
long_term:
  strict_scene_match: true       # scene_id="all" 通配规则被屏蔽
  strict_weather_match: false    # weather="all" 通配仍生效
```

排序：先按匹配分数降序，再按 priority 升序，取 top-5（`top_k`）。

### 9.4 统一检索入口 (MemoryRetriever)

```python
result = retriever.retrieve(
    query_feature=np.array,        # DINOv2 特征
    scene_text="...",
    scene_id="straight_road",
    weather_id="sunny",
    nav_instruction="straight",
    ego_state={"speed": 10.0, ...},
    use_short_term=True,           # memory_off 时全部 False
    use_mid_term=True,
    use_long_term=True,
)
# result["short_term_summary"]   -- str，文本摘要
# result["mid_term_results"]     -- List[dict]，每个含 record/final_score/sub_scores
# result["long_term_rules"]      -- List[LongTermRule]
```

### 9.5 单帧审计日志

每帧 `step()` 末尾自动写一段「单帧完整审计 dump」到 `outputs/logs/online_loop_<ts>.log`，
含本帧用到的图片路径、自车状态、检索到的三层记忆、决策模型输入输出等所有上下文。

```bash
# 查任意帧的完整决策链路
grep -A 60 "AUDIT frame=<sample_token>" outputs/logs/online_loop_*.log
```

---

## 10. VLM 输入输出全景

项目调用 **2 个独立 VLM**（默认都是 Qwen-VL-Max，可在 [`config/api_models.yaml`](config/api_models.yaml) 切换）：

| 模型 | 何时调用 | 输入 | 输出 |
|---|---|---|---|
| **场景理解 VLM** | 每帧 `step` 开头（无状态） | 1 张当前帧图 + 静态 prompt | 结构化 JSON |
| **决策 VLM** | 每帧 `step` 中段（含上下文） | N 张图 + 动态 prompt（三层记忆） | 行为 + 推理 + 轨迹 |

### 10.1 场景理解 VLM 输入

| 输入 | 来源 | 格式 |
|---|---|---|
| 图像（**1 张当前帧**） | `kf["image_path"]`（surround_mosaic 模式下为 `outputs/mosaic/<token>.jpg` 拼接图） | base64 编码后包成 `data:image/jpeg;base64,...` 嵌入 `image_url` block |
| user prompt | [`config/prompts.yaml -> scene_understanding.user`](config/prompts.yaml) | 静态文本模板（每帧都一样） |
| system prompt | [`config/api_models.yaml -> scene_understanding.system_prompt`](config/api_models.yaml) | "场景分析专家..." |

**不传**：自车状态、历史轨迹、导航、记忆、DINOv2 特征。这是有意的隔离：保证场景理解输出**只受图像本身**影响。

```json
[
  {"role": "system", "content": "<场景分析专家 system_prompt>"},
  {"role": "user",   "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "text",      "text": "<scene_understanding.user 静态模板>"}
  ]}
]
```

输出 JSON 由 [scene_understanding.py::_parse_and_validate](src/vla_memory/perception/scene_understanding.py) 解析，
含 `scene_id / weather_id / lanes[] / vehicles[] / pedestrians[] / traffic_lights[] / intersections{} / traffic_density / risk_factors[]`。

### 10.2 决策 VLM 输入

| # | 输入 | 来源 | prompt 位置 | memory_off 时 |
|---|---|---|---|---|
| 1 | **图像（1-N 张）** | 短期记忆滑窗最近 N-1 张 + 当前帧（mosaic 模式为拼接图） | `image_url` blocks | 仅当前帧 1 张 |
| 2 | **当前场景理解** | 本帧场景 VLM 的 JSON 输出 | `## 当前场景理解` 段 | 同左 |
| 3 | **自车状态** | `EgoStateBuilder.build()` 输出 | `## 自车状态` 段 | 同左 |
| 4 | **历史轨迹** | `TrajectoryBuilder` 过去 5s | `## 最近历史轨迹` 段 | 同左 |
| 5 | **导航指令** | `RouteInfer` 推断的伪标签 | `## 导航指令` 段 | 同左 |
| 6 | **短期记忆文本摘要** | `short_term.generate_summary()` | `## 短期记忆` 段 | **空（不出现）** |
| 7 | **中期记忆 top-3** | `mid_term.search()` 6 路融合 | `## 相似历史经验` 段（4 字段） | **空（不出现）** |
| 8 | **长期规则文本** | `long_term.search_rules()` 严格 scene 匹配 | `## 相关驾驶规则` 段 | **空（不出现）** |
| 9 | **system prompt** | [`config/api_models.yaml -> decision.system_prompt`](config/api_models.yaml) | 独立 `system` 消息 | 同左 |
| 10 | **图像布局说明** | `online_loop._build_image_layout_desc`（按 `perception.mode`） | `## 输入图像布局` 段 | 同左（mosaic 模式描述六视角 2×3 布局） |
| 11 | **Oracle 感知对象** | `oracle_perception.get_perception_objects`（nuScenes GT 投影） | `## Oracle 感知对象` 段 | 同左（仅 `oracle_objects=true` 时非空，`is_oracle=true`） |

**image_paths 张数控制**：

```yaml
# config/decision.yaml
vlm_inputs:
  image_context_size: 3              # 总张数（含当前帧）
  include_current_frame: true        # 当前帧拼在末尾
  max_images_per_call: 4             # Qwen-VL-Max 建议 ≤ 4
```

实际发送图像数（memory_on，`image_context_size=3`）：

| 帧序号 | memory_off | memory_on |
|---|---|---|
| 第 1 帧 | 1 | 1（窗口空） |
| 第 2 帧 | 1 | 2 |
| 第 N≥3 帧 | 1 | 3（cap） |

### 10.3 决策 VLM 输出格式

```json
{
  "behavior": "KEEP_LANE",
  "behavior_reason": "前方畅通，保持当前车道直行",
  "target_speed": 5.0,
  "risk_level": "low",
  "trajectory": [
    {"t": 0.1, "x": 0.5, "y": 0.0, "optional_v": 5.0},
    ...共 20-30 个点...
  ],
  "safety_notes": []
}
```

#### 11 种行为枚举

`KEEP_LANE`, `FOLLOW`, `SLOW_DOWN`, `STOP`, `TURN_LEFT`, `TURN_RIGHT`,
`CHANGE_LANE_LEFT`, `CHANGE_LANE_RIGHT`, `AVOID_OBSTACLE`, `YIELD`, `UNKNOWN`。

#### 校验规则

[`output_parser.parse_decision_output`](src/vla_memory/decision/output_parser.py)：
- `behavior` 必须在枚举内
- `risk_level` ∈ {low, medium, high}
- `trajectory` 长度 ∈ [`waypoint_min_num`, `waypoint_max_num`]（默认 20-30，从 `decision.yaml` 读取）
- 每个点必须含 `x`, `y`

校验失败走 fallback（[`rule_fallback.generate_fallback_decision`](src/vla_memory/decision/rule_fallback.py)）生成开环轨迹，
record 里 `fallback_used=True`，`parser_status="fallback"`。

### 10.4 图像 base64 编码细节

```
本地 .jpg (140 KB)
    │ base64.b64encode → str (~187 KB，膨胀 33%)
    ▼
"data:image/jpeg;base64,/9j/4AAQSk..."（Data URI 格式）
    │ HTTPS POST 嵌入 JSON 请求体
    ▼
API 服务端 base64.b64decode → JPEG 字节 → 解码 RGB 像素 → VLM 视觉编码器
```

* **base64 不是图像格式**，只是把二进制套一层 ASCII 马甲以便走 JSON
* VLM 服务端**收到后还原原始 JPEG 字节**，模型实际"看到"的就是 JPEG（保留分辨率/色彩）
* MIME 前缀 `image/jpeg` 告诉服务端用哪个解码器；换 `image/png` 即可换格式

### 10.5 轨迹坐标系

* `coordinate_system = "ego_centric"`：x 前向，y 左向，z 上向，单位米
* `t` 字段是相对当前帧的相对时间（秒），正值未来、负值过去
* 决策轨迹的 `t` 单调递增（如 0.1, 0.2, ..., 3.0）
* `optional_v` 是可选的瞬时速度（m/s），用于运动学规划

---

## 11. 数据建模

### 11.1 关键帧策略

| 项 | 默认值 | 来源 |
|---|---|---|
| 数据集帧率 | 12 Hz（nuScenes sensor data） | — |
| nuScenes keyframe 频率 | ~2 Hz | `data/nuscenes/raw/samples/` |
| 采样步长 | `step=2`（取 1/2） | `config/data_nuscenes.yaml -> keyframe.step` |
| 实际关键帧频率 | **~1 Hz** | `PeriodicSampler` 切片 `frames[::2]` |
| 实现 | [`NuScenesKeyframeSampler`](src/vla_memory/keyframes/nuscenes_keyframe_sampler.py) | 调用 `adapter.iter_frames()` 沿 `first_sample_token → next` 链严格按时间遍历 |

**结论**：单 scene 内关键帧严格按时间顺序连续；跨 scene 边界会有跳变（scene-A 最后一帧 → scene-B 第一帧）。

### 11.2 自车状态估计

R3 起优先使用 **CAN bus 真值**，失败回退到 ego_pose 差分：

```
EgoStateBuilder.build(current_pose, scene_name=..., can_bus_loader=..., prev_pose=..., prev_prev_pose=...)
    │
    ├─ if can_bus_loader and scene_name:
    │     try:
    │         build_from_can_bus(scene_name, ts, loader)   ← CAN bus 真值
    │         → source="can_bus"，含 yaw_rate/steering/throttle/brake
    │     except (KeyError, FileNotFoundError):
    │         warning (每 scene 1 次) → 回退
    │
    └─ build_from_poses(current, prev, prev_prev)          ← 差分回退
         → source="pose_diff"
```

**CAN bus 来源**：

* `data/nuscenes/raw/can_bus/scene-XXXX_pose.json` (~50 Hz)：`pos`, `orientation`, `vel`, `accel`, `rotation_rate`
* `data/nuscenes/raw/can_bus/scene-XXXX_vehicle_monitor.json` (~2 Hz)：`vehicle_speed (km/h)`, `steering (deg)`, `yaw_rate (deg/s)`, `throttle/brake (0-100)`, `gear_position`

[`CanBusLoader.query_at(scene, utime, tolerance_us=60000)`](src/vla_memory/data/can_bus_loader.py)
二分查找最近邻 + 单位换算到 SI：km/h→m/s, 度→弧度, 0-100→0-1。

**开关**：

```yaml
# config/data_nuscenes.yaml
ego_state:
  use_can_bus: true
can_bus:
  enabled: true
  root: "data/nuscenes/raw/can_bus"
  tolerance_us: 60000
  fallback_to_pose_diff: true
```

### 11.3 历史轨迹构建

[`TrajectoryBuilder.build_history_trajectory(current_pose, past_poses, history_seconds=5.0)`](src/vla_memory/data/trajectory_builder.py)：

* 取过去 N 秒（默认 5s，可配 `data_nuscenes.yaml -> history_seconds`）的所有 ego_pose
* 全局坐标 → ego-centric 坐标（绕 z 轴旋转 -current_yaw）
* `Trajectory.points` 含每个点 `(t, x, y)`，t 为相对当前帧的负值

### 11.4 导航语义伪标签

[`RouteInfer.infer(future_poses, current_speed)`](src/vla_memory/data/route_infer.py) 用未来轨迹的
航向变化、横向位移、速度推断**伪导航标签**：

| 标签 | 触发条件（默认阈值） |
|---|---|
| `slow_or_stop` | `current_speed < 1.0 m/s` |
| `left_turn` | `yaw_change > 0.15 rad` |
| `right_turn` | `yaw_change < -0.15 rad` |
| `lane_change_left` | `lateral > 2.0 m, yaw_change < 0.15` |
| `lane_change_right` | `lateral < -2.0 m, yaw_change < 0.15` |
| `straight` | `|yaw_change| < 0.05, |lateral| < 1.0` |
| `lane_follow` | 其他情况 |
| `unknown` | 信息不足 |

> ⚠️ 这是 **demo 级伪标签**，不是人工标注的真值。`enrich_keyframes_with_state` 用它注入到 `kf["nav_instruction"]`，
> 决策模型当作导航指令；评测时也作为行为准确率的"参考真值"，因此存在循环依赖，**不严谨**。
> 评测报告对此显式声明。

---

## 12. 评测

### 12.1 评测模式

| 模式 | 命令 |
|---|---|
| **单 mode** | `python scripts/06_run_evaluation.py --decisions <jsonl>` |
| **双 mode 对比** | `python scripts/06_run_evaluation.py --compare <on.jsonl> <off.jsonl>` |

mode 标签从文件名 `decisions_<mode>_<run_id>.jsonl` 自动推断；
未推断出可用 `--mode LABEL` 覆盖。

### 12.2 评测指标

| 指标 | 含义 | 来源 | 单位 |
|---|---|---|---|
| `ADE` | 全程平均位移误差 | 预测 vs 真值轨迹（重采样到 25 点） | 米 |
| `FDE` | 终点位移误差 | 同上 | 米 |
| `L2_1s` / `L2_2s` / `L2_3s` | 【P6】特定 horizon 时刻 L2 误差 | 重采样轨迹按 `dt = 3.0 / 24` 取对应索引 | 米 |
| `valid_trajectory_rate` | 通过有效性校验的样本占比 | waypoint 数量、单步位移 ≤ 5m、字段完整性 | % |
| `behavior_accuracy` | 行为预测正确率 | pred vs nav→behavior 映射后的 gt（大小写归一化） | % |
| `fallback_count` | 用了规则 fallback 的样本数 | 解析失败计数 | 个 |

L2 计算公式（[metrics.py](src/vla_memory/evaluation/metrics.py)）：

```python
def compute_l2_at_horizon(pred, gt, horizon_s, total_horizon_s=3.0):
    n = min(len(pred), len(gt))           # 预设 n=25
    dt = total_horizon_s / (n - 1)        # = 3.0 / 24 ≈ 0.125
    idx = round(horizon_s / dt)
    return euclidean(pred[idx], gt[idx])
```

L2 horizons 可配：

```yaml
# config/evaluation.yaml
l2_per_horizon:
  horizons_seconds: [1.0, 2.0, 3.0]
```

### 12.3 真值轨迹构建

[`build_ground_truth_from_nuscenes`](src/vla_memory/pipeline/eval_pipeline.py)：用
`NuScenesAdapter.get_future_ego_trajectory(sample_token, future_seconds=3.0)` 沿
`sample["next"]` 链向前取 ego_pose，转 ego-centric。

不允许伪造真值。无法构造时该样本跳过；全部失败时 hard fail。

### 12.4 行为准确率（P6 修复）

旧版本 bug：默认 `nav_to_behavior_map={}`，导致 `pred="KEEP_LANE"` 与 `gt="straight"` 永远不等 → 准确率恒为 0%。

P6 修复：[`Evaluator.__init__`](src/vla_memory/evaluation/evaluator.py) 默认使用内置 `_DEFAULT_NAV_TO_BEHAVIOR_MAP`，
并在比较前对 pred / mapped gt **统一大写归一化**。

默认映射（可在 `evaluation.yaml -> behavior_accuracy.nav_to_behavior_map` 覆盖）：

| nav 伪标签 | behavior 枚举 |
|---|---|
| straight / lane_follow | KEEP_LANE |
| left_turn | TURN_LEFT |
| right_turn | TURN_RIGHT |
| lane_change_left | CHANGE_LANE_LEFT |
| lane_change_right | CHANGE_LANE_RIGHT |
| slow_or_stop | SLOW_DOWN |
| unknown | UNKNOWN |

### 12.5 评测输出

| 文件 | 内容 |
|---|---|
| `outputs/reports/eval_summary.csv` | 各 mode 的总体指标（含 L2_*s_mean 列） |
| `outputs/reports/eval_detail.jsonl` | 每样本 `EvalSampleResult`（含 `l2_per_horizon` 字段） |
| `outputs/reports/eval_report.md` | 完整 Markdown 报告 + 分组统计 + diff 表 |

### 12.6 评测的 hard-fail 场景

| 条件 | 行为 |
|---|---|
| 输入 jsonl 不存在 | `FileNotFoundError` 退出 |
| `--decisions` 和 `--compare` 都不提供 | exit code 2 + usage 提示 |
| 两者都提供 | exit code 2 + 互斥提示 |
| 所有样本都无法构造真值且 jsonl 无内置真值 | `RuntimeError: 无法构建任何真值轨迹` |

---

## 13. 常见问题（FAQ）

**Q: 没有 GPU 能跑吗？**
A: 能。DINOv2-base 在 CPU 上 ~1s/帧，VLM 调用本身受 API 网络延迟主导（5-30s/次），GPU 增益有限。

**Q: 没有 API Key 能跑吗？**
A: 不能。第一版严格要求真实 VLM，不允许 mock。设 `DASHSCOPE_API_KEY` 或修改 `api_models.yaml` 切换其他 provider。

**Q: 一次跑要花多少 API token？**
A: 每帧场景理解 ~3000 token、决策 ~5000 token（含 3 张图）。5 帧 demo 约 4 万 token，Qwen-VL-Max 成本约 0.1-0.2 元人民币。

**Q: 可以只跑 memory_off 不跑 memory_on 吗？**
A: 可以。但 `06_run_evaluation.py --compare` 需要两条 jsonl；单 mode 用 `--decisions` 也能出报告。

**Q: jsonl 写到一半 Ctrl+C，下次怎么继续？**
A: 直接重跑同样命令；默认 `--resume` 会扫已写 sample_token 自动跳过。强制重跑加 `--no-resume`。

**Q: 怎么切换六视角环视 / 单前视感知输入？**
A: 改 `config/data_nuscenes.yaml -> perception.mode`（`single_front` / `surround_mosaic`），无需改代码；详见 [§6.5](#65-感知输入模式六视角环视拼接--oracle-感知对象) 与 [docs/perception_upgrade.md](docs/perception_upgrade.md)。

**Q: `perception_objects` 是检测模型的预测结果吗？**
A: **不是**。它是 nuScenes **GT 标注（`sample_annotation`）投影**到 6 相机得到的 oracle 真值（每个对象 `is_oracle=true`），用于研究/评测下为决策提供准确感知先验。速度/加速度沿标注 `prev` 链因果差分，缺历史置空标记不可用，不填假值。

**Q: 中期记忆怎么跨次会话累积？**
A: 改 `config/memory.yaml -> mid_term.persistence.enabled: true`。详见 §9.2「数据库形态」。

**Q: 怎么修改决策路点数量？**
A: 改 `config/decision.yaml -> trajectory.waypoint_min_num / waypoint_max_num`，prompt + parser + schema 自动同步。

**Q: 怎么换成英文 prompt？**
A: 编辑 `config/prompts.yaml`，所有提示词在此集中。详见 [docs/prompts.md](docs/prompts.md)。

**Q: 行为准确率为什么是 0%？（已修复）**
A: P6 已修复（默认 nav_to_behavior_map 自动注入 + 大小写归一化）。若仍为 0%，检查 jsonl 中 `behavior` 字段是否非空。

**Q: 怎么接入 CARLA 闭环？**
A: 已实现（v1.0），见 `carla_bridge/`（[carla_bridge/README.md](carla_bridge/README.md)）。`python -m carla_bridge.run_carla_demo --scenario straight_traffic --mode memory_on` 即可。记忆系统零改动，CARLA 数据经 `KeyframeBuilder` 喂 `OnlineDrivingLoop.step()`，决策轨迹由 Pure Pursuit+PID 回控。`src/vla_memory/planning/dynamics_planner.py` 的 `DynamicsPlanner` 仍是预留 stub（CARLA 控制器在 `carla_bridge/control/` 独立实现，不依赖它）。

**Q: FAISS 装不上怎么办？**
A: 见 §3.4 故障排查表，**固定 `faiss-cpu==1.9.0`**，不要装 1.14.x。

**Q: 跨场景边界短期记忆需要清空吗？**
A: 当前实现不会自动清空，scene-A 最后几帧仍在 scene-B 的滑窗里。如需严格隔离需要加 yaml 开关
   `isolate_across_scenes: true`（未实现，留作扩展点）。

**Q: 一次决策 VLM 收到多少张图？**
A: 由 `decision.yaml -> vlm_inputs.image_context_size` 控制，默认 3。前几帧因短期记忆未满会少（1/2/3）。

**Q: 怎么看某一帧的完整审计信息？**
A: `grep -A 60 "AUDIT frame=<sample_token>" outputs/logs/online_loop_*.log`。

---

## 14. 贡献新数据集适配器

本节给出"把项目接入新数据源（CARLA / 视频 / 自录数据集等）"的标准流程。
设计上整个 demo 的「数据访问」与「主流程」是解耦的--

> ✅ **CARLA 已接入**（v1.0）：见 `carla_bridge/`。它不走 `BaseDrivingDataset`，而是直接构建 kf dict
> 喂 `OnlineDrivingLoop.step()`——因为 CARLA 在线闭环没有未来真值轨迹，`BaseDrivingDataset` 的
> `get_future_ego_trajectory` 不适用。本节流程适用于**离线数据源**（视频 / 自录数据集等）。

只要你实现 [`BaseDrivingDataset`](src/vla_memory/data/base_dataset.py) 的全部抽象方法，
就能让 `OnlineDrivingLoop` 透明地跑在新数据源上，**不需要改 pipeline / 决策 / 记忆 / 评测中任何代码**。

### 14.1 你需要实现的抽象接口

[`BaseDrivingDataset`](src/vla_memory/data/base_dataset.py) 一共 7 个抽象方法（按依赖顺序）：

| # | 方法 | 用途 | 调用方 |
|---|---|---|---|
| 1 | `load() -> None` | 加载数据集；缺失目录 hard fail | 启动时 |
| 2 | `is_loaded() -> bool` | 检查加载状态 | 守卫检查 |
| 3 | `list_scenes() -> List[str]` | 列出所有场景标识 | `NuScenesKeyframeSampler.sample_all_scenes` |
| 4 | `iter_frames(scene_token) -> Iterator[FrameMeta]` | 按时间顺序遍历场景内所有帧 | 关键帧抽样器 |
| 5 | `get_frame_image_path(sample_token, camera_name)` | 取图像绝对路径 | DINOv2 + VLM |
| 6 | `get_ego_pose(sample_token) -> EgoState` | 取自车状态（位置/速度/加速度/航向角 + 可选 CAN bus 字段） | 在线循环 |
| 7 | `get_history_trajectory(sample_token, history_seconds)` | 取过去 N 秒 ego-centric 轨迹 | 历史段拼 prompt |
| 8 | `get_future_ego_trajectory(sample_token, future_seconds)` | 取未来 N 秒 ego-centric 真值轨迹 | **评测**用 |

**额外建议实现**（供 `enrich_keyframes_with_state` 调用，[`NuScenesAdapter`](src/vla_memory/data/nuscenes_adapter.py) 用了私有方法）：

| 方法 | 用途 |
|---|---|
| `_get_scene_samples(scene_token) -> List[dict]` | 取整个场景的 sample 列表（按时间排序） |
| `_get_sample_ego_pose(sample_token) -> dict` | 取原始 pose dict（含 `translation`、`rotation`、`timestamp` 字段） |
| `can_bus_loader` 属性 | None 即可；若你有 CAN bus 真值源，可参照 [`CanBusLoader`](src/vla_memory/data/can_bus_loader.py) 实现 |

### 14.2 数据契约（输入输出格式）

#### 关键 Schema

* **`FrameMeta`** ([schemas/frame.py](src/vla_memory/schemas/frame.py))：
  ```python
  FrameMeta(
      frame_id="<scene>_<sample>",     # 唯一标识
      scene_token="<scene>",           # 场景 token（同场景内 sample 共享）
      sample_token="<sample>",         # 关键帧/sample 唯一标识
      timestamp=1532402927665106,      # 微秒级 Unix epoch
      camera_name="CAM_FRONT",         # 摄像头名
      image_path="<绝对路径.jpg>",     # 图像文件
  )
  ```

* **`EgoState`** ([schemas/ego_state.py](src/vla_memory/schemas/ego_state.py))：
  ```python
  EgoState(
      timestamp=1532402927665106,
      x=..., y=..., z=...,             # 全局坐标系（米）
      yaw=...,                          # 弧度
      vx=..., vy=..., speed=...,        # 速度 m/s
      ax=..., ay=..., acceleration=..., # 加速度 m/s²
      # CAN bus 真值字段（可选）
      yaw_rate=...,                     # rad/s
      steering_angle=...,               # rad
      throttle=..., brake=...,          # 0-1
      gear="D",
      source="can_bus",                 # 或 "pose_diff"
  )
  ```

* **历史/未来轨迹**：`List[Dict]`，每个 dict 至少含 `{"t": float, "x": float, "y": float}`，
  **ego-centric 坐标系**（x 前向，y 左向，单位米），`t` 为相对当前帧的相对时间（秒，过去为负值，未来为正值）。

#### 坐标系约定（务必遵守）

* **全局坐标系**：`EgoState.x/y/z, yaw` 用全局米和弧度（与 nuScenes 一致）
* **ego-centric 坐标系**：历史/未来轨迹的 `x` 前向、`y` 左向；用旋转矩阵 `R(-yaw)` 从全局转到 ego-centric
  ```python
  cos_y, sin_y = math.cos(-current_yaw), math.sin(-current_yaw)
  dx, dy = global_x - current_x, global_y - current_y
  ego_x = dx * cos_y - dy * sin_y
  ego_y = dx * sin_y + dy * cos_y
  ```
* **时间戳**：统一用 **微秒** Unix epoch。其他单位（毫秒/秒）需在 `__init__` 中带 `timestamp_unit` 参数转换。

### 14.3 实现步骤（以 CARLA 适配器为例）

#### Step 1：创建新文件

```
src/vla_memory/data/
└── carla_adapter.py    ← 新增
```

#### Step 2：继承 `BaseDrivingDataset` 并实现 8 个方法

```python
# src/vla_memory/data/carla_adapter.py
from typing import Dict, Iterator, List
from src.vla_memory.data.base_dataset import BaseDrivingDataset
from src.vla_memory.schemas.frame import FrameMeta
from src.vla_memory.schemas.ego_state import EgoState
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("carla_adapter")


class CarlaAdapter(BaseDrivingDataset):
    """CARLA 仿真器适配器。

    支持两种工作模式：
      1. 离线模式：从预录制 CARLA 录像（.log / .rrd）回放
      2. 在线模式：连接 CARLA server 实时获取传感器数据

    Args:
        host: CARLA server 地址（在线模式必填）
        port: CARLA server 端口（默认 2000）
        recording_path: 离线录像路径（离线模式必填）
        camera_name: 摄像头名称（与传感器配置一致）
        can_bus_loader: 可选，外部 CAN bus 真值源
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        recording_path: str | None = None,
        camera_name: str = "CAM_FRONT",
        can_bus_loader=None,
        fallback_to_pose_diff: bool = True,
    ):
        self.host = host
        self.port = port
        self.recording_path = recording_path
        self.camera_name = camera_name
        self.can_bus_loader = can_bus_loader  # 与 NuScenesAdapter 保持一致
        self.fallback_to_pose_diff = fallback_to_pose_diff
        self._client = None
        self._loaded = False
        self._scenes_cache: Dict[str, List[dict]] = {}

    # ---- 生命周期 ----
    def load(self) -> None:
        # 检查 CARLA Python API 可用
        try:
            import carla  # noqa
        except ImportError:
            raise ImportError(
                "carla 未安装！\n"
                "请按 CARLA 官方文档安装对应版本的 Python API。"
            )
        # 连接 server 或加载录像
        # ... 你的实现 ...
        self._loaded = True
        logger.info("CARLA 数据集加载成功")

    def is_loaded(self) -> bool:
        return self._loaded

    # ---- 场景 & 帧 ----
    def list_scenes(self) -> List[str]:
        # CARLA 中"场景"可定义为：单次仿真运行（episode）/ 单个录像段
        return list(self._scenes_cache.keys())

    def iter_frames(self, scene_token: str) -> Iterator[FrameMeta]:
        samples = self._scenes_cache[scene_token]
        for s in samples:
            yield FrameMeta(
                frame_id=f"{scene_token}_{s['token']}",
                scene_token=scene_token,
                sample_token=s["token"],
                timestamp=s["timestamp_us"],  # 必须微秒
                camera_name=self.camera_name,
                image_path=s["image_path"],
            )

    # ---- 数据获取 ----
    def get_frame_image_path(self, sample_token: str, camera_name: str = "CAM_FRONT") -> str:
        # 返回图像绝对路径
        # ... 你的实现 ...
        pass

    def get_ego_pose(self, sample_token: str) -> EgoState:
        # 与 NuScenesAdapter 保持同样的「CAN bus 优先 → 差分回退」逻辑：
        if self.can_bus_loader is not None:
            try:
                # ... 从 can_bus_loader 取真值 ...
                return EgoState(source="can_bus", ...)
            except (KeyError, FileNotFoundError) as e:
                if not self.fallback_to_pose_diff:
                    raise
                logger.warning("CAN bus 查询失败，回退到差分: %s", e)
        # 回退：用相邻帧 pose 差分（参考 NuScenesAdapter.get_ego_pose 第 218-249 行）
        return EgoState(source="pose_diff", ...)

    def get_history_trajectory(self, sample_token: str, history_seconds: float = 5.0) -> List[Dict[str, float]]:
        # 沿 prev 链取过去 N 秒，转 ego-centric
        # 输出格式：[{"t": -1.0, "x": 0.5, "y": 0.0}, ...]
        pass

    def get_future_ego_trajectory(self, sample_token: str, future_seconds: float = 3.0) -> List[Dict[str, float]]:
        # 沿 next 链取未来 N 秒，转 ego-centric
        # 输出格式：[{"t": 0.5, "x": 5.0, "y": 0.0}, ...]
        pass

    # ---- 内部辅助（enrich_keyframes_with_state 需要）----
    def _get_scene_samples(self, scene_token: str) -> List[dict]:
        return self._scenes_cache[scene_token]

    def _get_sample_ego_pose(self, sample_token: str) -> dict:
        # 必须返回含 translation/rotation/timestamp 的 dict
        # {"translation": [x, y, z], "rotation": [w, x, y, z], "timestamp": us}
        pass
```

#### Step 3：在配置中加新 dataroot

```yaml
# config/data_carla.yaml（新增）
dataroot: "data/carla/recordings/"
camera_name: "CAM_FRONT"
host: "localhost"
port: 2000
recording_path: "data/carla/recordings/episode_001.log"
history_seconds: 5.0
future_seconds: 3.0
keyframe:
  step: 2
```

#### Step 4：在 `prepare_nuscenes.py` 的对等模块里注入新 adapter

参照 [`prepare_nuscenes.py`](src/vla_memory/pipeline/prepare_nuscenes.py) 写一个 `prepare_carla.py`：

```python
# src/vla_memory/pipeline/prepare_carla.py
from src.vla_memory.data.carla_adapter import CarlaAdapter

def run_prepare_carla(config):
    adapter = CarlaAdapter(
        host=config.get("host", "localhost"),
        port=config.get("port", 2000),
        recording_path=config.get("recording_path"),
        camera_name=config.get("camera_name", "CAM_FRONT"),
    )
    adapter.load()
    # ... 关键帧采样（可复用 NuScenesKeyframeSampler）...
    return {"adapter": adapter, "keyframe_index": ...}
```

或者更简洁的方案：在 `full_demo_pipeline.run_full_demo` 加 `dataset: str` 参数分发：

```python
def run_full_demo(config, mode, resume, dataset="nuscenes"):
    if dataset == "nuscenes":
        prep = run_prepare_nuscenes(config)
    elif dataset == "carla":
        prep = run_prepare_carla(config)
    else:
        raise ValueError(f"未知 dataset: {dataset}")
    # 下面所有代码（enrich_keyframes_with_state, OnlineDrivingLoop）不需要改
```

#### Step 5：在 `07_run_full_demo.py` 加 CLI 参数

```python
parser.add_argument(
    "--dataset",
    choices=["nuscenes", "carla"],
    default="nuscenes",
    help="数据源",
)
```

#### Step 6：写 fixture 和单元测试

参照 [`tests/test_nuscenes_adapter.py`](tests/test_nuscenes_adapter.py) 写
`tests/test_carla_adapter.py`：

* 用 `pytest.mark.skipif(not _carla_available(), reason="CARLA 未安装")` 保护集成测试
* 必测项：`load()`、`list_scenes()`、`iter_frames()` 按时间顺序、`get_ego_pose().source` 字段、
  `get_future_ego_trajectory()` 的 ego-centric 坐标系正确性

### 14.4 哪些代码**不需要改**

完成上述 6 步后，以下模块全部自动兼容新数据源：

* ✅ [`OnlineDrivingLoop`](src/vla_memory/pipeline/online_loop.py)（核心循环）
* ✅ 全部三层记忆（`ShortTermMemory` / `MidTermMemory` / `LongTermMemory`）
* ✅ `MemoryRetriever` / `DecisionClient` / `parse_decision_output`
* ✅ DINOv2 特征提取（接受任意 `.jpg` / `.png` / `.webp` 路径）
* ✅ VLM 调用（base64 编码逻辑通用）
* ✅ 评测器 `Evaluator` 和 `ReportWriter`（只要 `EvalSampleResult` 字段对齐）
* ✅ 所有 yaml 配置（`prompts.yaml` / `decision.yaml` / `memory.yaml` 等）

### 14.5 检查清单（PR 前自查）

- [ ] 实现了 `BaseDrivingDataset` 全部 8 个抽象方法（含 `_get_scene_samples` / `_get_sample_ego_pose` 私有辅助）
- [ ] `iter_frames()` 严格按时间戳升序 yield `FrameMeta`
- [ ] `timestamp` 字段统一为**微秒** Unix epoch
- [ ] `get_history_trajectory()` / `get_future_ego_trajectory()` 输出**ego-centric** 坐标，t 单位**秒**
- [ ] `get_ego_pose()` 返回 `EgoState`，`source` 字段标识真实来源（"can_bus" / "pose_diff" / "ground_truth" 等）
- [ ] 缺失数据 hard fail 并输出中文错误信息，**不允许返回假数据**
- [ ] 新增 `config/data_<source>.yaml` 配置文件
- [ ] 新增 `tests/test_<source>_adapter.py` 单元测试（用 `pytest.mark.skipif` 保护外部依赖）
- [ ] 在 README §7 项目结构和 §8 核心模块表格里更新对应条目
- [ ] 跑 `pytest tests/test_<source>_adapter.py` 全绿
- [ ] （可选）跑 `python scripts/07_run_full_demo.py --dataset <source> --mode memory_off --max-frames 3` 端到端通过

### 14.6 设计原则

| 原则 | 说明 |
|---|---|
| **数据契约不变** | 不要修改 `BaseDrivingDataset` 接口签名；扩展用 `Optional` 参数或子类专有方法 |
| **坐标系一致** | 全局米/弧度 + ego-centric 米；微秒时间戳 |
| **Hard fail 优于假数据** | 数据缺失立刻报错，不允许伪造 |
| **配置驱动** | 所有可调参数走 yaml，代码中不写死 |
| **CAN bus 接口可选** | 没有真值数据时 `can_bus_loader=None`，自动走差分；不要假装有 |
| **测试隔离** | 外部依赖（CARLA server / 网络）的测试必须 `skipif` 保护 |

### 14.7 已有参考实现

| 实现 | 参考文件 | 数据特点 |
|---|---|---|
| nuScenes | [src/vla_memory/data/nuscenes_adapter.py](src/vla_memory/data/nuscenes_adapter.py) | 静态数据集 + CAN bus 真值 + 多场景 |
| CAN bus 加载器（可复用模式） | [src/vla_memory/data/can_bus_loader.py](src/vla_memory/data/can_bus_loader.py) | JSON 时间序列 + 二分查找 + 单位换算 |

新适配器可以**完全复用**：
* `EgoStateBuilder` 的 `build_from_can_bus` / `build_from_poses` 两条路径
* `TrajectoryBuilder.build_history_trajectory` / `build_future_trajectory` 的 ego-centric 转换公式
* `RouteInfer.infer` 推断伪导航语义

---

## 15. 后续迭代方向

### 已交付（v0.1）

* ✅ **R1-R4**: OnlineDrivingLoop 在线循环架构，彻底消除 batch data leakage
* ✅ **P1**: 提示词集中模板化（`prompts.yaml`）
* ✅ **P2**: 中期记忆持久化（FAISS + JSON 落盘）+ 长期记忆严格 scene 匹配
* ✅ **P3**: CAN bus 真值自车状态（pose.json + vehicle_monitor.json）
* ✅ **P4**: 结构化场景理解（lanes/vehicles/pedestrians/traffic_lights/intersections）
* ✅ **P5**: 决策接收 N 张图历史
* ✅ **P6**: L2@1s/2s/3s 评测 + 行为准确率 bug 修复
* ✅ **P7**: Planning 接口预留（DynamicsPlanner + TrajectorySampler）

### v0.2 计划

* 跨场景隔离开关（短期记忆 + 中期记忆按 scene 分库）
* 中期记忆增量 save（每 N 帧 save 一次，不只 close 时一次）
* 单帧 prompt 完整日志（当前 raw_response 已记，prompt 全文需手动重建）
* 评测增加：**碰撞率代理**（用 `sample_annotation` bbox）、**off-road 率代理**（用 `nuscenes.map_expansion`）
* 优化：多场景并行处理（asyncio 调用 VLM API）

### v0.5+ 长期路线

* 接入 CARLA 闭环（实现 `DynamicsPlanner.plan()`）
* 多模态轨迹规划（实现 `TrajectorySampler.select()`，配合扩散 / AR 规划器）
* 多摄像头融合 — ⚡ **部分已实现**：六视角 surround_mosaic 2×3 拼接 + oracle 多相机 GT 投影（见 [§6.5](#65-感知输入模式六视角环视拼接--oracle-感知对象) / [docs/perception_upgrade.md](docs/perception_upgrade.md)）；真 BEV / 特征级融合仍待办
* 接入 nuScenes trainval / 自录数据 — ⚡ **trainval 已支持**（`config/data_nuscenes.yaml -> version: v1.0-trainval`）；自录数据仍待办
* 接入官方 nuScenes planning benchmark

---

## 16. 贡献与上传

### Git 工作流

```bash
# 初始化
git init
git remote add origin <your_repo_url>

# 提交
git add .
git commit -m "feat: ..."
git push -u origin main
```

### .gitignore 关键项

```
# 数据集
data/nuscenes/raw/
data/nuscenes/processed/

# 模型 / 缓存
.cache/
*.npy

# 输出
outputs/
*.jsonl
*.log

# 环境
.env
__pycache__/
*.pyc
```

完整规则见 [.gitignore](.gitignore)。

---

## 17. 许可证 & 致谢

### 许可证

MIT License。详见 [LICENSE](LICENSE)。

### 致谢

* [nuScenes](https://www.nuscenes.org/) — 数据集与官方 devkit
* [FAISS](https://github.com/facebookresearch/faiss) — 向量检索引擎
* [DINOv2](https://github.com/facebookresearch/dinov2) — 自监督视觉特征
* [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — 视觉语言模型
* [OpenAI API](https://platform.openai.com/) — 兼容 API 标准