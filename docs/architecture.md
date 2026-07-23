# 系统架构说明文档

> 本文档描述 VLA Memory Demo 项目的总体架构设计，包括分层记忆系统的模块划分、
> 模块间依赖关系以及扩展点说明。

---

## 1. 系统总体架构

VLA Memory Demo 采用**分层记忆驱动的自动驾驶决策架构**，核心思想是将感知信息
存入不同时间跨度的记忆层，在决策时检索相关记忆以辅助视觉语言模型（VLM）做出
更准确的驾驶决策。

### 1.1 架构图（ASCII）

```
┌─────────────────────────────────────────────────────────────────┐
│                      VLA Memory Demo 系统架构                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │  数据层        │    │  感知层        │    │  记忆层        │   │
│  │  (Data Layer)  │───>│(Perception)    │───>│ (Memory Layer) │   │
│  │               │    │               │    │               │   │
│  │ - nuScenes    │    │ - DINOv2      │    │ - 短期记忆     │   │
│  │ - 路线推断    │    │ - VLM (API)   │    │ - 中期记忆     │   │
│  │ - 轨迹构建    │    │ - 场景理解     │    │ - 长期记忆     │   │
│  └───────────────┘    └───────────────┘    └───────┬───────┘   │
│                                                     │           │
│                                                     ▼           │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │  评测层        │<───│  决策层        │<───│  检索层        │   │
│  │ (Evaluation)  │    │ (Decision)    │    │ (Retrieval)   │   │
│  │               │    │               │    │               │   │
│  │ - ADE/FDE     │    │ - VLM 推理    │    │ - FAISS 检索  │   │
│  │ - 轨迹重采样  │    │ - 规则 Fallback│    │ - 联合评分    │   │
│  │ - 报告生成    │    │ - 输出解析     │    │ - 规则匹配    │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 分层记忆架构

```
┌──────────────────────────────────────────┐
│              长期记忆 (Long-term)         │
│  - 类型：规则型知识                       │
│  - 存储：YAML/JSON 知识文件              │
│  - 特点：持久化，不随场景变化             │
│  - 示例：安全车距规则、信号灯处理规则      │
├──────────────────────────────────────────┤
│              中期记忆 (Mid-term)          │
│  - 类型：特征向量检索                     │
│  - 存储：FAISS 向量索引 (IndexFlatIP)    │
│  - 特点：相似场景检索，跨片段关联          │
│  - 示例：历史相似路口的驾驶经验           │
├──────────────────────────────────────────┤
│              短期记忆 (Short-term)        │
│  - 类型：滑动窗口序列                    │
│  - 存储：内存队列 (deque)                │
│  - 特点：最近 N 帧的连续上下文            │
│  - 示例：最近 10 帧的场景描述和自车状态    │
└──────────────────────────────────────────┘
```

---

## 2. 模块职责划分

### 2.1 数据层 (`src/vla_memory/data/`)

| 模块 | 职责 |
|------|------|
| `nuscenes_adapter.py` | 加载 nuScenes 数据集，封装 nuscenes-devkit，提供场景遍历、帧迭代、多相机图像路径、ego_pose 查询、oracle 感知对象（`get_perception_objects`） |
| `route_infer.py` | 根据未来轨迹推断伪导航语义（直行/左转/右转/变道/停车） |
| `ego_state_builder.py` | 从 ego_pose / CAN bus 构建自车状态（P3 起优先 CAN bus 真值，差分回退） |
| `can_bus_loader.py` | nuScenes CAN bus 真值加载器（pose.json + vehicle_monitor.json） |
| `trajectory_builder.py` | 构建 ego-centric 坐标系下的历史轨迹和未来真值轨迹 |
| `oracle_perception.py` | 基于 nuScenes GT 标注（sample_annotation）投影到 6 相机 + 因果运动学，生成 oracle `perception_objects`（GT 真值，非模型预测） |
| `base_dataset.py` | 数据集抽象基类 |

> ⚠️ P7 已移除 `video_adapter.py` 和 `image_sequence_adapter.py` —— 它们曾是空 stub
> 且因导入 `BaseDataset`（不存在）破损。后续接入新数据源（CARLA / 视频 / 图片序列）
> 时，请基于 `base_dataset.BaseDrivingDataset` 重新实现。

- **输入**：nuScenes 数据集路径、配置参数
- **输出**：标准化帧数据（`FrameMeta`、`EgoState`、`Trajectory`）

### 2.2 感知层 (`src/vla_memory/perception/`)

| 模块 | 职责 |
|------|------|
| `dinov2_extractor.py` | 加载 DINOv2 真实模型权重，提取 768 维图像特征向量 |
| `image_feature_extractor.py` | 特征提取抽象基类 |
| `vlm_client.py` | VLM 客户端抽象基类 |
| `openai_compatible_client.py` | OpenAI 兼容 VLM 客户端（支持 Qwen-VL 等所有兼容 API） |
| `scene_understanding.py` | DINOv2 + VLM 集成的场景理解流水线 |
| `surround_mosaic.py` | 六视角环视拼接（2×3 surround-view mosaic），surround_mosaic 模式下替代单前视图作为感知图像 |

- **默认模型**：`facebook/dinov2-base`（768 维，L2 归一化）
- **默认 VLM**：Qwen-VL（通过 DashScope API）
- **输入**：图像文件路径（surround_mosaic 模式下为 `outputs/mosaic/<token>.jpg` 拼接图）
- **输出**：归一化特征向量（numpy array）+ 场景结构化 JSON

### 2.3 记忆层 (`src/vla_memory/memory/`)

| 模块 | 职责 |
|------|------|
| `short_term_memory.py` | 短期记忆管理（deque 滑动窗口、摘要生成） |
| `mid_term_memory.py` | 中期记忆管理（FAISS 联合评分、上下文构建） |
| `long_term_memory.py` | 长期记忆管理（YAML 规则加载、场景+天气匹配） |
| `faiss_store.py` | FAISS 向量存储与检索（IndexFlatIP） |
| `vector_store.py` | 向量存储抽象基类 |
| `retrieval.py` | 三层记忆统一检索入口（MemoryRetriever） |
| `memory_record_io.py` | 记忆序列化/反序列化（JSONL + FAISS 持久化） |

### 2.4 决策层 (`src/vla_memory/decision/`)

| 模块 | 职责 |
|------|------|
| `decision_client.py` | 调用 VLM API 进行驾驶决策 |
| `prompt_builder.py` | 构建 VLM 决策 Prompt（含记忆上下文） |
| `output_parser.py` | 解析 VLM 输出为结构化决策（JSON 校验 + 字段验证） |
| `rule_fallback.py` | 规则兜底，当 VLM 输出格式错误时生成 11 种行为的可评测轨迹 |
| `dynamics_adapter.py` | 动力学适配器（预留 stub，第一版不实现） |

### 2.5 评测层 (`src/vla_memory/evaluation/`)

| 模块 | 职责 |
|------|------|
| `metrics.py` | 计算 ADE、FDE、轨迹有效率、行为准确率等指标，预留 collision/offroad 接口 |
| `evaluator.py` | 评测管理器，支持分组统计（scene_id / weather_id / behavior） |
| `report_writer.py` | 生成 CSV / JSONL / Markdown 三种评测报告 |

### 2.6 流水线层 (`src/vla_memory/pipeline/`)

| 模块 | 职责 |
|------|------|
| `prepare_nuscenes.py` | 数据准备流水线（加载数据集、关键帧采样） |
| `scene_pipeline.py` | 场景理解流水线（DINOv2 + VLM） |
| `memory_pipeline.py` | 记忆构建流水线（三层记忆初始化与填充） |
| `decision_pipeline.py` | 决策流水线（记忆检索 + VLM 决策 + 输出解析） |
| `eval_pipeline.py` | 评测流水线（真值构建 + 指标计算 + 报告生成） |
| `full_demo_pipeline.py` | 完整 Demo 流水线（串联所有步骤） |

---

## 3. 模块间依赖关系

### 3.1 依赖关系图

```
data ──> perception ──> memory ──> decision ──> evaluation
  │                       ▲                │
  │                       │                │
  └───────────────────────┴────────────────┘
          （common 和 schemas 被所有模块共享）
```

### 3.2 详细依赖说明

```
common/config.py              ← 被所有模块依赖，提供统一配置加载
schemas/                      ← 被所有模块依赖，定义数据模型
data/nuscenes_adapter.py      ← 被 perception, evaluation 依赖
data/ego_state_builder.py     ← 被 pipeline, keyframes 依赖
data/trajectory_builder.py    ← 被 pipeline, evaluation 依赖
data/route_infer.py           ← 被 pipeline, evaluation 依赖
perception/dinov2_extractor.py ← 被 memory（中期记忆特征入库）依赖
memory/retrieval.py           ← 被 decision 依赖
decision/decision_client.py   ← 被 evaluation 依赖
evaluation/evaluator.py       ← 无下游依赖（终端模块）
```

### 3.3 公共模块

| 模块 | 说明 |
|------|------|
| `common/config.py` | 配置加载与管理，支持多 YAML 合并和嵌套访问 |
| `common/logging_utils.py` | 统一日志管理（文件+控制台双输出） |
| `common/path_utils.py` | 路径工具函数 |
| `common/json_utils.py` | JSON 解析和校验工具 |
| `common/image_io.py` | 图像读取和预处理 |

---

## 3.5 在线循环时序（R2 起的核心 demo 架构）

R2 重构后，demo 的运行单位从「批处理瀑布」变为「逐帧在线循环」（`OnlineDrivingLoop`）。
单次运行只跑一种 mode（`memory_on` 或 `memory_off`）；评测作为独立步骤跑。

### 3.5.1 一次 07_run_full_demo 的高层时序

```
scripts/07_run_full_demo.py
        │
        ▼
full_demo_pipeline.run_full_demo(config, mode, resume)
        │
        ├──[1/3]──> prepare_nuscenes.run_prepare_nuscenes
        │             ├─ NuScenesAdapter.load (含 CanBusLoader 注入)
        │             └─ NuScenesKeyframeSampler → keyframe_index
        │
        ├──[2/3]──> enrich_keyframes_with_state
        │             ├─ EgoStateBuilder.build (CAN bus 优先, 差分回退)
        │             ├─ TrajectoryBuilder (history + GT future)
        │             └─ RouteInfer (伪导航语义)
        │             → List[Dict] keyframes
        │
        └──[3/3]──> OnlineDrivingLoop.run(keyframes)
                      │
                      └── for kf in keyframes:  ← 见 3.5.2
                            OnlineDrivingLoop.step(kf)
```

### 3.5.2 单帧 step 的时序（核心正确性保证位置）

```
OnlineDrivingLoop.step(kf)
    │
    ├─ resume_set 含 sample_token? → return None      (中断恢复)
    │
    ├─[a] SceneUnderstandingPipeline.process_frame
    │      ├─ DINOv2.extract → 768-d feature → save .npy
    │      └─ VLM.understand_scene → 结构化 JSON (含 lanes/vehicles/...)
    │
    ├─[b] MemoryRetriever.retrieve(use_*=use_memory)
    │      ├─ short_term: 滑窗中过去 [0, i-1] 帧
    │      ├─ mid_term:   FAISS 6 路融合检索过去 [0, i-1] 条记忆 ★
    │      └─ long_term:  按 scene_id 严格匹配 YAML 规则
    │
    │  ★ 关键正确性：mid_term.add_record 在末尾 (h)，
    │     所以第 i 帧检索时索引里只有过去帧。无 data leakage。
    │
    ├─[c] image_paths = short_term.get_recent_image_paths(N-1) + [current_image]
    │      （memory_off 模式下只含当前帧）
    │
    ├─[d] DecisionClient.decide(prompt, image_paths)
    │      → OpenAICompatibleVLMClient.decide
    │      → VLM raw response (含推理过程, 日志可见 [DECISION])
    │
    ├─[e] parse_decision_output / generate_fallback_decision
    │
    ├─[f] 组装 record dict (含 scene/decision/image_paths/...)
    │
    ├─[g] append_decision_record(jsonl)  + flush + fsync   (持久化)
    │
    ├─[h] short_term.add(ShortTermMemoryItem)            ← 滑窗 push
    │     mid_term.add_record(MidTermMemoryRecord 含 behavior/reason/trajectory)
    │                                                      ← FAISS 索引增长
    │     (memory_off 模式下跳过 h 全部，保持对照组纯净)
    │
    └─ return record
```

### 3.5.3 中断恢复

* 每条 record `append_decision_record` 后立即 `flush + os.fsync`，保证已写帧不丢。
* 重启时 `OnlineDrivingLoop.setup` 扫已存在的 jsonl，把所有 `sample_token` 加进 `_resume_set`。
* 主循环每帧开头检查 `_resume_set`，命中则 `return None` 跳过（不调 VLM、不更新记忆）。
* 配合 `mid_term.persistence.enabled=true`（[config/memory.yaml](../config/memory.yaml)），FAISS 索引也可跨次会话自动加载，让重启后第 K+1 帧能用到第 K 帧的中期记忆。

### 3.5.4 评测分离

评测**不再混在主循环里**——`run_full_demo` 只写 jsonl，不跑评测。

```
单 mode:  python scripts/06_run_evaluation.py --decisions <jsonl>
双 mode:  python scripts/06_run_evaluation.py --compare <on.jsonl> <off.jsonl>
```

`eval_pipeline.run_eval_pipeline(results, mode, ...)` 与 `run_eval_compare(results_by_mode, ...)`
是 R3 拆出来的两个公开入口，复用同一个内部 `_prepare_ground_truth` + `_evaluate_one_mode` 辅助。

---

## 4. 扩展点说明

### 4.1 接入新数据源

要支持新的自动驾驶数据集（如 Waymo、Argoverse），需要：

1. 在 `src/vla_memory/data/` 下创建新的适配器（如 `waymo_adapter.py`）
2. 继承 `BaseDrivingDataset` 接口，实现 `load()`、`iter_frames()`、`get_ego_pose()` 等方法
3. 将原始数据转换为统一的 `FrameMeta`、`EgoState`、`Trajectory` 格式
4. 在 `config/` 下添加对应的数据配置文件（如 `data_waymo.yaml`）
5. 在 `default.yaml` 中注册新数据源

**关键约束**：新适配器的输出必须与现有 Schema 完全兼容。

### 4.2 接入新特征模型

要使用新的视觉特征提取模型（如 ViT、EfficientNet），需要：

1. 在 `src/vla_memory/perception/` 下创建新的提取器（如 `vit_extractor.py`）
2. 继承 `ImageFeatureExtractor` 接口，实现 `extract()` 方法
3. 在配置文件中设置 `feature_extractor.model_name` 为新模型
4. 确保输出向量的维度与 FAISS 索引维度一致（需更新 `memory.yaml` 中的 `dimension`）

**注意**：更换特征模型后需要重建 FAISS 索引。

### 4.3 接入新 VLM

要使用新的视觉语言模型进行决策，需要：

1. 在 `config/api_models.yaml` 中添加新的服务商配置
2. 设置 `base_url`、`api_key_env`、`model_name`
3. 如果新 VLM 的输入/输出格式不同，需在 `decision/output_parser.py` 中添加适配逻辑
4. 如果 Prompt 格式有差异，在 `decision/prompt_builder.py` 中添加新模板

### 4.4 接入新记忆类型

要扩展记忆系统（如增加"工作记忆"层），需要：

1. 在 `src/vla_memory/memory/` 下创建新的记忆模块
2. 在 `schemas/memory.py` 中定义新的数据结构
3. 在 `memory/retrieval.py` 的 `MemoryRetriever` 中注册新记忆层
4. 更新决策 Prompt 以利用新的记忆信息

### 4.5 接入新评测指标

要添加新的评测指标，需要：

1. 在 `evaluation/metrics.py` 中实现新的计算函数
2. 在 `evaluation/evaluator.py` 的评测流程中调用新指标
3. 在 `evaluation/report_writer.py` 的报告模板中添加新指标展示
4. 在 `config/evaluation.yaml` 中添加新指标的配置项

---

## 5. 目录结构总览

```
vla_memory_demo/
├── config/                     # 配置文件目录
│   ├── default.yaml            # 默认主配置
│   ├── api_models.yaml         # VLM API 模型配置
│   ├── data_nuscenes.yaml      # nuScenes 数据配置
│   ├── memory.yaml             # 记忆系统配置
│   ├── decision.yaml           # 决策配置
│   └── evaluation.yaml         # 评测配置
├── src/vla_memory/             # 源代码目录
│   ├── common/                 # 公共模块
│   ├── data/                   # 数据层
│   ├── keyframes/              # 关键帧采样
│   ├── perception/             # 感知层（DINOv2 + VLM）
│   ├── memory/                 # 记忆层
│   ├── decision/               # 决策层
│   ├── evaluation/             # 评测层
│   ├── pipeline/               # 流水线层
│   └── schemas/                # 数据模型定义
├── tests/                      # 测试文件目录
├── docs/                       # 文档目录
├── data/knowledge/             # 长期记忆知识文件
└── outputs/                    # 输出结果目录
```

> **六视角环视 + Oracle 感知特性**新增文件：`src/vla_memory/schemas/perception.py`
> （`PerceptionObject`）、`src/vla_memory/data/oracle_perception.py`（GT 投影+因果运动学）、
> `src/vla_memory/perception/surround_mosaic.py`（2×3 拼接）、产物目录 `outputs/mosaic/`、
> 文档 `docs/perception_upgrade.md`。详见 [perception_upgrade.md](perception_upgrade.md)。

---

## 6. 技术栈

| 类别 | 技术 |
|------|------|
| 编程语言 | Python 3.10+ |
| 数据模型 | Pydantic v2 |
| 视觉特征 | DINOv2 (facebook/dinov2-base) |
| 向量检索 | FAISS (IndexFlatIP) |
| 视觉语言模型 | Qwen-VL / 智谱 GLM-4V / DeepSeek-VL |
| 数据集 | nuScenes v1.0-mini |
| 配置管理 | YAML（多文件合并） |
| 测试框架 | pytest |
