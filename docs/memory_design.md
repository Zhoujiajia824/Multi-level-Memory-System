# 记忆设计说明文档

> 本文档详细描述 VLA Memory Demo 系统中三层记忆的设计理念、实现细节、
> 构建与检索流程，以及记忆与 VLM 决策的交互方式。

---

## 1. 设计理念

### 1.1 为什么需要分层记忆

在自动驾驶场景中，决策所需的信息具有不同的时间跨度和检索特性：

| 信息类型 | 时间跨度 | 检索方式 | 示例 |
|----------|----------|----------|------|
| 即时上下文 | 秒级 | 顺序访问 | 最近 5 帧的场景变化 |
| 相似经验 | 分钟~小时级 | 相似度检索 | 之前遇到的类似路口 |
| 通用规则 | 永久 | 规则匹配 | 安全车距、红灯停绿灯行 |

分层记忆的设计灵感来源于人类驾驶员的认知过程：

1. **短期记忆**：驾驶员对最近几秒路况的实时感知
2. **中期记忆**：驾驶员在本次驾驶中积累的"类似场景经验"
3. **长期记忆**：驾驶员通过学习和训练获得的驾驶规则知识

### 1.2 设计原则

- **分层解耦**：三层记忆独立运作，互不干扰
- **统一接口**：通过 `MemoryManager` 提供统一的管理接口
- **可配置**：记忆参数（容量、权重等）通过配置文件调整
- **可扩展**：支持新增记忆层或替换存储后端

---

## 2. 短期记忆（Short-Term Memory）

### 2.1 概述

短期记忆维护最近 N 帧的场景信息，形成连续的上下文窗口，帮助 VLM 理解
场景的时间演化趋势（如车辆正在接近、行人正在过马路等）。

### 2.2 参数配置

```yaml
# config/memory.yaml - 短期记忆配置
short_term:
  capacity: 5                # 滑动窗口容量（保留最近 5 帧）
  enable_summary: true       # 是否启用摘要生成
  summary_max_length: 200    # 摘要最大字符数
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `capacity` | 5 | 滑动窗口大小，保留最近 N 帧的场景描述 |
| `enable_summary` | true | 是否对短期记忆生成文本摘要 |
| `summary_max_length` | 200 | 摘要的最大字符长度 |

### 2.3 数据结构

```python
class ShortTermMemoryItem:
    """短期记忆项"""
    frame_id: str              # 帧唯一标识
    timestamp: int             # 时间戳（微秒）
    image_path: str            # 图像文件路径（surround_mosaic 模式下为六视角拼接图 mosaic 路径）
    scene_description: str     # 场景文本描述（由场景理解 VLM 生成）
    scene_id: str              # 场景分类 ID（如 "straight_road", "intersection"）
    weather_id: str            # 天气分类 ID（如 "sunny", "rainy"）
    ego_speed: float           # 自车速度（m/s）
```

### 2.4 滑动窗口策略

```
初始状态：[]
添加帧0：[帧0]                    ← 窗口未满，直接添加
添加帧1：[帧0, 帧1]
添加帧2：[帧0, 帧1, 帧2]
添加帧3：[帧0, 帧1, 帧2, 帧3]
添加帧4：[帧0, 帧1, 帧2, 帧3, 帧4] ← 窗口已满（capacity=5）
添加帧5：[帧1, 帧2, 帧3, 帧4, 帧5] ← 淘汰最旧的帧0
添加帧6：[帧2, 帧3, 帧4, 帧5, 帧6] ← 淘汰最旧的帧1
```

- 窗口满后，新帧进入时自动淘汰最旧的帧
- 窗口内的帧保持时间顺序

### 2.5 摘要生成策略

当 `enable_summary=True` 时，短期记忆会生成文本摘要供决策使用。

**摘要生成规则**：
1. 如果窗口内只有 1 帧，摘要为该帧的场景描述
2. 如果窗口内有 2-5 帧，拼接所有帧的场景描述，并附加统计信息
3. 摘要包含：
   - 场景变化趋势（如"从直路进入路口"）
   - 天气状态（取最新帧）
   - 速度变化趋势（加速/减速/匀速）

**摘要格式示例**：
```
【短期记忆摘要（最近5帧）】
- 最新场景：城市路口，有多方向车流，天气晴朗
- 场景变化：从直行道路逐渐接近路口
- 自车速度趋势：从 8.0 m/s 减速至 3.5 m/s
- 历史行为：KEEP_LANE -> FOLLOW -> SLOW_DOWN
```

---

## 3. 中期记忆（Mid-Term Memory）

### 3.1 概述

中期记忆基于 FAISS 向量检索，存储历史帧的视觉特征向量和元数据。
当遇到新场景时，检索视觉上相似的历史场景及其驾驶决策，为当前决策
提供参考。

### 3.2 参数配置

```yaml
# config/memory.yaml - 中期记忆配置
mid_term:
  dimension: 768             # 特征向量维度（与 DINOv2-base 一致）
  index_type: "IndexFlatIP"  # FAISS 索引类型（内积索引）
  top_k: 3                   # 检索返回的最近邻数量
  score_threshold: 0.5       # 相似度阈值，低于此值的结果被过滤
  weights:                   # 联合评分各维度权重
    visual_similarity: 0.4   # 视觉特征相似度权重
    scene_match: 0.3         # 场景类型匹配权重
    weather_match: 0.1       # 天气匹配权重
    recency: 0.2             # 时间近因性权重
```

### 3.3 数据结构

```python
class MidTermMemoryRecord:
    """中期记忆记录"""
    record_id: str              # 记录唯一标识
    frame_id: str               # 对应的帧 ID
    feature_vector: np.ndarray  # 视觉特征向量（768维；surround_mosaic 模式下从单张 mosaic 拼接图提取 CLS token，维度不变）
    scene_id: str               # 场景分类
    weather_id: str             # 天气分类
    timestamp: int              # 时间戳
    decision: DecisionOutput    # 该帧的驾驶决策
    ego_speed: float            # 自车速度
```

### 3.4 FAISS 检索流程

```
1. 接收当前帧的视觉特征向量 v_query
2. 对 v_query 进行 L2 归一化
3. 在 FAISS 索引中检索 top_k 个最近邻
4. 获取初步结果：[(id_1, score_1), (id_2, score_2), ..., (id_k, score_k)]
5. 对每个结果进行联合评分（见 3.5）
6. 按联合得分排序，过滤低于阈值的结果
7. 返回最终结果列表
```

### 3.5 联合评分公式

对每个 FAISS 检索结果，计算综合得分：

```
综合得分 = w1 * S_visual + w2 * S_scene + w3 * S_weather + w4 * S_recency

其中：
- S_visual  = FAISS 内积得分（视觉特征相似度），范围 [0, 1]
- S_scene   = 1.0（场景类型匹配）或 0.0（不匹配）
- S_weather = 1.0（天气匹配）或 0.0（不匹配）
- S_recency = exp(-λ * Δt)，Δt 为时间差（帧数），λ=0.1

权重（默认值）：
- w1 = 0.4（视觉相似度）
- w2 = 0.3（场景匹配）
- w3 = 0.1（天气匹配）
- w4 = 0.2（时间近因性）
```

**设计说明**：
- 视觉相似度权重最高，因为视觉上相似的场景通常需要相似的驾驶策略
- 场景匹配次之，确保检索结果的场景类型一致性
- 天气匹配权重最低，因为天气对驾驶策略的影响相对间接
- 时间近因性确保优先参考最近的经验

### 3.6 各维度权重调整建议

| 场景 | 建议调整 |
|------|----------|
| 场景多样性高 | 提高 `visual_similarity` 权重至 0.5 |
| 场景单一但路况复杂 | 提高 `scene_match` 权重至 0.4 |
| 天气变化频繁 | 提高 `weather_match` 权重至 0.2 |
| 长时间驾驶 | 降低 `recency` 权重至 0.1 |

### 3.7 记忆记录 metadata 扩展（Phase 1 / schema v0.2）

> 本节为 **Phase 1 新增**：扩展 `MidTermMemoryRecord` 的 metadata 结构，为后续价值门控、
> 淘汰、长期沉淀做准备。**Phase 1 不改变写入触发逻辑**（memory_on 仍逐帧全存），只让每条
> 记忆携带结构化字段。完整改造路线见 [`docs/mid_term_memory_value_gating_plan.md`](mid_term_memory_value_gating_plan.md)。

#### 3.7.1 字段分组（8 类，共 37 个新字段）

每条 `MidTermMemoryRecord` 在原 12 字段（`record_id` / `scene_id` / ... / `behavior` / `trajectory`）
基础上，新增以下 8 类 metadata。**所有新字段都有默认值**，旧 `mid_term_meta.json`（12 字段）
加载时经 Pydantic 自动补默认，不报错（向后兼容）。

| 类别 | 字段 | Phase 1 默认值 | 说明 |
|------|------|----------------|------|
| 1. 基础状态 | `memory_id` / `memory_type` / `status` / `version` / `created_at` / `updated_at` | `=record_id` / `frame_memory` / `active` / `v0.2` / 帧时间戳 / 同 created_at | 记录唯一标识、类型、状态、schema 版本、创建/更新时间 |
| 2. 来源 | `source_dataset` / `source_version` / `source_scene_token` / `source_scene_name` / `source_sample_token` / `source_frame_id` / `source_mode` | `nuscenes` / `v1.0-trainval` / 来自 kf / 来自 kf / sample_token / record_id / `memory_on`\|`memory_off` | 数据集与场景溯源（配置驱动：`data_nuscenes.yaml:dataset_name` / `version`） |
| 3. 视觉输入 | `visual_input_type` / `image_path` / `feature_path` / `feature_dim` | `perception.mode` / 主图路径 / 特征路径 / 实际维度(768) | 视觉输入类型与路径（`feature_dim` 从实际特征向量推导） |
| 4. 场景标签 | `event_type` / `scene_tags` / `risk_tags` | `frame_memory` / `[]` / `[]` | **Phase 1 不做事件识别**，留默认/空，绝不伪造标签 |
| 5. 写入价值 | `admission_score` / `admission_reasons` / `admission_policy_version` | `1.0` / `["legacy_no_gating"]` / `legacy` | **Phase 1 无门控**，标 legacy（逐帧全存）；配置驱动：`memory.yaml:mid_term.legacy_admission_*` |
| 6. 记忆价值 | `memory_value_score` / `salience_score` / `rarity_score` / `confidence_score` / `redundancy_score` / `retrieval_utility` | 全部 `None` | **Phase 1 不计算**，保持 `None`，绝不伪造分数（后续阶段由 `MemoryValueScorer` 填充） |
| 7. 使用统计 | `hit_count` / `successful_hit_count` / `failed_hit_count` / `last_retrieved_at` | `0` / `0` / `0` / `None` | 检索命中统计（后续阶段在检索命中时更新） |
| 8. 更新与删除 | `conflict_count` / `superseded_by` / `deleted_reason` / `is_active` | `0` / `None` / `None` / `True` | soft delete 标记（`is_active=False` = 逻辑删除，Phase 2 淘汰用） |

#### 3.7.2 写入位置与先读后写约束

- 唯一写入入口仍是 `OnlineDrivingLoop.step()` 末尾的 `self.mid_term.add_record(mt_record, feature)`
  （`src/vla_memory/pipeline/online_loop.py`）。Phase 1 在构造 `mt_record` 时一并填充上述字段。
- **先读后写不变**：检索在 `step()` 开头，`add_record` 在末尾；第 i 帧检索只能看到 [0, i-1] 帧。
  metadata 扩展不引入任何新的写入点，不前移到检索之前。
- **memory_on / memory_off 公平性不变**：所有 metadata 写入都在 `if self.use_memory:` 块内，
  memory_off 路径不构造 `mt_record`、不写中期记忆，对照基准保持纯净。

#### 3.7.3 decision jsonl 新增字段

每帧决策记录（`outputs/decisions_<mode>_<run_id>.jsonl`）新增 5 个字段，供离线复盘价值门控：

| 字段 | memory_on | memory_off |
|------|-----------|------------|
| `mid_term_memory_added` | `true` | `false` |
| `mid_term_memory_id` | `= sample_token` | `""` |
| `memory_admission_score` | `1.0`（legacy） | `null` |
| `memory_admission_reasons` | `["legacy_no_gating"]` | `[]` |
| `memory_record_status` | `"active"` | `""` |

#### 3.7.4 相关配置

```yaml
# config/memory.yaml -> mid_term
metadata_schema_version: "v0.2"          # 写入 MemoryRecord.version
default_memory_type: "frame_memory"      # 新记录 memory_type
default_status: "active"                 # 新记录 status
enable_value_metadata: true              # 是否填充/持久化 value_* metadata（Phase 1 价值分仍为 None）
# Phase 1 legacy 写入价值标记（无门控时默认准入元数据，配置驱动）
legacy_admission_score: 1.0
legacy_admission_reason: "legacy_no_gating"
legacy_admission_policy_version: "legacy"
```

```yaml
# config/data_nuscenes.yaml
dataset_name: "nuscenes"   # 写入 source_dataset
version: "v1.0-trainval"   # 写入 source_version
```

#### 3.7.5 向后兼容

- 旧 `outputs/memory_db/mid_term_meta.json`（12 字段）可正常加载：`load_mid_term_meta` 用
  `MidTermMemoryRecord(**data)` 反序列化，缺失字段由 Pydantic 默认值兜底。
- 不改变 FAISS 索引类型（`IndexFlatIP` / 768d）与检索逻辑，检索结果不受影响。
- `enable_value_metadata=false` 时价值类 metadata 仍持久化（Phase 1 价值分本就为 `None`），
  该开关预留给后续阶段控制是否计算/写入价值分。

### 3.8 价值门控写入（Phase 2 / MemoryAdmissionController）

> 本节为 **Phase 2 新增**：中期记忆不再逐帧全存。每帧**决策完成后、写入前**由
> `MemoryAdmissionController` 判断是否入库。低价值帧拒绝，高价值事件强制写入。
> `enabled=false` + `store_all_when_disabled=true` 时退化为阶段 1 逐帧全存（回归基线）。
> 完整设计见 [`docs/stage2_admission_design.md`](stage2_admission_design.md)。

#### 3.8.1 模块

`src/vla_memory/memory/admission.py`：
- `MemoryAdmissionPolicy`：从 `memory.yaml -> mid_term.admission` 加载（权重/阈值/过滤/事件开关，配置驱动）。
- `MemoryAdmissionResult`（dataclass）：`should_store` / `admission_score` / `admission_reasons` /
  `reject_reasons` / `event_type` / `scene_tags` / `risk_tags` / `policy_version` / `signals`。
- `MemoryAdmissionController.decide(ctx, prev_ctx)`：纯逻辑、无 IO、可离线单测。

#### 3.8.2 6 信号（各 0~1，加权求和，权重和=1.0）

| 信号 | 权重 | 来源 |
|---|---|---|
| dynamics_surprise | 0.20 | 速度/加速度/jerk/yaw_rate/航向/目标速度变化（与上一帧差分） |
| scene_salience | 0.25 | scene_id 高价值 + traffic_density=high + 路口 + 行人 + risk_factors |
| perception_change | 0.20 | 对象数变化 + 最近距离缩小 + cut_in + 行人近距 + cyclist 近距 |
| decision_change | 0.15 | behavior 变化 / risk 升级 / target_speed 大变 / 轨迹形态变 / fallback / 解析失败 |
| memory_novelty | 0.15 | 1 − 与已有中期记忆最大相似度（复用检索结果，空库=1.0） |
| posthoc_outcome_value | 0.05 | **只用当前帧决策质量代理**（fallback/解析失败/risk=high），绝不读未来 |

#### 3.8.3 高价值事件（任一命中即 force store，填 event_type/标签）

`lane_change / start / hard_brake / hard_acceleration / obstacle_avoidance / intersection /
dense_traffic / pedestrian_interaction / cyclist_interaction / cut_in / merge / turn_left /
turn_right / crosswalk / occlusion / ghost_probing_risk / long_tail`。
来源：scene_id 枚举、behavior 枚举、traffic_density、intersections.present、vehicles/pedestrians/oracle
字段、动力学阈值（`ego_state.ax` 有符号前向加速度用于急停/急起）、memory_novelty。
`occlusion`/`ghost_probing_risk` 为 risk_factors 关键词 best-effort（召回有限）。

#### 3.8.4 低价值过滤（仅在无高价值事件时生效，命中即拒绝）

- `stable_stop`：连续两帧静止 + 低风险 + STOP
- `normal_cruise`：巡航 + 低风险 + 低密度 + 低动态 + 低感知变化
- `redundant_frame`：与已有记忆最大相似度 > 0.85 + 无决策变化

#### 3.8.5 决策逻辑

```
if not enabled and store_all_when_disabled:  store=True (legacy 逐帧全存)
elif enabled:
    if 高价值事件命中 or score >= force_store_threshold(0.80):  store=True
    elif 低价值过滤命中:  store=False (reject_reasons=[filter])
    elif score >= score_threshold(0.55):  store=True
    else:  store=False (reject_reasons=[score_below_threshold])
```

#### 3.8.6 约束保护

- **先读后写不变**：`decide()` 只读当前帧 ctx + 已检索 `memory_result`（[0,i-1]）+ `prev_frame_ctx`（i-1），
  不写、不读未来。`add_record` 仍在 `step()` 末尾，且仅在 `should_store=True` 时执行。
- **memory_on / memory_off 公平性**：门控只在 `if self.use_memory:` 写入路径生效；memory_off 不入库，
  路径不经过门控（`debug_memory_off=false` 默认，保持对照纯净）。admission 在决策后运行，不影响 memory_off 决策。
- **短期记忆不受门控**：`short_term.add` 仍每帧 push（保持决策上下文窗口完整），只有中期记忆写入被门控。
- **prev_frame_ctx**：`OnlineDrivingLoop` 维护上一帧快照（不持久化、只存最近一帧），用于变化检测；
  不改 `ShortTermMemoryItem` schema，避开 push/add 顺序陷阱。

#### 3.8.7 jsonl 新增字段

`memory_admission_enabled` / `memory_admission_should_store` / `memory_admission_reject_reasons` /
`memory_event_type` / `memory_scene_tags` / `memory_risk_tags`（与阶段 1 的 `mid_term_memory_added` /
`mid_term_memory_id` / `memory_admission_score` / `memory_admission_reasons` / `memory_record_status` 共存）。

#### 3.8.8 写入率统计

```
写入率 = mid_term_meta.json 记录数 / decisions_<mode>_<run_id>.jsonl 帧数
```
开启门控后写入率应显著 < 100%（低价值帧被拒），高价值事件帧应 100% 入库。

### 3.9 容量上限与价值淘汰（Phase 3 / MemoryEvictionManager）

> 本节为 **Phase 3 新增**：中期记忆容量上限 + 价值淘汰（soft delete）+ FAISS rebuild。
> 长期运行接近容量时自动清理低价值记忆，高价值/长尾/高风险受保护；检索过滤 inactive。
> 完整设计见 [`docs/stage3_eviction_design.md`](stage3_eviction_design.md)。

#### 3.9.1 模块
- `src/vla_memory/memory/value_scorer.py` — `MemoryValueScorer`：对存量 active 记忆算持续价值
  `memory_value_score`（区别于阶段 2 admission 的写入价值），淘汰前重算并写回。
- `src/vla_memory/memory/eviction.py` — `MemoryEvictionManager`（容量触发 + soft delete）+
  `MemoryCompactionManager`（FAISS rebuild 压缩）。

#### 3.9.2 容量触发
`active_size >= max_records · trigger_ratio`（常规）或 `· emergency_ratio`（紧急）或估算磁盘
`>= max_disk_mb · trigger_ratio` → 触发淘汰，目标降到 `max_records · target_ratio`。
`max_disk_mb` 为估算（active×(特征字节+2KB)），真磁盘在 save 时精确。

#### 3.9.3 价值评分公式
`memory_value_score = 0.25·admission + 0.20·event_highvalue + 0.15·recency + 0.15·retrieval_utility
 + 0.10·(1−redundancy) + 0.10·confidence − 0.05·conflict − 0.10·lowvalue_penalty`（权重配置驱动）。
- recency：`last_retrieved_at` 优先，否则 `created_at`，指数衰减（半衰期 300s）。
- retrieval_utility：`hit_count` 归一化（检索命中追踪见 §3.9.5）。
- redundancy：按 `(scene_id, event_type)` 组频率近似（避免 O(n²)）。
- confidence：`behavior==UNKNOWN` 或 `trajectory` 空 → 低。
- lowvalue_penalty：event_type ∈ {normal_cruise, stable_stop, redundant_frame}。

#### 3.9.4 淘汰与保护
按 `memory_value_score` 升序淘汰（低价值先删）。保护：`protect_long_tail`（long_tail 事件）、
`protect_high_risk`（risk_tags 非空）、`protect_recent_high_value`（admission≥0.7 且 recency≥0.5）、
`min_keep_per_event_type`（每类至少保留 N 条）。紧急模式仅保留 min_keep，强制降到 target。

#### 3.9.5 soft delete 与检索过滤
- soft delete：`is_active=False` / `status="deleted"` / `deleted_reason` / `deleted_at`；`_records` 保留元数据，
  FAISS 在 rebuild 时才物理剔除。
- 检索过滤：`MidTermMemory.search` 跳过 `is_active=False`。
- 命中追踪：`search(now_ts=...)` 对返回 top_k 更新 `hit_count` / `last_retrieved_at`（元数据簿记，
  不改本次检索结果，不破坏先读后写）。`retriever.retrieve` / `online_loop.step` 透传 `now_ts`。

#### 3.9.6 FAISS rebuild
`IndexFlatIP` 不支持原生删除 → `reconstruct_n` 取回全部向量 → 过滤 active → 新建 IndexFlatIP 重新 add。
触发：`inactive_ratio >= rebuild_faiss_when_inactive_ratio` 或 `rebuild_after_eviction`。

#### 3.9.7 约束保护
- 先读后写不变：淘汰在 `add_record` 末尾同步触发，不前移到检索前；检索过滤 inactive 不改时间顺序。
- memory_on/off 公平性：容量管理只在 memory_on 写入路径（`add_record`）生效；memory_off 不入库、不触发淘汰。
- `capacity.enabled=false` → 不注入淘汰器，无容量上限（回归阶段 2 行为）。

#### 3.9.8 小规模测试
设 `config/memory.yaml -> mid_term.capacity.max_records: 5`，跑 `scripts/07_run_full_demo.py --mode memory_on --max-frames 20`，
检查 `mid_term_meta.json`：`is_active=False` 记录数 > 0、active 记录数 ≤ 5×target_ratio、高价值事件未被淘汰。

### 3.10 价值感知检索重排与多样性（Phase 4）

> 本节为 **Phase 4 新增**：在原有 6 路相似度基础上加 active 过滤、价值重排、多样性控制。
> 不改变 FAISS 维度，不破坏 prompt 注入（仅 top-K 渲染逻辑不变）。

#### 3.10.1 检索流程
```
FAISS 视觉相似（取足够候选）
→ 过滤 inactive/deleted/deprecated/低置信
→ 6 路相似度 final_score（visual+text+scene+weather+nav+state，权重不变）
→ 候选池 top-N（candidate_pool_size）
→ 价值重排 value_aware_score = w_sim·final_score + w_val·memory_value_score
→ 多样性约束 → top-K
→ 更新 hit_count / last_retrieved_at
```
`MidTermMemory.search` 返回 `{"results": [...], "stats": {...}}`；`MemoryRetriever.retrieve` 透传为
`mid_term_results`（列表）+ `mid_term_stats`（统计）。

#### 3.10.2 过滤
inactive（is_active=False）**始终过滤**（soft delete 硬约束）；`exclude_deleted`（status=deleted）、
`exclude_deprecated`（superseded_by 非空）、`min_confidence_score`（confidence_score 低于阈值，仅对已评分
记忆生效，None 不排除）可配。

#### 3.10.3 价值重排
`value_aware_score = 0.80·final_score + 0.20·memory_value_score`（权重配置驱动）。
`enable_value_rerank=false` 或无 retrieval_cfg → 退化为仅相似度排序（向后兼容）。
`memory_value_score` 来自阶段 3（未评分则按 0 计）。

#### 3.10.4 多样性
- `max_per_event_id`：同一 event_type 最多返回 N 条（"event_id" 用 event_type 近似）。
- `max_per_scene_token`：同一 source_scene_token 最多返回 N 条。
- `suppress_near_duplicates`：同 event_type 且 `final_score` 差 < `1 - duplicate_similarity_threshold` → 判近重复，跳过。
  （无候选-候选相似度，用 final_score 近似 + event_type 约束。）

#### 3.10.5 输出字段（jsonl）
`retrieval_candidate_count` / `retrieval_active_candidate_count` / `retrieval_filtered_count` /
`retrieved_memory_ids` / `retrieved_memory_scores` / `retrieved_memory_value_scores` /
`retrieved_memory_event_types` / `retrieved_memory_statuses`。

#### 3.10.6 约束保护
- 先读后写不变：检索在 step 开头，重排/多样性/命中统计都在检索结果算完后，不触发写入。
- 命中统计（hit_count/last_retrieved_at）是元数据簿记，不改本次结果。
- memory_on/off 公平性：检索增强对两种模式的检索逻辑一致（memory_off 不检索中期记忆，不受影响）。

### 3.11 事件级记忆（Phase 5 / EventMemory）

> 本节为 **Phase 5 新增**：帧级 → 事件级。连续高价值帧合并为一个 `event_memory`（start/peak/end
> 关键帧 + 结构化摘要），显著降低 memory_db 规模；检索优先 event_memory；保留 frame_memory 兼容。
> 完整设计见 [`docs/stage5_event_memory_design.md`](stage5_event_memory_design.md)。

#### 3.11.1 事件检测
阶段 2 admission 仍逐帧判定高价值。`should_store=True` 的帧 → 进入事件缓冲；非高价值帧 → patience 计数。
事件结束条件（任一）：高价值信号消失（patience 达 `event_end_patience_frames`）/ 达 `max_event_length_frames` /
scene 切换 / run 结束（close flush）。`EventMemoryManager` 有状态缓冲，`online_loop.step` 每帧调
`on_frame(admission_result, frame_ctx)` → 返回 `None` 或 `(event_memory_record, peak_feature)`。

#### 3.11.2 关键帧与摘要
- 关键帧：`start`=缓冲首帧、`peak`=admission 最高帧、`end`=缓冲末帧（去重，`keyframes_per_event=3`）；
  `anchor`=peak。event_memory 的 FAISS 特征 = peak 帧特征（从 `feature_path` 重 `np.load`）。
- 摘要（确定性模板，不调 VLM）：`ego_summary`（速度趋势）、`perception_summary`（对象/行人/路口计数）、
  `decision_summary`（behavior 序列）、`admission_summary`（事件类型 + peak admission + 帧数）。
- 事件 `event_type` = peak 帧的 admission event_type。

#### 3.11.3 短事件处理
`len(buffer) < min_event_length_frames` → 丢弃（噪声过滤；`min_event_length=1` 保留全部）。

#### 3.11.4 schema 扩展（向后兼容）
`MidTermMemoryRecord` 新增 event 专属字段（默认值）：`event_id` / `event_start/peak/end_sample_token` /
`anchor_sample_token` / `key_sample_tokens` / `anchor_image_path` / `key_image_paths` /
`ego_summary` / `perception_summary` / `decision_summary` / `admission_summary` / `usage`。
frame_memory 记录这些字段留空。`memory_type` 区分 `event_memory`/`frame_memory`。

#### 3.11.5 检索优先 event_memory
`search` 重排时 `prefer_event_memory=true` → event_memory 的 `value_aware_score` 加 `event_memory_bonus`，
同分时 event 居前。候选 dict 与 jsonl 含 `memory_type`。event_memory 的 event_type 本就是高价值，
阶段 3 淘汰自然给高分。

#### 3.11.6 兼容与约束
- `event_memory.enabled=false` + `store_frame_memory_when_event_disabled=true` → 阶段 2 逐帧 frame_memory。
- 旧 frame_memory 记录仍可检索；event_memory 与 frame_memory 可混存。
- 先读后写不变：事件缓冲/finalize 都在 admission 之后（写入路径），不前移到检索前。
- jsonl `memory_type` 反映本帧 admission（`event_buffered`/`frame_memory`/空）；事件 `event_id` 在
  `mid_term_meta.json`（finalize 在 jsonl 之后，单帧 jsonl 留空）。

### 3.12 冲突感知更新（Phase 6 / MemoryUpdateManager）

> 本节为 **Phase 6 新增**（"改"）：检索到相似记忆但当前决策/安全评价冲突时，对旧记忆软更新
> （降权/标记 deprecated/superseded/增 conflict_count/版本化）。**不物理删除**；unsafe 新证据不覆盖安全旧。
> 完整设计见 [`docs/stage6_update_design.md`](stage6_update_design.md)。

#### 3.12.1 冲突检测时机
`online_loop.step` 写入(i)之后调 `update_manager.process(current_ctx, mid_term_results, new_record, now_ts)`。
对每条 `final_score >= min_similarity_for_conflict_check` 的检索旧记忆分类冲突并软更新。仅在
`if self.use_memory:` 内；不前移到检索前，先读后写不变。

#### 3.12.2 冲突分类（5 类，按优先级）
- `context_mismatch`：导航或 scene_id 不同 → 不视为冲突，不更新。
- `unsafe_new_evidence`：当前决策不安全（risk=high 或 fallback/parser 失败）→ 不覆盖旧；新记忆标 `low_confidence`。
- `unsafe_old_memory`：旧记忆 `risk_tags` 非空（涉及风险场景）且新安全 → 旧 mild 更新（conflict_count++/衰减，**不自动 deprecate**，v1 近似）。
- `policy_conflict`：同情境同 scene，behavior 跨战略类别（cruise↔lateral/turn/avoid）→ conflict_count++/衰减；达 `supersede_after_conflicts` → `superseded`。
- `style_conflict`：同情境同 scene，behavior 同类别（都 cruise/都 lateral）但不同 → 风格变体，两条都保留，不衰减不删除。
- behavior 相同且同情境 → 无冲突。

#### 3.12.3 软更新（不物理删除）
只改字段：`conflict_count`/`last_conflict_at`/`conflict_reasons`/`confidence_score`/`status`/
`superseded_by`/`previous_versions`/`update_history`。`_records` 保留，FAISS 不动。

#### 3.12.4 版本化
supersede 时：`old.status="superseded"`、`old.superseded_by=new.memory_id`、
`new.previous_versions += [old.memory_id]`、双方 `update_history` 记录。
（event 模式事件未 finalize 时 `new_record=None` → 仅标 old，不建链，文档明示。）

#### 3.12.5 status 与检索过滤
status 取值：`active`/`low_confidence`/`deprecated`/`superseded`/`inactive`/`deleted`。
- `is_active=False`（inactive/deleted）→ 始终过滤（soft delete 硬约束）。
- `exclude_deprecated=true`（配置，默认 false）→ 过滤 `status in {deprecated, superseded}`（阶段 4 已对齐）。
- `low_confidence` → 不按 status 过滤（confidence_score 低 → 阶段 4 `min_confidence_score`/阶段 3 value_scorer 自然降权）。

#### 3.12.6 兼容与约束
- `update.enabled=false` → 无冲突更新（阶段 5 行为）。
- 不破坏写入/检索；旧记录新字段默认。
- unsafe 新证据绝不覆盖安全旧记忆（安全红线）。

### 3.13 中期记忆沉淀为长期记忆候选（Phase 7 / MemoryConsolidationManager）

> 本节为 **Phase 7 新增**（"中期→长期"）：离线从中期记忆库挖掘高价值、稳定、可泛化经验，总结为
> 长期记忆**候选**规则（`pending_review`），**不自动覆盖**正式长期记忆库（需人工审核晋升）。
> 完整说明见 `scripts/08_consolidate_long_term_candidates.py`。

#### 3.13.1 流程（离线批处理，不依赖 faiss/VLM）
筛选高价值 active `source_memory_type`（默认 event_memory）记忆 → 按 `(event_type, risk_tags)` 分组 →
找多次出现（≥ `min_evidence_count`）且平均价值 ≥ `min_average_memory_value_score` 的组 →
生成候选规则 → 安全过滤（剔除危险偏好）→ 按 `min_confidence` 过滤 → 写 `output_path` YAML。
`memory_value_score` 缺失回退 `admission_score`（便于未评分库也能沉淀）。

#### 3.13.2 候选规则结构
`rule_id`（`CANDIDATE_RULE_<PREFIX>_<NNN>`）/ `source` / `status=pending_review` /
`condition`（scene_tags/event_types/risk_tags）/ `recommended_strategy`（behavior/risk_level/
target_speed_adjustment）/ `rationale` / `evidence`（memory_ids/evidence_count）/ `confidence` /
`safety_guard`（must_not_override=[traffic_rules, collision_avoidance], requires_human_review）。

#### 3.13.3 候选类型
- `safety`：组有 risk_tags（如遮挡减速、行人靠近提高风险）。
- `strategy`：场景事件（路口/变道/cut-in/避障等）。
- `style`：其余（平滑起步/保守变道等用户风格）。
**风格候选不得覆盖安全规则**；所有候选带 `safety_guard`；危险偏好（高风险+激进变道）剔除。

#### 3.13.4 confidence
`confidence = 0.5·avg(memory_value_score) + 0.5·min(1, evidence_count/5)`，裁剪 [0,1]。
仅 `confidence ≥ min_confidence` 的候选保留。

#### 3.13.5 安全与约束
- **不写正式长期记忆库**（`data/knowledge/long_term_rules.yaml`）；候选写 `output_path`，`auto_promote_to_long_term=false`。
- 候选 `status=pending_review`，需人工审核后晋升。
- 危险驾驶偏好不总结（高风险变道等剔除）。

#### 3.13.6 命令
```
python scripts/08_consolidate_long_term_candidates.py
python scripts/08_consolidate_long_term_candidates.py --min-evidence-count 2 --source-memory-type event_memory
```

---

## 4. 长期记忆（Long-Term Memory）

### 4.1 概述

长期记忆存储持久化的驾驶规则和知识，通过场景匹配为当前决策提供
安全约束和行为建议。与短期/中期记忆不同，长期记忆不随场景变化，
而是在系统初始化时从知识文件加载。

### 4.2 参数配置

```yaml
# config/memory.yaml - 长期记忆配置
long_term:
  knowledge_dir: "knowledge"   # 知识文件目录
  rules_file: "rules.yaml"     # 规则文件名
  max_rules_per_scene: 5       # 每个场景最多返回的规则数量
```

### 4.3 数据结构

```python
class LongTermRule:
    """长期记忆规则"""
    rule_id: str               # 规则唯一标识（如 "R001"）
    title: str                 # 规则标题
    description: str           # 规则详细描述
    scene_id: str              # 适用场景 ID（"all" 表示所有场景）
    weather_id: str            # 适用天气 ID（"all" 表示所有天气）
    priority: int              # 优先级（1 最高，数字越大优先级越低）
    condition: str             # 触发条件描述
    action: str                # 建议行动描述
```

### 4.4 规则匹配策略

```
1. 获取当前帧的 scene_id 和 weather_id
2. 遍历所有已加载规则
3. 对每条规则进行匹配判断：
   a. 规则的 scene_id == "all" 或 == 当前 scene_id → 场景匹配
   b. 规则的 weather_id == "all" 或 == 当前 weather_id → 天气匹配
   c. 同时匹配场景和天气 → 该规则命中
4. 按优先级排序命中的规则
5. 返回前 max_rules_per_scene 条规则
```

**匹配示例**：

当前场景：`scene_id="intersection"`, `weather_id="rainy"`

| 规则 | scene_id | weather_id | 是否命中 |
|------|----------|------------|----------|
| R001 | all | all | 命中 |
| R002 | intersection | all | 命中 |
| R003 | straight_road | all | 未命中（场景不匹配） |
| R004 | intersection | rainy | 命中 |
| R005 | all | sunny | 未命中（天气不匹配） |

### 4.5 知识文件格式

知识文件存储在 `knowledge/rules.yaml`，格式如下：

```yaml
# 驾驶规则知识库
rules:
  # 通用安全规则
  - rule_id: "R001"
    title: "安全车距规则"
    description: "行驶中应与前车保持安全距离，城市道路至少 2 秒时距"
    scene_id: "all"
    weather_id: "all"
    priority: 1
    condition: "前方有车辆且速度 > 5 m/s"
    action: "保持安全车距，随时准备减速"

  - rule_id: "R002"
    title: "路口通行规则"
    description: "通过路口时应减速观察，确认安全后通行"
    scene_id: "intersection"
    weather_id: "all"
    priority: 1
    condition: "接近路口或正在路口内"
    action: "减速至 5 m/s 以下，观察各方向来车"

  - rule_id: "R003"
    title: "雨天减速规则"
    description: "雨天路滑，应降低车速并增大跟车距离"
    scene_id: "all"
    weather_id: "rainy"
    priority: 2
    condition: "天气为雨天"
    action: "降低目标速度 20%，增大跟车距离 50%"

  - rule_id: "R004"
    title: "行人礼让规则"
    description: "检测到行人时应减速或停车让行"
    scene_id: "all"
    weather_id: "all"
    priority: 1
    condition: "前方检测到行人"
    action: "减速或停车，等待行人通过"

  - rule_id: "R005"
    title: "变道规则"
    description: "变道前应观察目标车道，确认安全后打转向灯变道"
    scene_id: "all"
    weather_id: "all"
    priority: 2
    condition: "需要变道（导航指示或避障）"
    action: "观察目标车道，确认安全距离后平稳变道"
```

---

## 5. 记忆的构建和检索流程

### 5.1 记忆构建流程（写入）

```
┌─────────────────────────────────────────────────┐
│              记忆构建流程（每帧执行一次）          │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. 接收当前帧数据                               │
│     ├── KeyFrame（帧基本信息）                    │
│     ├── EgoState（自车状态）                      │
│     ├── feature_vector（视觉特征）               │
│     └── SceneUnderstanding（场景理解结果）        │
│                                                 │
│  2. 写入短期记忆                                 │
│     ├── 构建 ShortTermMemoryItem                 │
│     ├── 添加到滑动窗口                            │
│     └── 如有旧帧被淘汰，触发摘要更新              │
│                                                 │
│  3. 写入中期记忆                                 │
│     ├── 构建 MidTermMemoryRecord                 │
│     ├── 将特征向量添加到 FAISS 索引               │
│     └── 将元数据存入元数据表                      │
│                                                 │
│  4. （长期记忆在系统初始化时加载，运行时只读）      │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 5.2 记忆检索流程（读取）

```
┌─────────────────────────────────────────────────┐
│              记忆检索流程（每帧决策前执行）         │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. 短期记忆检索                                 │
│     ├── 获取滑动窗口内所有帧                      │
│     ├── 生成短期记忆摘要                          │
│     └── 输出：摘要文本 + 最近帧的场景描述         │
│                                                 │
│  2. 中期记忆检索                                 │
│     ├── 用当前帧特征向量查询 FAISS                │
│     ├── 对结果进行联合评分                        │
│     ├── 过滤低分结果                              │
│     └── 输出：相似历史场景列表（含决策参考）       │
│                                                 │
│  3. 长期记忆检索                                 │
│     ├── 用当前场景 ID + 天气 ID 匹配规则          │
│     ├── 按优先级排序                              │
│     └── 输出：适用的驾驶规则列表                  │
│                                                 │
│  4. 合并三层记忆结果                              │
│     ├── 构建统一的记忆上下文文本                   │
│     └── 输出：完整的记忆上下文（用于决策 Prompt）  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 5.3 记忆上下文格式

三层记忆合并后，形成以下格式的上下文文本，嵌入决策 Prompt 中：

```
【驾驶记忆上下文】

── 短期记忆（最近5帧）──
最新场景：城市直行道路，前方有车辆缓行
场景变化：一直处于直行路段
速度趋势：从 8.0 m/s 减速至 5.0 m/s
历史行为：KEEP_LANE -> KEEP_LANE -> FOLLOW

── 中期记忆（相似场景经验）──
[相似场景1] 距离=0.85，场景=直行道路
  描述：城市直行道路，前方车辆减速
  决策：FOLLOW，目标速度 4.5 m/s

[相似场景2] 距离=0.78，场景=直行道路
  描述：城市直行道路，前方有多辆车
  决策：FOLLOW，目标速度 3.0 m/s

── 长期记忆（驾驶规则）──
[R001] 安全车距规则：保持安全车距，随时准备减速
[R004] 行人礼让规则：减速或停车，等待行人通过
```

---

## 6. 与 VLM 决策的交互方式

### 6.1 交互流程

```
                          ┌─────────┐
                          │  VLM    │
                          │  决策   │
                          └────┬────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────┐  ┌──────▼─────────┐
    │  短期记忆上下文  │  │  当前帧图像  │  │  长期规则提示   │
    │  (文本)         │  │  (视觉输入) │  │  (文本)        │
    └────────────────┘  └────────────┘  └────────────────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  决策 Prompt 模板    │
                    │  （整合所有上下文）   │
                    └─────────────────────┘
```

### 6.2 Prompt 模板结构

决策 Prompt 由以下部分组成：

1. **系统指令**：角色设定（你是一个自动驾驶决策系统）
2. **导航指令**：当前导航语义（直行/左转/右转等）
3. **自车状态**：当前速度、航向角等
4. **短期记忆上下文**：最近帧的场景演化摘要
5. **中期记忆上下文**：相似历史场景的决策参考
6. **长期规则提示**：适用的驾驶规则
7. **输出格式要求**：要求 VLM 输出结构化的 JSON

### 6.3 记忆对决策的影响

| 记忆层 | 对决策的影响 | 示例 |
|--------|-------------|------|
| 短期记忆 | 帮助 VLM 理解场景的时间连续性 | "前方车辆正在减速"→ FOLLOW 而非 KEEP_LANE |
| 中期记忆 | 为 VLM 提供相似场景的参考决策 | "之前类似路口选择左转"→ 增强左转决策的置信度 |
| 长期记忆 | 为 VLM 提供安全约束 | "安全车距规则"→ 避免过于激进的跟车策略 |

### 6.4 memory_on vs memory_off 对比

系统支持通过配置关闭记忆功能（`memory_off` 模式），用于评测记忆的实际效果：

| 模式 | 短期记忆 | 中期记忆 | 长期记忆 |
|------|----------|----------|----------|
| `memory_on` | 启用（滑动窗口+摘要） | 启用（FAISS 检索） | 启用（规则匹配） |
| `memory_off` | 仅保留当前帧 | 不检索 | 不检索 |

在 `memory_off` 模式下，VLM 仅基于当前帧图像和自车状态进行决策，
没有任何历史上下文。这为对比实验提供了基准（baseline）。

---

## 7. 配置参考

完整的记忆配置参见 `config/memory.yaml`：

```yaml
# config/memory.yaml 完整配置
short_term:
  capacity: 5
  enable_summary: true
  summary_max_length: 200

mid_term:
  dimension: 768
  index_type: "IndexFlatIP"
  top_k: 3
  score_threshold: 0.5
  weights:
    visual_similarity: 0.4
    scene_match: 0.3
    weather_match: 0.1
    recency: 0.2

long_term:
  knowledge_dir: "knowledge"
  rules_file: "rules.yaml"
  max_rules_per_scene: 5
```
