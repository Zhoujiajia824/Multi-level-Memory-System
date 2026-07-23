# 阶段 5 设计：事件级记忆 EventMemory

> 本文档为阶段 5（帧级 → 事件级记忆）的实现设计。承接阶段 1-4（metadata / 写入门控 /
> 容量淘汰 / 价值感知检索）。连续高价值帧合并为一个 `event_memory`（少量关键帧 + 结构化摘要），
> 检索优先 event_memory，保留 frame_memory 兼容。

## 0. 目标

把"逐高价值帧入库"升级为"连续高价值帧合并为一个事件入库"：每事件保存 start/peak/end 关键帧 +
事件摘要，显著降低 memory_db 规模，经验更结构化。`event_memory.enabled=false` 退化为阶段 2
逐帧 frame_memory（回归基线）。

---

## 1. 关键设计决策

### 1.1 事件检测在 online_loop.step（admission 之后）
阶段 2 的 admission 仍逐帧判定高价值。`should_store=True` 的帧 → 进入事件缓冲；非高价值帧 →
patience 计数。事件结束条件（任一）：高价值信号消失（patience 达 `event_end_patience_frames`）、
达 `max_event_length_frames`（强制结束）、scene 切换（`scene_token` 变化）、run 结束（close flush）。

### 1.2 状态化缓冲在 EventMemoryManager
`EventMemoryManager` 持有 `_buffer`（帧上下文列表）/`_event_type`/`_patience`。online_loop 每帧调
`on_frame(admission_result, frame_ctx)` → 返回 `None`（仍缓冲中）或 `(event_memory_record, peak_feature)`
（事件结束finalize）。online_loop 负责调 `mid_term.add_record(record, feature=peak_feature)`。
EventMemoryManager 不直接依赖 mid_term（避免循环），只返回记录+特征。

### 1.3 关键帧选择
- `start` = 缓冲首帧（事件 onset）
- `peak` = `admission_score` 最高的帧（事件高潮 / 最代表性）
- `end` = 缓冲末帧（状态恢复）
- `keyframes_per_event=3`，不足去重；`anchor_sample_token` = peak。
- event_memory 的 FAISS 特征 = **peak 帧特征**（从 `feature_path` 重新 `np.load`，每事件一次 IO）。

### 1.4 事件 event_type = peak 帧的 admission event_type
（最代表性；跨类型事件按高潮归类。`stop_and_go` 等复合事件 v1 不特殊处理，按 peak 标签。）

### 1.5 短事件处理
`len(buffer) < min_event_length_frames` → **丢弃**（噪声过滤，过滤孤立单帧抖动；`min_event_length=1` 可保留全部）。
文档明示该取舍。

### 1.6 事件摘要（确定性模板，不调 VLM）
从缓冲帧生成 4 个文本摘要（cheap，可后续升级 VLM）：
- `ego_summary`：速度/加速度区间与趋势（如 "speed 8.2→3.1 m/s, decelerated"）。
- `perception_summary`：对象/行人/路口计数与最近距离（如 "3 vehicles, 1 pedestrian crossing @6m"）。
- `decision_summary`：behavior 序列（如 "KEEP_LANE → SLOW_DOWN → STOP"）。
- `admission_summary`：事件类型 + peak admission + 帧数（如 "hard_brake event, peak=0.85, 4 frames"）。

### 1.7 schema 扩展（MidTermMemoryRecord 新增 event 专属字段，默认值，向后兼容）
`event_id` / `event_start_sample_token` / `event_peak_sample_token` / `event_end_sample_token` /
`anchor_sample_token` / `key_sample_tokens`(List) / `anchor_image_path` / `key_image_paths`(List) /
`ego_summary` / `perception_summary` / `decision_summary` / `admission_summary` / `usage`(dict)。
frame_memory 记录这些字段留默认空值。`memory_type` 复用现有字段（`event_memory` / `frame_memory`）。

### 1.8 检索优先 event_memory
`mid_term_memory.search` 重排时，`prefer_event_memory=true` 则 `memory_type==event_memory` 的候选
`value_aware_score` 加 configurable bonus（`event_memory_bonus`）。候选 dict 与 jsonl 增加 `memory_type`。
（event_memory 的 event_type 本就是高价值，阶段 3 value_scorer 自然给高分，淘汰不会误删。）

### 1.9 兼容
- `event_memory.enabled=false` + `store_frame_memory_when_event_disabled=true` → 阶段 2 逐帧 frame_memory。
- 旧 frame_memory 记录仍可检索（memory_type=frame_memory，无 event 字段，默认空）。
- event_memory 与 frame_memory 可混存于同一 memory_db。

---

## 2. 新增模块 `src/vla_memory/memory/event_memory.py`

- `EventMemoryConfig`：从 `mid_term.event_memory` 加载（enabled / prefer_event_memory /
  max_event_length_frames / min_event_length_frames / event_end_patience_frames / keyframes_per_event /
  store_frame_memory_when_event_disabled / event_memory_bonus）。
- `EventMemoryManager`：
  - `on_frame(admission_result, frame_ctx)` → `Optional[(MidTermMemoryRecord, np.ndarray)]`：
    高价值→缓冲；非高价值→patience→达阈值 finalize；缓冲达 max→强制 finalize。
  - `_finalize()` → 选关键帧（start/peak/end）+ 生成 4 摘要 + 建 event_memory record（含阶段1 metadata
    从 peak 帧填充）+ `np.load(peak.feature_path)` 取特征；`len<min` → 丢弃返回 None。
  - `flush()` → 强制 finalize（scene 切换 / close）。
  - `_generate_summaries(buffer)` / `_select_keyframes(buffer, peak_idx, k)` / `_make_event_id()` / `_reset()`。

---

## 3. `online_loop.py` 改动

- `__init__`：`self._event_manager=None` / `self._event_memory_enabled=False` / `self._prev_scene_token=""`。
- `setup`：建 `EventMemoryManager`（从 config），缓存 enabled；把 `prefer_event_memory`/`event_memory_bonus`
  并入 `mt_retrieval` 传给 MidTermMemory。
- `step` i 块改造：
  - `event_memory_enabled` → 组装 frame_ctx（sample_token/scene_token/image_path/feature_path/ego_state/
    perception_objects/scene_result/parsed/admission_score/event_type/scene_tags/risk_tags/timestamp/...）→
    `self._event_manager.on_frame(admission_result, frame_ctx)` → 若返回 (record, feature) 则 `add_record`。
    （event 模式下不逐帧 add frame_memory。）
  - else → 阶段 2 frame_memory（现有 i 块逻辑）。
- `step`：scene 切换检测（`kf["scene_token"] != self._prev_scene_token`）→ `flush()` → add_record。
- `close`：`flush()` 残留事件。
- jsonl：新增 `memory_type` / `event_id`（本帧入库的记忆类型与事件 id；非事件帧为 frame_memory/空）。

---

## 4. `mid_term_memory.py` 改动

- `search`：候选 dict 加 `memory_type`；重排时 `prefer_event_memory` 给 event_memory 加 bonus；
  结果含 `memory_type`。
- （`add_record` 不变：event_memory 与 frame_memory 都走同一入口，feature 分别为 peak 特征 / 帧特征。）

---

## 5. config（`memory.yaml -> mid_term.event_memory`）

```yaml
event_memory:
  enabled: true
  prefer_event_memory: true          # 检索优先返回 event_memory（重排加成）
  event_memory_bonus: 0.10           # event_memory 的 value_aware_score 加成
  max_event_length_frames: 10        # 事件最长帧数（达此强制结束）
  min_event_length_frames: 2         # 事件最短帧数（不足则丢弃，过滤孤立抖动）
  event_end_patience_frames: 2       # 连续非高价值帧达此值→事件结束
  keyframes_per_event: 3             # 每事件关键帧数（start/peak/end）
  store_frame_memory_when_event_disabled: true  # event_memory 关闭时是否退化为逐帧 frame_memory
```

---

## 6. 验证（mulmem + 真实 FAISS）

**事件逻辑（纯逻辑，event_manager.on_frame 序列，不需 faiss）**：
- 连续 3 高价值帧（lane_change）→ 2 非高价值帧 → finalize 1 个 event_memory（keyframes=start/peak/end，
  4 摘要非空，event_type=lane_change，anchor=peak）。
- 短事件（1 高价值帧 + 非高价值）→ `len<min` 丢弃（无记录）。
- `max_event_length` 达上限 → 强制 finalize。
- scene 切换 → flush finalize。

**端到端（真实 FAISS）**：
- event_memory 入 FAISS（peak 特征）→ 检索能返回 event_memory；`prefer_event_memory` bonus 生效
  （event_memory 排序优先于同分 frame_memory）。
- 混合库（event_memory + 旧 frame_memory）检索兼容。
- `enabled=false` → 逐帧 frame_memory（回归阶段 2）。

---

## 7. 风险点

1. 事件边界启发式（patience/max_length/scene）需实测调参。
2. 摘要是确定性模板非 VLM（信息量有限，可后续升级为 VLM 总结）。
3. peak 特征从 `feature_path` 重加载（每事件一次 IO，可接受；若文件缺失则该事件不入 FAISS，记 warning）。
4. 短事件丢弃可能丢失孤立高价值帧（`min_event_length=1` 可保留，文档明示）。
5. 事件 event_type 取 peak 帧，跨类型复合事件（如 stop_and_go）归类可能不准（v1 不特殊处理）。
6. event_memory 与阶段 3 淘汰/阶段 4 检索兼容（memory_type 区分；event_type 高价值自然高分）。
7. 完整 demo 未实跑（需 VLM API）；事件/检索核心逻辑用真实 FAISS 单测覆盖。

---

## 8. 文件改动清单

- 新增 `src/vla_memory/memory/event_memory.py`（`EventMemoryManager`）
- 改 `src/vla_memory/schemas/memory.py`（+13 个 event 专属字段，默认值）
- 改 `src/vla_memory/pipeline/online_loop.py`（`__init__`/`setup`/`step` i 块 event 模式/scene flush/`close`/jsonl）
- 改 `src/vla_memory/memory/mid_term_memory.py`（`search` 加 `memory_type` + prefer bonus）
- 改 `config/memory.yaml`（`mid_term.event_memory` 块）
- 改 `docs/memory_design.md`（+§3.11 事件级记忆）+ `README.md`（§9.2）
- 新增 `docs/stage5_event_memory_design.md`（本设计）
