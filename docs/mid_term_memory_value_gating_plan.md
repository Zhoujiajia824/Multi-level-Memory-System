# 中期记忆价值门控改造设计（Value-Gated Event Memory）

> 本文档为 **Phase 0 审计 + 设计**，不改动任何代码、不改变当前运行结果。后续分阶段实施（见 §7），
> 每阶段保持 demo 可运行。

## 0. 阶段目标与范围

把中期记忆从「逐帧无脑保存」升级为「价值门控的事件经验记忆系统」（带容量上限 + 价值淘汰）。
**本阶段（Phase 0）只做审计与设计**，不重构核心逻辑、不改变当前运行结果。

---

## 1. 当前中期记忆现状（审计）

### 1.1 写入流程
- 唯一写入入口：`OnlineDrivingLoop.step()` 的 **`self.mid_term.add_record(mt_record, feature=feature)`**
  （`src/vla_memory/pipeline/online_loop.py:478`），全库仅此 1 处调用（grep 确认）。
- `mt_record` 构造：`online_loop.py:465-477`，12 字段 `MidTermMemoryRecord`。
- `MidTermMemory.add_record(record, feature)`（`src/vla_memory/memory/mid_term_memory.py:90`）：
  - `self._records[record.record_id] = record`（`:97`，无条件写入 dict）；
  - 若 `feature is not None`：`self.faiss_store.add(feature.reshape(1,-1), [record_id])`（`:102-103`）；
  - **没有任何价值判定、没有容量上限、没有去重、没有淘汰**——每一帧 `memory_on` 都会被保存。

### 1.2 检索流程
- 入口：`MemoryRetriever.retrieve(use_mid_term=...)`（`src/vla_memory/memory/retrieval.py`）→
  `MidTermMemory.search(...)`（`mid_term_memory.py`，6 路联合评分）。
- 评分：`final_score = 0.40·visual + 0.10·text + 0.20·scene + 0.10·weather + 0.10·nav + 0.10·state`
  （权重来自 `config/memory.yaml -> mid_term.weights`）；按总分降序取 top-3。
- 返回 `mid_term_results`（含 record + final_score + sub_scores），渲染进决策 prompt。

### 1.3 持久化格式
- 3 文件（`outputs/memory_db/`）：`mid_term_faiss.index`（FAISS 二进制）+
  `mid_term_faiss.ids.json`（位置→record_id 映射）+ `mid_term_meta.json`（每条记录元数据 dict）。
- 开关：`memory.yaml -> mid_term.persistence.{enabled, save_on_close, auto_load_on_init, strict_load}`。
- `auto_load_on_init=true` 时跨次累积；`strict_load=false` 加载失败仅 warning。

### 1.4 关键代码位置（file:line，后续改动锚点）
| 角色 | 位置 |
|---|---|
| 中期检索（决策前） | `online_loop.py:330-340` |
| 决策 VLM 调用 | `online_loop.py:377-391` |
| jsonl 落盘 | `online_loop.py:431` |
| 短期 push（`if self.use_memory:` 外层 gate） | `online_loop.py:445-461` |
| **中期 mt_record 构造 + add_record** | **`online_loop.py:465-478`** |
| `add_record` 实现（无条件写 dict + FAISS） | `mid_term_memory.py:90-105` |
| `_records` 无界 dict | `mid_term_memory.py:97` |
| FAISS store add/search/save/load | `memory/faiss_store.py` |
| MemoryRecord 12 字段 | `schemas/memory.py`（MidTermMemoryRecord） |
| memory.yaml mid_term 块 | `config/memory.yaml` |

---

## 2. 当前存在的问题
1. **每帧无脑入库**：巡航/稳定停车/红灯等待等低价值帧大量涌入，稀有高价值场景被淹没，
   中期记忆的"经验复用"价值被稀释。
2. **无容量上限**：`_records` 单调增长（README §9.2 第 807 行原文"库无界增长"），长期/大规模
   运行内存与检索延迟膨胀。
3. **无淘汰**：无法丢弃低价值/过期记忆。
4. **检索无价值感知**：top-3 仅按相似度，不区分记忆质量（fallback 帧、低价值帧同样会被召回）。
5. **文档与实现脱节**：`docs/memory_design.md §3` 写的是"4 路+recency+阈值 0.5"，实际是"6 路无阈值"，
   改造时需一并校准。

---

## 3. 新增模块规划（后续阶段实施，本阶段仅设计）

### 3.1 `MemoryValueScorer`（价值打分器）
- 位置：新增 `src/vla_memory/memory/value_scorer.py`。
- 输入（写入点 `online_loop.py:465` 作用域内全部可得，见审计 §1.4）：
  `parsed(behavior/risk_level/target_speed)`、`scene_result(scene_id/weather/traffic_density/risk_factors/vehicles/pedestrians/traffic_lights/intersections)`、
  `ego_state(speed/accel/yaw_rate/steering/brake)`、`perception_objects(oracle, 可选)`、
  `prev_behavior`、`prev_ego_state`、`fallback_used`。
- 输出：`ValueScore{score: float, tags: List[str], reasons: List[str]}`。
- 规则（全部配置驱动，权重/阈值在 `memory.yaml`）：
  **高价值命中（任一即 admit）**：
  - `scene_id ∈ {lane_change, obstacle_avoidance, intersection, turning, merge, crosswalk}`（变道/避障/路口/转弯/汇入/人行横道）
  - `behavior ∈ {CHANGE_LANE_LEFT, CHANGE_LANE_RIGHT, AVOID_OBSTACLE, TURN_LEFT, TURN_RIGHT, YIELD}`
  - **决策变化**：`behavior ≠ prev_behavior`
  - **起步**：`prev_speed < stop_thr 且 cur_speed ≥ move_thr`
  - **急停急起/急转**：`|Δspeed|≥thr 或 |Δaccel|≥thr 或 |Δyaw_rate|≥thr`
  - `traffic_density == high`、`risk_level == high`
  - **cut-in**：`vehicles` 有 `motion=approaching` 且 `relative_position ∈ {front_left,front_right}` 且 `distance<thr`
  - **鬼探头/行人风险**：`pedestrians` 有 `intent=crossing` 且 `distance<thr`；或 `perception_objects` 有 `category∈{pedestrian,cyclist}` 近距且 `speed>thr`
  - **信号灯变化**：`traffic_light.state` 跨帧变化
  **低价值（默认拒绝）**：`behavior∈{KEEP_LANE,FOLLOW}` 且 `risk=low` 且 `traffic_density∈{low,unknown}` 且无突变且无 cut-in/行人；
  连续 `STOP` 且 `speed≈0`（稳定等待，首个 STOP 视为急停→高价值）。
  **降权**：`fallback_used=True`（决策质量低）。
- **决策变化检测的数据缺口**：`ShortTermMemoryItem` 当前无 `behavior` 字段（11 字段，见审计）。
  → 需给 `ShortTermMemoryItem` 增 `behavior/risk_level/target_speed`（Phase 1），并在 `online_loop.py:447`
  short_term.add 时传入当前 `parsed`。**注意顺序陷阱**：push(447) 早于 add(478)，取上一帧需
  `short_term.get_latest(2)[0]`。

### 3.2 `MemoryAdmissionController`（写入门控）
- 位置：新增 `src/vla_memory/memory/admission.py`。
- 注入点：**`online_loop.py:465` 构造 `mt_record` 之后、`add_record(:478)` 之前**。
- 逻辑：调 `MemoryValueScorer` → 若 `score ≥ threshold` 或命中任一 hard 高价值规则 → admit（写 mt_record + value_score/tags）；
  否则 **跳过 add_record**（不入库）。全程仅在 `if self.use_memory:` 内，memory_off 天然不受影响。

### 3.3 `MemoryEvictionManager`（容量淘汰，soft delete + rebuild）
- 位置：新增 `src/vla_memory/memory/eviction.py`。
- 注入点：**`mid_term_memory.py:90 add_record` 内**，写入后若 `size ≥ capacity·high_watermark` 触发。
- 策略：淘汰 `value_score` 最低的 bottom K%（可选：保护最近 N 帧不淘汰，防"刚写即删"）。
- 技术方案见 §6（FAISS flat 不支持原生删除 → soft delete + rebuild）。

### 3.4 检索价值感知（Phase 3）
- `MidTermMemory.search` 过滤已 soft-delete 的记录；可选按 `value_score` 做重排（`final_score' = α·sim + β·value`）。

---

## 4. 新增配置规划（`config/memory.yaml -> mid_term` 下新增块，全部注释）
```yaml
mid_term:
  # ... 现有 top_k/weights/persistence 不变 ...
  value_gating:
    enabled: true                 # 总开关；false 则退化为当前"逐帧全存"行为（向后兼容）
    admission_threshold: 0.5      # 价值分阈值（0-1），≥ 即入库
    hard_rules_admit: true        # 命中任一 hard 高价值规则直接 admit（无视阈值）
    scorer_weights:               # 各信号权重（配置驱动，禁硬编码）
      scene_type: 0.2
      behavior_type: 0.2
      behavior_change: 0.2
      ego_mutation: 0.15
      traffic_density: 0.05
      risk_level: 0.1
      cut_in: 0.15
      pedestrian_risk: 0.15
      fallback_penalty: -0.2
    thresholds:                   # 连续信号阈值
      hard_brake_accel: 2.0       # m/s²
      hard_accel_accel: 2.0
      yaw_rate_change: 0.3        # rad/s
      stop_speed: 0.5             # m/s
      move_speed: 1.0             # m/s
      cut_in_distance: 15.0       # m
      pedestrian_distance: 10.0   # m
  eviction:
    enabled: true
    capacity: 5000                # 记忆条数上限
    high_watermark: 0.95          # 达到容量×该比例触发淘汰
    evict_bottom_ratio: 0.1       # 每次淘汰最低价值的 10%
    protect_recent_frames: 50     # 保护最近 N 帧不被淘汰（0=不保护）
    rebuild_on_evict: true        # soft delete 后重建 FAISS 索引
```

---

## 5. 新增 metadata 字段规划
- **`MidTermMemoryRecord`（`schemas/memory.py`）新增**（向后兼容，旧记录加载时填默认值）：
  - `value_score: float = 0.0`
  - `value_tags: List[str] = []`（如 `["lane_change","behavior_change"]`）
  - `admitted_at: Optional[int] = None`（时间戳）
  - `access_count: int = 0`、`last_accessed: Optional[int] = None`（检索命中时 +1，供后续价值刷新）
  - `is_active: bool = True`（soft delete 标记，False=逻辑删除待 rebuild）
- **`ShortTermMemoryItem`（`schemas/memory.py`）新增**（解锁"决策变化"检测）：
  - `behavior: str = ""`、`risk_level: str = ""`、`target_speed: Optional[float] = None`
  - 写入处 `online_loop.py:447` short_term.add 一并传入当前 `parsed`。
- **持久化兼容**：`mid_term_meta.json` 反序列化时对缺失的新字段填默认值（pydantic 模型默认值即兜底，
  旧 meta.json 可正常加载，不破坏 `auto_load_on_init`）。

---

## 6. FAISS 删除能力与淘汰技术方案
- 当前 `faiss_store.py` 用 `IndexFlatIP`（flat 内积索引）+ 旁挂 `.ids.json`（位置→record_id）。
- **`IndexFlatIP` 不支持原生 `remove_ids`**（仅 `IndexIVF`/`IndexIDMap2` 等支持）。
- **采用 soft delete + rebuild**（用户指定方案）：
  1. 淘汰时在 `_records` 里把目标记录 `is_active=False`（逻辑删除，tombstone）；
  2. 触发 `rebuild`：从 `_records` 中 `is_active=True` 的记录重新构造 FAISS 索引（新建 IndexFlatIP +
     批量 add + 重写 `.ids.json`），并同步 `mid_term_meta.json`；
  3. rebuild 在 `add_record` 内同步触发（数据量小，<1s 可接受）或按 `eviction.rebuild_on_evict`。
- **不改 FAISS 索引类型**（保持 IndexFlatIP / 768d / 现有检索逻辑），淘汰对检索透明（search 时过滤 `is_active=False`）。

---

## 7. 后续阶段开发顺序（每阶段保持 demo 可运行）
- **Phase 1（核心）**：`MemoryValueScorer` + `MemoryAdmissionController` + `MemoryRecord/ShortTermMemoryItem` 新字段
  + `memory.yaml value_gating` 配置 + online_loop 注入（`enabled=false` 时完全等价现状）。验收：低价值帧不再入库，高价值帧正常入库；`enabled=false` 行为零变化。
- **Phase 2**：`MemoryEvictionManager` + 容量上限 + soft delete + rebuild + `eviction` 配置。
- **Phase 3**：检索价值感知（过滤 inactive + 可选 value 重排）。
- **Phase 4（可选/未来）**：高价值中期记忆 → 长期记忆**候选**（只产出候选，**不自动覆盖**长期规则库）。
- 每阶段完成后输出：修改文件列表、核心逻辑、运行命令、验证命令、风险点。

---

## 8. 先读后写约束保护说明
- **当前顺序**（审计确证）：检索 `online_loop.py:330` → 决策 `:377` → jsonl `:431` → 短期 push `:447` → **中期 add `:478`**。
  第 i 帧检索时 `_records`/FAISS 只含 [0,i-1]；`add_record` 全库唯一调用点。
- **改造后不变**：价值门控与淘汰都只在 `:465-478` 写入路径内生效，**绝不前移到检索之前**；
  `MemoryValueScorer` 只读当前帧 + 已 push 的短期记忆（历史），**绝不读未来帧**；淘汰（soft delete + rebuild）
  在 `add_record` 内同步发生，不影响本帧已完成的检索结果。检索阶段只过滤 `is_active`，不改时间顺序。
- **因果性红线**：速度/加速度/状态突变等"跨帧"信号仅用当前+历史（short_term.get_latest(2)），禁止读未来。

---

## 9. memory_on / memory_off 公平性保护说明
- 门控与淘汰逻辑**全部嵌在 `online_loop.py:445` 的 `if self.use_memory:` 块内**（memory_off 完全跳过 `add_record`，审计 §1.4 已证）。
- memory_off 路径**不经过** `MemoryValueScorer`/`AdmissionController`/`EvictionManager`，对照基准保持纯净。
- **数据/感知/场景序列/ego/导航对两种模式完全一致**（同一 keyframes、同一 mosaic/oracle 输入、同一 scene_result、同一 ego_state）；
  唯一差异是 memory_on 启用三层记忆辅助决策——这是实验设计本意，价值门控不破坏该对照。
- 价值门控 `enabled=false` 时，memory_on 行为与当前逐帧全存完全一致（向后兼容回归基线）。

---

## 10. 验收标准（Phase 0）与风险点
**验收**：① demo 原有运行方式不受影响；② 中期记忆写入/检索行为零变化（本阶段不改代码）；
③ 新增设计文档 `docs/mid_term_memory_value_gating_plan.md`；④ 后续关键文件/函数已标注（见 §1.4、§3）；
⑤ 输出下一阶段建议修改点（见 §11）。

**风险点**：① 价值阈值/权重需实测调参（Phase 1 后用 jsonl 离线复盘 value_score 分布）；
② `ShortTermMemoryItem` 加字段后，旧持久化（若有短期记忆导出）需兼容——短期记忆默认不持久化，风险低；
③ FAISS rebuild 在超大库（数万条）时延迟上升，Phase 2 需评估（当前 demo 规模无忧）；
④ `docs/memory_design.md §3` 本就与实现脱节，Phase 1 起顺带校准。

---

## 11. 下一阶段（Phase 1）建议修改点
1. 新增 `src/vla_memory/memory/value_scorer.py`（`MemoryValueScorer`，规则 + 权重全配置驱动）。
2. 新增 `src/vla_memory/memory/admission.py`（`MemoryAdmissionController`）。
3. 改 `src/vla_memory/schemas/memory.py`：`MidTermMemoryRecord` +value_score/value_tags/admitted_at/access_count/last_accessed/is_active；
   `ShortTermMemoryItem` +behavior/risk_level/target_speed。
4. 改 `src/vla_memory/memory/mid_term_memory.py`：`add_record` 接受已打分的 record；`search` 过滤 `is_active=False`。
5. 改 `src/vla_memory/pipeline/online_loop.py:447`（short_term.add 传 parsed）+ `:465-478`（注入 AdmissionController，
   `enabled=false` 时直通当前行为）。
6. 改 `config/memory.yaml`：新增 `mid_term.value_gating` 块。
7. 改 `README.md §9.2` + `docs/memory_design.md §3`：同步价值门控语义 + 校准 6 路评分。
8. 验证：`value_gating.enabled=false` 回归（行为不变）；`enabled=true` 跑 1 场景，统计入库率（应显著 <100%）、
   高价值场景（变道/路口/急停）是否 100% 入库、巡航帧是否被拒。
