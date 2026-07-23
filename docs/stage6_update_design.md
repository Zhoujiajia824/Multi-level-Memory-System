# 阶段 6 设计：冲突感知更新 MemoryUpdateManager

> 本文档为阶段 6（"改"——冲突感知更新）的实现设计。承接阶段 1-5。检索到相似记忆但当前决策/
> 后验/安全评价与旧记忆冲突时，对旧记忆降权/标记 deprecated/superseded/增 conflict_count，
> 新记忆作为新版本或替代写入。**不物理删除**，unsafe 新证据不覆盖安全旧记忆。

## 0. 目标

中期记忆支持"改"：冲突检测 → 分类（policy/style/context/unsafe_old/unsafe_new）→ 软更新
（降权/标记/版本化）+ `update_history` 记录。`update.enabled=false` 退化为阶段 5 行为（无冲突更新）。

---

## 1. 关键设计决策

### 1.1 冲突检测时机：online_loop.step 写入(i)之后
检索(step b)给出相似旧记忆；决策(step e)给出当前 behavior/risk；在写入(i)之后调
`update_manager.process(current_ctx, mid_term_results, new_record, now_ts)`。此时新记忆已入库，
可建立版本链（old.superseded_by → new.memory_id；new.previous_versions += old.memory_id）。
仅在 `if self.use_memory:` 内（memory_off 不检索、不更新）。不前移到检索前，先读后写不变。

### 1.2 冲突分类（5 类，按优先级）
对每条 `final_score >= min_similarity_for_conflict_check` 的检索旧记忆，比较当前帧 vs 旧记忆：
- **context_mismatch**：导航目标不同 或 scene_id 不同 → 不视为冲突（不同情境），不更新。
- **unsafe_new_evidence**：当前决策不安全（risk=high 或 fallback/parser 失败）且旧安全 → 不覆盖旧；
  新记忆（若已写）标 `low_confidence`。
- **unsafe_old_memory**：旧记忆 risk=high 且当前安全 → 旧标记 `deprecated`（或 `superseded` 若有新）。
- **policy_conflict**：同情境同 scene，behavior 跨"战略类别"不同（cruise↔lateral/turn/avoid）→
  conflict_count++、confidence 衰减；达 `supersede_after_conflicts` → `superseded`。
- **style_conflict**：同情境同 scene，behavior 同类别（都 cruise，或都 lateral）但不同 → 风格变体，
  两条都保留，不衰减不删除。
- behavior 相同且同情境 → 无冲突，不更新。

behavior 类别：cruise={KEEP_LANE,FOLLOW,SLOW_DOWN,STOP} / lateral={CHANGE_LANE_*} /
turn={TURN_*} / avoid={AVOID_OBSTACLE,YIELD}。跨类别=policy，同类别=style。

### 1.3 软更新（不物理删除）
所有更新只改字段：`conflict_count`/`last_conflict_at`/`conflict_reasons`/`confidence_score`/
`status`/`superseded_by`/`previous_versions`/`update_history`。`_records` 保留，FAISS 不动
（inactive/deleted 才在 rebuild 剔除；deprecated/superseded 仍可检索，按配置过滤）。

### 1.4 unsafe 保护
`unsafe_new_evidence` 时旧记忆保持 `active`，新记忆标 `low_confidence`（不删除、不覆盖）。
**绝不**用 unsafe 新证据覆盖安全旧记忆。

### 1.5 版本化
supersede 时：`old.status="superseded"`、`old.superseded_by=new.memory_id`、
`new.previous_versions += [old.memory_id]`、双方 `update_history` 记录。
（event 模式下事件未 finalize 时 new_memory_id 未知 → 仅标 old deprecated，不建链，文档明示。）

### 1.6 status 值与检索过滤协调
status 取值：`active`/`low_confidence`/`deprecated`/`superseded`/`inactive`/`deleted`。
检索过滤（`mid_term_memory.search`，阶段 4）：
- `is_active=False`（inactive/deleted）→ **始终过滤**（soft delete 硬约束）。
- `exclude_deprecated=true`（配置，默认 false）→ 过滤 `status in {deprecated, superseded}`。
  （阶段 4 原 `exclude_deprecated` 检查 `superseded_by`，改为检查 status，与阶段 6 对齐。）
- `low_confidence` → 不按 status 过滤（仍可检索，但 confidence_score 低 → 阶段 4 `min_confidence_score`
  与阶段 3 value_scorer 自然降权）。
- `exclude_deleted` 保留（status=deleted，与 is_active=False 互补）。

---

## 2. schema 改动（`schemas/memory.py`，默认值，向后兼容）

`MidTermMemoryRecord` 新增：
- `last_conflict_at: Optional[int] = None`（最近冲突时间 μs）
- `conflict_reasons: List[str] = []`
- `previous_versions: List[str] = []`（该记忆取代的旧 memory_id 列表）
- `update_history: List[dict] = []`（`{action, conflict_type, reason, at, by_new}`）

（`conflict_count`/`confidence_score`/`status`/`superseded_by` 阶段 1 已有。）

---

## 3. 新增模块 `src/vla_memory/memory/update.py`

- `MemoryUpdateConfig`：从 `mid_term.update` 加载（enabled / conflict_detection_enabled /
  versioning_enabled / soft_update_only / min_similarity_for_conflict_check /
  confidence_decay_on_conflict / supersede_after_conflicts）。
- `MemoryUpdateManager`：
  - `process(current_ctx, mid_term_results, new_record, now_ts) → List[dict]`（更新动作，供日志/jsonl）：
    遍历 `final_score >= min_sim` 的检索记忆 → `_classify(current, old)` → `_apply(old, type, new_record, now_ts)`。
  - `_classify(current, old) → str`（5 类 + "none"）。
  - `_apply(...)`：按类型 mutate old（+ new for unsafe_new）。返回 action dict。
  - 行为类别常量 + 安全判定（risk=high 或 fallback/parser 失败）。

---

## 4. `online_loop.py` 改动

- `__init__`：`self._update_manager=None` / `self._update_enabled=False`。
- `setup`：建 `MemoryUpdateManager`（从 `mid_term.update`），缓存 enabled。
- `step`：写入(i)之后新增 `i.1` 块——`if self._update_enabled and self.use_memory:` 调
  `update_manager.process(current_ctx, mid_term_results, new_record, now_ts)`。
  - `current_ctx`：scene_id / behavior / risk_level / nav_instruction / fallback_used / parser_status / timestamp。
  - `new_record`：本帧入库的记忆对象（event 模式=ev_record 若 finalize，frame 模式=mt_record 若 admitted，否则 None）。
  - 动作列表记入 jsonl `memory_update_actions`（可空）。
- jsonl 新增：`memory_update_enabled` / `memory_update_actions`。

---

## 5. `mid_term_memory.py` 改动（检索过滤对齐）

- `search` 的 `exclude_deprecated` 过滤：从检查 `superseded_by` 改为检查
  `status in {"deprecated", "superseded"}`（与阶段 6 status 对齐）。其余过滤不变。

---

## 6. config（`memory.yaml -> mid_term.update`）

```yaml
update:
  enabled: true
  conflict_detection_enabled: true
  versioning_enabled: true
  soft_update_only: true              # true=只软更新（不改 FAISS/不物理删）
  min_similarity_for_conflict_check: 0.75  # 检索相似度≥此值才做冲突检查
  confidence_decay_on_conflict: 0.10  # 每次冲突 confidence 衰减
  supersede_after_conflicts: 3        # 累计冲突达此值→superseded
```

---

## 7. 验证（mulmem + 真实 FAISS）

**冲突分类与软更新（真实记忆对象 mutate）**：
- policy_conflict：旧 KEEP_LANE / 新 CHANGE_LANE_LEFT，同 scene 同 nav → 旧 conflict_count++、
  confidence 衰减、status 不变（未达 3 次）；达 3 次 → superseded + superseded_by。
- style_conflict：旧 FOLLOW / 新 SLOW_DOWN（都 cruise）→ 两条都保留，无衰减，conflict_reasons 记 style_variant。
- context_mismatch：nav 不同 → 不更新。
- unsafe_new_evidence：新 risk=high → 旧保持 active，新标 low_confidence。
- unsafe_old_memory：旧 risk=high / 新安全 → 旧 deprecated（+ superseded_by 若有新）。
- update_history 非空、previous_versions 链正确。
- 不物理删除（is_active 仍 True for deprecated/superseded/low_confidence）。

**检索过滤对齐（真实 FAISS）**：
- status=deprecated/superseded + exclude_deprecated=true → 不返回；=false → 返回。
- status=deleted/is_active=False → 始终不返回。
- low_confidence → 仍返回（不按 status 过滤）。

**兼容**：`update.enabled=false` → 无冲突更新（阶段 5 行为）。旧记录新字段默认。

---

## 8. 风险点

1. 冲突分类是启发式（behavior 类别 + risk + nav），可能误判（如 style 误为 policy）；阈值/类别可配/可调。
2. "unsafe" 判定仅用 risk=high/fallback/parser 失败代理（无真实事故后验），保守。
3. supersede 累计冲突跨 run（conflict_count 持久化），需多帧/多次才触发；`supersede_after_conflicts` 可调。
4. event 模式事件未 finalize 时无法建 superseded_by 版本链（仅标 old deprecated），文档明示。
5. update 在写入后同步触发（demo 规模无忧）。
6. 多条检索记忆同时冲突 → 各自独立更新（无交叉）。
7. 完整 demo 未实跑（需 VLM API）；冲突/过滤核心逻辑用真实 FAISS 单测覆盖。

---

## 9. 文件改动清单

- 新增 `src/vla_memory/memory/update.py`（`MemoryUpdateManager`）
- 改 `src/vla_memory/schemas/memory.py`（+4 字段：last_conflict_at/conflict_reasons/previous_versions/update_history）
- 改 `src/vla_memory/pipeline/online_loop.py`（`__init__`/`setup`/`step` i.1 块/jsonl）
- 改 `src/vla_memory/memory/mid_term_memory.py`（`search` exclude_deprecated 改查 status）
- 改 `config/memory.yaml`（`mid_term.update` 块）
- 改 `docs/memory_design.md`（+§3.12 冲突感知更新）+ `README.md`（§9.2）
- 新增 `docs/stage6_update_design.md`（本设计）
