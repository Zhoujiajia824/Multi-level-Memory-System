# 阶段 3 设计：容量上限与价值淘汰 MemoryEvictionManager

> 本文档为阶段 3（容量管理 + 价值淘汰 + soft delete + FAISS rebuild）的实现设计。
> 承接阶段 1（metadata v0.2）、阶段 2（价值门控写入 admission）。设计总览见
> `docs/mid_term_memory_value_gating_plan.md` §3.3/§6。

## 0. 目标

中期记忆设容量上限；接近上限时按价值淘汰低价值记忆（soft delete），高价值/长尾/高风险受保护；
检索过滤 inactive；inactive 比例高时 rebuild FAISS 物理压缩。`_records` 保留 inactive 元数据
（不物理删除），仅 FAISS 索引在 rebuild 时剔除。

---

## 1. 关键设计决策

### 1.1 FAISS rebuild 用 `reconstruct_n`
`IndexFlatIP` 是 flat 索引（原地存向量），不支持原生 `remove_ids`，但支持 `reconstruct_n(0, ntotal)`
取回全部向量。rebuild 流程：reconstruct 全部 → 过滤出 active 的 (vector, id) → `_init_index()`
新建空 IndexFlatIP → `add(keep_vecs, keep_ids)` → 重写 `_ids`。`_records` 保留 inactive 元数据。

### 1.2 容量按 active 记录数
新增 `MidTermMemory.active_size()`（`is_active=True` 计数）。触发条件（任一）：
- `active_count >= max_records · eviction_trigger_ratio`（常规）
- `active_count >= max_records · emergency_trigger_ratio`（紧急，强制淘汰到 target）
- `estimated_disk_mb >= max_disk_mb · eviction_trigger_ratio`（磁盘，**估算**：`active_count · (feature_dim·4 + 2048)` 字节）

淘汰目标：`active_count <= max_records · eviction_target_ratio`。`max_disk_mb` 为估算（真磁盘在 save 时才精确可知，文档明示）。

### 1.3 价值评分 `MemoryValueScorer`（对存量记忆打分）
区别于阶段 2 的 admission（写入时价值）：本评分对**已存记忆**算"持续价值"`memory_value_score`，
作为淘汰排序依据。在 eviction 前对全部 active 记忆重算并写回（同时更新 `recency_score` /
`retrieval_utility` / `confidence_score` / `redundancy_score`）。

`memory_value_score = 0.25·admission + 0.20·event_highvalue + 0.15·recency + 0.15·retrieval_utility
 + 0.10·(1−redundancy) + 0.10·confidence − 0.05·conflict − 0.10·lowvalue_penalty`（权重配置驱动）。
- `event_highvalue`：event_type/scene_tags/risk_tags 命中高价值/长尾/高风险 → 高。
- `recency`：`last_retrieved_at` 优先，否则 `created_at`，归一化到 [0,1]（越近越高）。
- `retrieval_utility`：`hit_count` 归一化（被多次检索→高）。
- `redundancy`：按 `(scene_id, event_type)` 分组频率近似（过度代表→冗余→低价值）；避免 O(n²)。
- `confidence`：数据质量——`behavior==UNKNOWN` 或 `trajectory` 空 → 低；否则高。
- `conflict`：`conflict_count` 归一化（高→扣分）。
- `lowvalue_penalty`：event_type ∈ {normal_cruise, stable_stop} → 1（扣分）。

### 1.4 检索命中追踪（为 recency / retrieval_utility 供数）
`MidTermMemory.search` 增 `now_ts` 参数；对返回的 top_k 候选 `hit_count += 1`、`last_retrieved_at = now_ts`。
**这是元数据簿记**：不新增记忆、不改当前检索结果、不破坏先读后写（检索结果已算完才更新）。
`retriever.retrieve` 透传 `now_ts`，`online_loop.step` 传 `frame_ts`。`now_ts=None` 时不更新（向后兼容）。

### 1.5 集成：依赖注入
`online_loop.setup` 构建 `MemoryValueScorer` + `MemoryEvictionManager` + `MemoryCompactionManager`，
通过 `mid_term.set_eviction_manager / set_value_scorer / set_compaction_manager` 注入。
`MidTermMemory.add_record` 末尾：`if self._eviction_manager: self._eviction_manager.after_add(self)`。
`after_add` 内按容量决定是否触发 `evict`（可能再触发 `compaction.maybe_rebuild`）。eviction.py 不在模块顶层
import mid_term_memory（避免循环），通过传入的 mid_term 实例调用其公开方法。

### 1.6 保护策略
淘汰时跳过：
- `protect_long_tail`：event_type==long_tail 或 memory_novelty 高（用 rarity_score / event_type 判）
- `protect_high_risk`：risk_tags 非空（cut_in/pedestrian/hard_brake/ghost 等）
- `protect_recent_high_value`：admission_score 高 **且** recency 高（近期写入的高价值）
- `min_keep_per_event_type`：每类 event_type 至少保留 N 条（已保留不足 N → 跳过淘汰该条）

### 1.7 soft delete
`is_active=False`、`status="deleted"`、`deleted_reason`（如 `"low_value_eviction"` / `"normal_cruise_eviction"`）、
`deleted_at=帧时间戳`。`_records` 保留。rebuild 时 FAISS 剔除。检索过滤。

---

## 2. schema 改动（`schemas/memory.py`）

`MidTermMemoryRecord` 新增（默认值，向后兼容，旧 meta.json 自动补默认）：
- `recency_score: Optional[float] = None`
- `deleted_at: Optional[int] = None`

（`is_active` / `status` / `memory_value_score` / `hit_count` / `successful_hit_count` / `failed_hit_count` /
`last_retrieved_at` / `conflict_count` / `redundancy_score` / `retrieval_utility` / `confidence_score` /
`deleted_reason` 阶段 1 已有。）

---

## 3. 新增模块

### `src/vla_memory/memory/value_scorer.py` — `MemoryValueScorer`
- `score_all(mid_term, now_ts)`：遍历 active 记录，算 `memory_value_score` + 4 子分，写回 record。
- `score_record(record, now_ts, redundancy_ctx)`：单条打分（纯逻辑，可单测）。
- 配置驱动权重 + 阈值（recency 半衰期、redundancy 归一化等）。

### `src/vla_memory/memory/eviction.py`
- `MemoryEvictionManager`：
  - `after_add(mid_term)`：检查容量 → `evict(mid_term, now_ts)`。
  - `evict(mid_term, now_ts)`：调 `value_scorer.score_all` → 取 active 按 `memory_value_score` 升序 →
    逐条判保护 → `mid_term.soft_delete(rid, reason, now_ts)` 直到 `active_count <= target`。返回 `[(rid, reason)]` 记日志。
  - `emergency` 路径：紧急触发时放宽保护（仅保留 `min_keep_per_event_type` 与最高分），淘汰到 target。
- `MemoryCompactionManager`：
  - `maybe_rebuild(mid_term)`：`inactive_ratio >= rebuild_faiss_when_inactive_ratio` 或 `rebuild_after_eviction` → `mid_term.rebuild_index()`。

---

## 4. `MidTermMemory` 改动（`mid_term_memory.py`）

- `active_size()` / `get_active_records()` / `inactive_size()`
- `soft_delete(record_id, reason, deleted_at)`：设 `is_active=False`/`status="deleted"`/`deleted_reason`/`deleted_at`；从 `_text_corpus` 移除（检索不再用其文本）
- `rebuild_index()`：`faiss_store` reconstruct_n → 过滤 active → 重建 IndexFlatIP + `_ids`（inactive 元数据保留在 `_records`）
- `set_eviction_manager` / `set_value_scorer` / `set_compaction_manager`
- `add_record` 末尾：`if self._eviction_manager: self._eviction_manager.after_add(self)`
- `search` 增 `now_ts` 参数：候选循环 `if not record.is_active: continue`（过滤 inactive）；返回前对 top_k `hit_count+=1`、`last_retrieved_at=now_ts`（`now_ts` 非 None 时）
- `size()` 仍返回全部（含 inactive）；新增 `active_size()`

---

## 5. `retrieval.py` / `online_loop.py` 改动

- `MemoryRetriever.retrieve(..., now_ts=None)`：透传给 `mid_term.search(..., now_ts=now_ts)`（仅 use_mid_term 时）。
- `online_loop.setup`：建 scorer/eviction/compaction，注入 mid_term。
- `online_loop.step`：`retriever.retrieve(..., now_ts=frame_ts)`。
- `online_loop.close`：中期记忆 save_on_close 已有（inactive 元数据一并落盘，FAISS 已 rebuild 则落盘压缩后索引）。

---

## 6. config（`memory.yaml -> mid_term.capacity / .eviction / .compaction`，全注释）

```yaml
capacity:
  enabled: true
  max_records: 5000
  max_disk_mb: 2048
  eviction_trigger_ratio: 0.80
  eviction_target_ratio: 0.70
  emergency_trigger_ratio: 0.95
eviction:
  strategy: "value_based_soft_delete"
  protect_long_tail: true
  protect_high_risk: true
  protect_recent_high_value: true
  min_keep_per_event_type: { lane_change: 50, intersection: 50, obstacle_avoidance: 50,
                             pedestrian_interaction: 50, cut_in: 30, ghost_probing_risk: 30,
                             normal_cruise: 10, stable_stop: 5, frame_memory: 100 }
  weights: { admission: 0.25, event_highvalue: 0.20, recency: 0.15, retrieval_utility: 0.15,
             redundancy: 0.10, confidence: 0.10, conflict: 0.05, lowvalue_penalty: 0.10 }
  thresholds: { recency_half_life_frames: 500, high_value_admission: 0.7 }
compaction:
  rebuild_faiss_when_inactive_ratio: 0.20
  rebuild_after_eviction: false
```

---

## 7. 验证

**离线（value_scorer + eviction 决策/soft_delete，不依赖 faiss）**：
- 合成若干 active 记忆（含高价值 lane_change/cut_in + 低价值 normal_cruise/stable_stop），scorer 打分 →
  memory_value_score 高价值>低价值。
- 设 max_records=5、trigger 0.8、target 0.7，add 到超限 → evict 触发 → active 降到 target 附近 →
  低价值被 soft_delete（is_active=False/status=deleted/deleted_reason 非空），高价值保留。
- 保护：long_tail/high_risk/recent_high_value 不被淘汰；min_keep_per_event_type 生效。
- 返回 `[(rid, reason)]` 日志正确。

**需 faiss（rebuild + search 过滤，本机 faiss 未装）**：
- search 不返回 inactive；rebuild 后 `faiss_store.size() == active_size()`。
- 小规模测试：`config/memory.yaml` 设 `capacity.max_records: 5`，跑 `scripts/07_run_full_demo.py --mode memory_on --max-frames 20`，
  检查 `mid_term_meta.json` 中 is_active=False 的记录数 > 0、active 记录数 <= 5·target。

---

## 8. 风险点

1. `reconstruct_n` 依赖 `IndexFlatIP`（flat 支持重建）；若未来换 `IndexIVF` 需改 rebuild 实现。
2. `max_disk_mb` 为估算（真磁盘 save 时才精确），文档明示。
3. 冗余用 `(scene_id,event_type)` 频率近似，非真实视觉冗余（v1 取舍，避免 O(n²)）。
4. hit tracking 在 search 中 mutate 元数据（不破坏先读后写，但仅 close 时持久化；中途崩溃丢失统计——可接受）。
5. 阈值/权重需实测调参。
6. `successful_hit_count`（检索是否真帮到决策）无法离线测量，保持 0；`retrieval_utility` 用 `hit_count` 近似。
7. eviction 在 `add_record` 内同步触发（数据量小可接受；超大库需评估延迟，当前 demo 规模无忧）。

---

## 9. 文件改动清单

- 新增 `src/vla_memory/memory/value_scorer.py`（`MemoryValueScorer`）
- 新增 `src/vla_memory/memory/eviction.py`（`MemoryEvictionManager` + `MemoryCompactionManager`）
- 改 `src/vla_memory/schemas/memory.py`（+`recency_score` / +`deleted_at`）
- 改 `src/vla_memory/memory/mid_term_memory.py`（`active_size`/`soft_delete`/`rebuild_index`/`set_*`/`add_record` hook/`search` 过滤+命中+`now_ts`）
- 改 `src/vla_memory/memory/retrieval.py`（`retrieve` +`now_ts`）
- 改 `src/vla_memory/pipeline/online_loop.py`（setup 注入 + step 传 `now_ts`）
- 改 `config/memory.yaml`（`capacity`/`eviction`/`compaction`）
- 改 `docs/memory_design.md`（+§3.9 容量管理与 soft delete）
- 改 `README.md`（§9.2 补充容量管理）
- 新增 `docs/stage3_eviction_design.md`（本设计）
