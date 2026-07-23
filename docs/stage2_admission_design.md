# 阶段 2 设计：中期记忆价值门控写入 MemoryAdmissionController

> 本文档为阶段 2（价值门控写入）的实现设计，承接 `docs/mid_term_memory_value_gating_plan.md`。
> 阶段 1 已扩展 metadata（v0.2，37 字段）；阶段 2 解决“每帧都存”的核心问题。

## 0. 目标

中期记忆不再逐帧全存：每帧**决策完成后、写入中期记忆之前**，由 `MemoryAdmissionController` 判断该帧是否值得入库。低价值帧（普通巡航 / 稳定停车 / 冗余帧）拒绝写入，高价值事件强制写入。`enabled=false` 时完全退化为阶段 1 的逐帧全存（回归基线）。

---

## 1. 关键设计决策（4 项）

### 1.1 prev-frame 状态：循环实例状态追踪，不改 ShortTermMemoryItem schema
- 决策变化 / 动力学突变需要“上一帧”的 behavior / risk_level / target_speed / ego_state / 对象数。`ShortTermMemoryItem` 当前无这些字段（阶段 1 未加）。
- **方案**：`OnlineDrivingLoop` 维护 `self._prev_frame_ctx`（dict），在 `step()` 末尾更新；admission 读取它作为“历史帧”。
- **否决备选**（设计文档 §5：扩展 `ShortTermMemoryItem` 加 behavior/risk_level/target_speed）：改动面大、`short_term.add`(447) 早于 `mid_term.add`(478) 存在顺序陷阱、短期记忆本就不持久化无收益。

### 1.2 memory_novelty 复用检索结果，不额外查 FAISS
- `memory_result["mid_term_results"][0]["final_score"]` = 当前帧与已有中期记忆的**最大联合相似度**（已在 `step()` 开头检索得到，含 [0,i-1] 帧）。
- `memory_novelty = 1 - max_score`；空库 → 1.0（首帧全新颖）。
- 不读未来、不破坏先读后写、无额外 FAISS 查询开销。

### 1.3 posthoc_outcome_value 只用当前帧代理，绝不读未来
- 用户要求“后验评估信息只能用于运行结束后的 posthoc，不得进入当前帧决策”。阶段 2 在 admission 时无法用未来结果。
- **方案**：该信号（权重 0.05）用**当前帧决策质量代理**——`fallback_used` / parser 失败 / `risk_level=high`（这些是“值得记住的边缘情况”，对应“decision parser 失败”“规则兜底触发”“VLM 输出风险显著升高”）。真后验留给运行后离线分析，**不进入 admission**。代码明确注释此约束。

### 1.4 jerk / yaw_rate 从历史帧差分推导
- `ego_state` 无 `jerk`；`yaw_rate` 仅 CAN bus 路径有（pose-diff 路径为 None）。
- `jerk ≈ (accel_cur - accel_prev) / dt`；`yaw_rate` 缺失时用 `(yaw_cur - yaw_prev) / dt`。`dt` 取帧时间戳差（微秒→秒）。
- 字段名注意：加速度字段是 `acceleration`（非 `accel`）；方向盘是 `steering_angle`（非 `steering`）。

---

## 2. 新增模块 `src/vla_memory/memory/admission.py`

- **`MemoryAdmissionPolicy`**：持有 config（enabled / policy_version / store_all_when_disabled / debug_memory_off / score_threshold / force_store_threshold / weights / low_value_filters / high_value_events / thresholds）。配置驱动，禁硬编码。
- **`MemoryAdmissionResult`**（dataclass）：`should_store` / `admission_score` / `admission_reasons` / `reject_reasons` / `event_type` / `scene_tags` / `risk_tags` / `policy_version` / `signals`(dict，6 信号明细，供离线复盘)。
- **`MemoryAdmissionController`**：`decide(ctx, prev_ctx) -> MemoryAdmissionResult`。纯逻辑、无副作用、无 IO，可离线单测。
- **输入 `ctx`**（由 online_loop 组装）：`parsed`(behavior/risk_level/target_speed/trajectory/fallback_used/parser_status)、`scene_result`(scene_id/traffic_density/risk_factors/vehicles/pedestrians/intersections/traffic_lights)、`ego_state`、`perception_objects`(oracle，防御式兼容 PerceptionObject 实例与 dict)、`max_mid_term_score`、`timestamp`、`nav_instruction`。
- **输入 `prev_ctx`**：上一帧的 behavior/risk_level/target_speed/ego 关键量(speed/acceleration/yaw)/scene_id/object_count/nearest_distance/traffic_density/instance_tokens 集合/timestamp。

---

## 3. 6 信号（各归一化 0~1，配置权重加权求和，权重和=1.0）

| 信号 | 权重 | 计算来源 |
|---|---|---|
| dynamics_surprise | 0.20 | speed/accel/jerk/yaw_rate/yaw_change/target_speed_change 归一化取 max |
| scene_salience | 0.25 | scene_id 高价值映射 + traffic_density=high + intersections.present + pedestrians 非空 + risk_factors 数 |
| perception_change | 0.20 | 对象数变化 + 最近距离缩小 + cut_in + 行人近距 + cyclist 近距 |
| decision_change | 0.15 | behavior 变化 / risk 升级 / target_speed 大变 / 轨迹形态变 / fallback / parser 失败 |
| memory_novelty | 0.15 | 1 - max_mid_term_score（空库=1.0） |
| posthoc_outcome_value | 0.05 | fallback / parser 失败 / risk=high 代理（不读未来） |

`final_score = Σ weight_i · signal_i`。

---

## 4. 事件识别（高价值，命中即 force store；同时填 event_type / scene_tags / risk_tags）

`lane_change` / `start` / `hard_brake` / `hard_acceleration` / `obstacle_avoidance` / `intersection` / `dense_traffic` / `pedestrian_interaction` / `cyclist_interaction` / `cut_in` / `merge` / `turn_left` / `turn_right` / `crosswalk` / `occlusion` / `ghost_probing_risk` / `long_tail`。

映射来源（全部从已确认的真实字段，不伪造）：
- scene_id 枚举：`lane_change/obstacle_avoidance/intersection/merge/crosswalk/turning`
- behavior 枚举：`CHANGE_LANE_*/TURN_LEFT/TURN_RIGHT/AVOID_OBSTACLE`
- traffic_density=`high` → dense_traffic
- intersections.present → intersection
- vehicles{motion=approaching, relative_position∈{front_left,front_right}, distance_m<cut_in_thr} → cut_in
- pedestrians{intent=crossing, distance_m<ped_thr} 或 oracle pedestrian 近距 → pedestrian_interaction
- oracle category=`cyclist` 近距 → cyclist_interaction
- 动力学阈值：prev_speed<stop 且 speed≥move → start；accel<hard_brake_thr → hard_brake；accel>hard_accel_thr → hard_acceleration
- novelty>long_tail_thr 且非巡航 → long_tail
- risk_factors 关键词（“遮挡/盲区”→occlusion，“鬼探头/突发”→ghost_probing）：**best-effort，召回有限，文档明示**

---

## 5. 低价值过滤（force reject 候选，仅在无高价值事件时生效）

- **stable_stop**：连续两帧 speed<stop_thr + risk=low + behavior=STOP + 无事件
- **normal_cruise**：behavior∈{KEEP_LANE,FOLLOW} + risk=low + traffic_density∈{low,unknown} + dynamics_surprise 低 + 感知变化低 + 无事件
- **redundant_frame**：max_mid_term_score>redundant_sim(0.85) + 无事件 + decision_change 低（与近期已写入记忆高度相似且无新决策价值）

---

## 6. 准入决策逻辑

```
if not enabled and store_all_when_disabled:
    should_store = True                       # legacy：阶段1逐帧全存
elif enabled:
    hard = 任一高价值事件命中
    if hard or score >= force_store_threshold(0.80):
        should_store = True                   # 高价值事件 / 高分强制写
    else:
        filter = stable_stop / normal_cruise / redundant_frame 任一命中（前提：无 hard 事件）
        if filter 命中:
            should_store = False              # reject_reasons = [filter 名]
        elif score >= score_threshold(0.55):
            should_store = True
        else:
            should_store = False              # reject_reasons = [score_below_threshold]
```

`event_type`：hard 命中→最高优先级事件名；filter 命中→`stable_stop/normal_cruise/redundant_frame`；否则 `frame_memory`。

---

## 7. config（`memory.yaml -> mid_term.admission`，全部注释，配置驱动）

```yaml
admission:
  enabled: true
  policy_version: "value_gated_v0.1"
  store_all_when_disabled: true
  debug_memory_off: false          # memory_off 是否仍算 admission debug（默认 false，保持对照纯净）
  score_threshold: 0.55
  force_store_threshold: 0.80
  weights: { dynamics_surprise: 0.20, scene_salience: 0.25, perception_change: 0.20,
             decision_change: 0.15, memory_novelty: 0.15, posthoc_outcome_value: 0.05 }
  low_value_filters: { filter_stable_stop: true, filter_normal_cruise: true, filter_redundant_frames: true }
  high_value_events: { lane_change: true, start: true, hard_brake: true, hard_acceleration: true,
                       obstacle_avoidance: true, intersection: true, dense_traffic: true,
                       pedestrian_interaction: true, cyclist_interaction: true, cut_in: true,
                       merge: true, turn_left: true, turn_right: true, crosswalk: true,
                       occlusion: true, ghost_probing_risk: true, long_tail: true }
  thresholds: { speed_change: 2.0, accel_change: 2.0, jerk: 6.0, yaw_rate: 0.3, yaw_change: 0.2,
                target_speed_change: 3.0, stop_speed: 0.5, move_speed: 1.0,
                hard_brake_accel: -3.0, hard_accel_accel: 3.0,
                cut_in_distance: 15.0, pedestrian_distance: 10.0, cyclist_distance: 10.0,
                object_count_change: 3, nearest_distance_shrink: 5.0,
                redundant_sim: 0.85, long_tail_novelty: 0.9 }
```

---

## 8. OnlineDrivingLoop 集成

- `__init__`：`self._prev_frame_ctx = None`、`self._admission_controller = None`。
- `setup()`：从 config 建 `MemoryAdmissionPolicy` + `MemoryAdmissionController`；缓存 `self._admission_enabled`、`self._admission_debug_memory_off`。
- `step()` e.1 块改造（替换阶段 1 的 `mid_term_admit = self.use_memory`）：
  - **memory_on + enabled**：组装 ctx → `controller.decide(ctx, prev_ctx)` → `mid_term_admit = result.should_store`；jsonl 字段取自 result；`mt_record` 用 result 的 `event_type/scene_tags/risk_tags/admission_score/admission_reasons`。
  - **memory_on + disabled**：`mid_term_admit = True`（legacy），reasons=`[admission_disabled_store_all]`。
  - **memory_off**：`mid_term_admit = False`；仅当 `debug_memory_off=True` 才算 admission debug（默认 False）。admission 在决策后运行，本就不影响 memory_off 决策，对照公平。
- `step()` 末尾（add 之后）：更新 `self._prev_frame_ctx`，每帧都更新（含拒绝帧）。
- **先读后写**：admission 只读 当前帧 + memory_result（已检索，[0,i-1]）+ prev_ctx（i-1），不写、不读未来。`add_record` 仍在 step 末尾。

---

## 9. jsonl 新增/完善字段

保留阶段 1：`mid_term_memory_added` / `mid_term_memory_id` / `memory_admission_score` / `memory_admission_reasons` / `memory_record_status`。
新增：`memory_admission_enabled` / `memory_admission_should_store` / `memory_admission_reject_reasons` / `memory_event_type` / `memory_scene_tags` / `memory_risk_tags`。

---

## 10. 文档

- `docs/memory_design.md` 新增 §3.8 价值门控写入机制。
- `README.md` §9.2 补充 admission 说明 + 写入率统计方法。
- `docs/stage2_admission_design.md`（本文件）。

---

## 11. 验证

**离线（不依赖 faiss/API，controller 纯逻辑单测，合成帧序列）**：
- normal_cruise → should_store=False
- 连续 stable_stop → False
- lane_change 帧 → True，event_type=lane_change
- hard_brake 帧(accel<-3) → True
- redundant 帧(sim>0.85) → False
- behavior 变化 → True
- enabled=False → 全 True（legacy）
- memory_off → added=False
- + config 加载 + jsonl 字段两模式取值

**完整端到端（写入率统计，需 faiss+API）**：
```
python scripts/07_run_full_demo.py --mode memory_on --max-scenes 1 --max-frames 5
# 统计写入率 = mid_term_meta.json 记录数 / jsonl 帧数
```

---

## 12. 风险点

1. 阈值/权重需实测调参（首版合理默认，jsonl 的 signals 明细可离线复盘 score 分布再调）。
2. occlusion / ghost_probing 仅关键词 best-effort，召回有限（文档明示）。
3. `perception_objects` 可能是 `PerceptionObject` 实例或 dict——controller 防御式访问两者。
4. memory_off debug 默认关，保证对照公平。
5. `enabled=false` 完全退化为阶段 1 逐帧全存（回归基线）。
6. `prev_frame_ctx` 仅内存（不持久化），跨进程不保留——符合短期上下文语义。

---

## 13. 文件改动清单

- 新增 `src/vla_memory/memory/admission.py`
- 改 `config/memory.yaml`（+ `mid_term.admission`）
- 改 `src/vla_memory/pipeline/online_loop.py`（`__init__` / `setup()` / `step()` e.1 / `step()` 末尾 prev_ctx）
- 改 `docs/memory_design.md`（+ §3.8）
- 改 `README.md`（§9.2）
- 新增 `docs/stage2_admission_design.md`（本设计）
