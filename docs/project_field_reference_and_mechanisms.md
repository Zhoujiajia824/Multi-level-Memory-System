# 项目字段参考与机制深度剖析（V1.0）

> 用途：科研汇报/论文的数据字典 + 中期记忆"增删查改沉淀"机制 + 各阶段 I/O + 感知信息使用链路。
> 所有字段、枚举、示例均来自项目真实 schema（schemas/ + 在线循环产物 decisions_*.jsonl）。

---

# 第一部分：全字段数据字典

## 1.1 决策记录 jsonl 顶层字段（`outputs/decisions_<mode>_<run_id>.jsonl`，每行一帧）

| 字段 | 类型 | 作用 | 取值/举例 |
|---|---|---|---|
| `frame_id` | str | 帧唯一标识 | = sample_token |
| `sample_token` | str | nuScenes 样本 token | `"e93e98b63d3b4020..."` |
| `scene_token` | str | 所属场景 token | `"73030fb6..."` |
| `timestamp` | int | 帧时间戳（μs） | `1531883530447526` |
| `memory_mode` | str | 运行模式 | `memory_on` / `memory_off` |
| `perception_mode` | str | 感知输入模式 | `single_front` / `surround_mosaic` |
| `current_scene` | dict | 场景理解结果（见 1.2） | — |
| `decision_output` | dict | 决策输出（见 1.3） | — |
| `ego_state` | dict | 自车状态（见 1.4） | — |
| `nav_instruction` | str | 伪导航语义 | `straight`/`left_turn`/`right_turn`/`lane_follow`/`lane_change_left`/`lane_change_right`/`slow_or_stop`/`unknown` |
| `history_trajectory` | list | 过去 5s ego-centric 轨迹 | `[{t:-0.5,x,y},...]`，t 负值 |
| `ground_truth_trajectory` | list | 未来 3s 真值轨迹（评测用） | `[{t:0.1,x,y,optional_v},...]` |
| `vlm_image_paths` | list[str] | 喂给决策 VLM 的图像路径（oldest→newest，末位=当前帧） | surround 模式为 mosaic 路径；`[".../mosaic/<token>.jpg"]` |
| `perception_objects` | list[dict] | Oracle 感知对象（见 1.5） | `[{category:"vehicle",distance_to_ego:10.2,...}]` |
| `retrieved_memory_ids` | list[str] | 本帧中期检索命中的 record_id | `["ca9a282c...",...]` |
| `retrieved_memory_scores` | list[float] | 命中的 6 路融合相似度（与 ids 对齐） | `[0.82, 0.71, ...]` |
| `retrieved_memory_value_scores` | list[float] | 命中的 memory_value_score（价值重排用） | `[0.9, 0.6, ...]` |
| `retrieved_memory_event_types` | list[str] | 命中的 event_type | `["intersection","lane_change",...]` |
| `retrieved_memory_statuses` | list[str] | 命中的 status（验证未返回 inactive） | `["active","active",...]` |
| `retrieval_candidate_count` | int | 检索候选总数（过滤前） | `12` |
| `retrieval_active_candidate_count` | int | 活跃候选数（过滤 inactive 后） | `11` |
| `retrieval_filtered_count` | int | 被过滤掉的数量 | `1` |
| `long_term_rule_ids` | list[str] | 命中的长期规则 ID | `["RULE_010","RULE_003"]` |
| `memory_admission_enabled` | bool | 准入门控是否启用 | `true`/`false`（memory_off 时仍可能为 true，但不下发） |
| `memory_admission_score` | float | 本帧价值分（0-1） | `0.8` |
| `memory_admission_should_store` | bool | 准入决策（是否入库） | `true`=admit / `false`=reject |
| `memory_admission_reasons` | list[str] | 准入原因 | `["high_value_event:intersection"]` / `["legacy_no_gating"]` |
| `memory_admission_reject_reasons` | list[str] | 拒绝原因（拒绝时填） | `["low_value:stable_stop"]` |
| `memory_event_type` | str | 事件类型 | `intersection`/`lane_change`/`hard_brake`/`cut_in`/...（见枚举表） |
| `memory_scene_tags` | list[str] | 场景标签 | `["intersection"]` |
| `memory_risk_tags` | list[str] | 风险标签 | `["occlusion"]`/`["ghost_probing_risk"]` |
| `memory_type` | str | 本帧记忆类型 | `frame_memory`/`event_buffered`（事件进行中）/`event_memory`（已 finalize） |
| `memory_record_status` | str | 记忆状态 | `active`/`skipped`（memory_off） |
| `mid_term_memory_added` | bool | 本帧是否实际写入中期库 | `true`/`false` |
| `mid_term_memory_id` | str | 写入的 record_id（=sample_token） | `"e93e98b6..."` |
| `memory_update_enabled` | bool | 冲突更新是否启用 | `true` |
| `event_id` | str | 所属事件 ID（frame_memory/buffered 时空） | `""` / `"event_e93e98"`（event_memory 非空） |
| `parser_status` | str | VLM 输出解析状态 | `success`/`fallback`/`parse_error` |
| `parser_errors` | list | 解析错误 | `[]` |
| `fallback_used` | bool | 是否用了规则兜底 | `false`/`true` |
| `raw_response` | str | VLM 原始响应 | 完整 JSON 文本 |

## 1.2 `current_scene`（场景理解 VLM 输出，SceneUnderstandingResult）

| 字段 | 类型 | 作用 | 取值/举例 |
|---|---|---|---|
| `scene_description` | str | 场景自然语言描述 | `"城市十字路口，自车停在停止线前等待红灯..."` |
| `ego_status_text` | str | 自车状态自然语言 | `"自车静止在路口停止线前..."` |
| `scene_id` | str | 场景类型枚举 | `intersection`/`straight_road`/`turning`/`lane_change`/`car_following`/`obstacle_avoidance`/`crosswalk`/`merge`/`dead_end`/`unknown` |
| `weather_id` | str | 天气枚举 | `sunny`/`cloudy`/`rainy`/`snowy`/`foggy`/`night`/`unknown` |
| `traffic_density` | str | 交通密度 | `low`/`medium`/`high`/`unknown` |
| `lanes[]` | list[dict] | 车道线 | `{side:ego/left/right/oncoming, type:solid/dashed/double/merge/exit, color, direction}` |
| `vehicles[]` | list[dict] | 周围车辆 | `{relative_position:"−30°", distance_m:25.0, type:truck, motion:stationary/approaching/receding/crossing}` |
| `pedestrians[]` | list[dict] | 行人 | `{relative_position:"−140°", distance_m:12.0, intent:crossing/standing/walking_along}` |
| `traffic_lights[]` | list[dict] | 信号灯 | `{state:red/yellow/green/off, relative_position, controls_ego_lane:true/false}` |
| `intersections` | dict | 路口 | `{present:true, type:four_way/three_way_T/three_way_Y/roundabout, distance_m, has_stop_sign}` |
| `risk_factors[]` | list[str] | 风险因素（自然语言） | `["前方红灯需等待","后方车辆通过路口需注意避让"]` |
| `surrounding_objects[]` | list | 旧混合对象（兼容回退） | — |
| `lane_description` | str | 旧车道描述（兼容） | — |

## 1.3 `decision_output`（决策 VLM 输出）

| 字段 | 类型 | 作用 | 取值/举例 |
|---|---|---|---|
| `behavior` | str | 驾驶行为枚举 | `KEEP_LANE`/`FOLLOW`/`SLOW_DOWN`/`STOP`/`TURN_LEFT`/`TURN_RIGHT`/`CHANGE_LANE_LEFT`/`CHANGE_LANE_RIGHT`/`AVOID_OBSTACLE`/`YIELD`/`UNKNOWN` |
| `behavior_reason` | str | 决策因果摘要 | `"前方红灯控制自车车道，需减速停车..."` |
| `target_speed` | float | 目标巡航速度（m/s） | `0.0`（停车）/ `5.0` |
| `risk_level` | str | 风险等级 | `low`/`medium`/`high` |
| `trajectory[]` | list[dict] | ego-centric 轨迹点 | `{t:0.1, x:0.4, y:0.05, optional_v:3.8}`（x 前向、y 左向，米） |
| `safety_notes[]` | list[str] | 安全注意事项 | `["前方红灯，必须在停止线前完全停车"]` |

## 1.4 `ego_state`（自车状态，CAN bus 优先）

| 字段 | 类型 | 作用 | 取值/举例 |
|---|---|---|---|
| `x/y/z` | float | 全局/ego 位置（米） | `1010.14, 610.89, 0` |
| `yaw` | float | 航向角（弧度） | `1.452` |
| `speed` | float | 速度（m/s） | `4.09` |
| `vx/vy` | float | 速度分量 | — |
| `ax/ay` | float | 加速度分量 | — |
| `acceleration` | float | 加速度幅值（m/s²） | `0.93` |
| `yaw_rate` | float | 偏航角速率（rad/s） | `0.3299` |
| `steering_angle` | float | 方向盘转角（rad） | `3.3493` |
| `throttle` | float | 油门 [0,1] | `0.55` |
| `brake` | float | 刹车 [0,1] | `0.0` |
| `gear` | str | 档位 | `"D"` |
| `source` | str | 数据来源 | `can_bus`/`can_bus_pose_only`/`pose_diff`（差分回退） |
| `timestamp` | int | 时间戳（μs） | — |

## 1.5 `perception_objects[]`（Oracle 感知对象，单个 PerceptionObject）

| 字段 | 类型 | 作用 | 取值/举例 |
|---|---|---|---|
| `annotation_token` | str | 本帧 GT 标注 token | — |
| `instance_token` | str | 实例 token（跨帧稳定 ID） | — |
| `category` | str | 映射后主类别 | `vehicle`/`pedestrian`/`cyclist`/`obstacle`/`traffic_light`/`unknown` |
| `category_name_raw` | str | nuScenes 原始点分类别 | `"vehicle.car.parked"`/`"human.pedestrian.adult"`/`"movable_object.barrier"` |
| `semantic_label` | str | 细粒度语义 | `car`/`truck`/`pedestrian`/`barrier` |
| `attributes` | list[str] | nuScenes 属性 | `["vehicle.moving"]`/`["vehicle.parked"]` |
| `size` | list[float] | 3D 尺寸 [w,l,h]（米） | `[1.8, 4.5, 1.7]` |
| `position_global` | list[float] | 全局坐标 [x,y,z]（米） | — |
| `position_ego` | list[float] | ego-centric [x 前向, y 左向]（米） | `[8.2, 3.1]` |
| `distance_to_ego` | float | 到 ego 平面距离（米） | `10.26` |
| `heading_global` | float | 全局朝向（弧度） | — |
| `heading_ego` | float | 相对 ego 朝向（弧度） | — |
| `visible_cameras` | list[str] | 可见的相机 | `["CAM_FRONT","CAM_FRONT_RIGHT"]` |
| `boxes_2d` | dict | 各相机 2D 投影框 | `{"CAM_FRONT":[x1,y1,x2,y2]}`（原图像素） |
| `velocity` | list[float]\|null | ego-centric 速度 [vx 前, vy 左]（m/s） | `[3.82, -3.02]` / `null`（无历史） |
| `velocity_frame` | str | 速度坐标系 | `ego` |
| `speed` | float\|null | 速度大小（m/s） | `4.87` / `null` |
| `acceleration` | list[float]\|null | ego-centric 加速度 | `[ax, ay]` / `null` |
| `acceleration_mag` | float\|null | 加速度幅值（m/s²） | — |
| `velocity_available` | bool | 速度是否可用 | `false`（首现无历史）→ velocity/speed 为 null |
| `acceleration_available` | bool | 加速度是否可用 | — |
| `kinematics_source` | str | 运动学来源标记 | `annotation_keyframe_diff_2hz`/`..._velocity_only`/`unavailable_no_history`/`unavailable_invalid_dt` |
| `num_lidar_pts` | int | 盒内 LiDAR 点数 | `14` |
| `visibility_level` | str | nuScenes 可见度 | `"v3"`（80-100% 可见） |
| `is_oracle` | bool | **恒 True**：GT 投影真值，非模型预测 | `true` |

## 1.6 中期记忆记录 `MidTermMemoryRecord`（68 字段，schema v0.2）

**原始 12 字段**（记录决策后完整经验）：
`record_id`(=sample_token) / `frame_meta` / `image_feature_path`(.npy) / `scene_text` / `scene_id` / `weather_id` / `nav_instruction` / `ego_state` / `history_trajectory` / `decision_reason` / `behavior` / `trajectory`

**Phase 1 metadata 37 字段（8 类）**：
- 基础状态：`memory_id`/`memory_type`(frame_memory/event_memory)/`status`(active/deprecated/...)/`version`(v0.2)/`created_at`/`updated_at`
- 来源：`source_dataset`/`source_version`/`source_scene_token`/`source_scene_name`/`source_sample_token`/`source_frame_id`/`source_mode`
- 视觉：`visual_input_type`(single_front/surround_mosaic)/`image_path`/`feature_path`/`feature_dim`(768)
- 标签：`event_type`/`scene_tags[]`/`risk_tags[]`
- 准入：`admission_score`(0-1)/`admission_reasons[]`/`admission_policy_version`(legacy/value_gated_v0.1)
- 价值：`memory_value_score`/`salience_score`/`rarity_score`/`confidence_score`/`redundancy_score`/`retrieval_utility`/`recency_score`（均 0-1 或 null，未计算时 null）
- 使用统计：`hit_count`/`successful_hit_count`/`failed_hit_count`/`last_retrieved_at`
- 删除：`conflict_count`/`superseded_by`/`deleted_reason`/`deleted_at`/`is_active`(bool)

**Phase 5 事件字段（13）**：`event_id`/`event_start_sample_token`/`event_peak_sample_token`/`event_end_sample_token`/`anchor_sample_token`/`key_sample_tokens[]`/`anchor_image_path`/`key_image_paths[]`/`ego_summary`/`perception_summary`/`decision_summary`/`admission_summary`/`usage`

**Phase 6 冲突字段（4）**：`last_conflict_at`/`conflict_reasons[]`/`previous_versions[]`/`update_history[]`

## 1.7 短期记忆 `ShortTermMemoryItem`（11 字段）
`frame_id`/`timestamp`/`image_path`/`image_feature_path`/`scene_description`/`scene_id`/`weather_id`/`nav_instruction`/`ego_state`/`history_trajectory`/`scene_understanding_result`

## 1.8 长期规则 `LongTermRule`
`rule_id`(RULE_010)/`scene_id`(intersection 或 "all" 通配)/`weather_id`/`title`/`content`/`priority`(1-5)。共 16 规则 + 13 策略。

## 1.9 枚举值速查表（汇报速查）
- **behavior**：KEEP_LANE/FOLLOW/SLOW_DOWN/STOP/TURN_LEFT/TURN_RIGHT/CHANGE_LANE_LEFT/CHANGE_LANE_RIGHT/AVOID_OBSTACLE/YIELD/UNKNOWN
- **scene_id**：intersection/straight_road/turning/lane_change/car_following/obstacle_avoidance/crosswalk/merge/dead_end/unknown
- **memory_event_type**：lane_change/hard_brake/hard_acceleration/start/obstacle_avoidance/intersection/dense_traffic/pedestrian_interaction/cyclist_interaction/cut_in/merge/turn_left/turn_right/crosswalk/decision_change/occlusion/ghost_probing_risk/long_tail
- **memory_type**：frame_memory / event_buffered / event_memory
- **status**：active / low_confidence / deprecated / superseded / inactive / deleted / archived
- **perception_mode**：single_front / surround_mosaic
- **kinematics_source**：annotation_keyframe_diff_2hz / _velocity_only / unavailable_no_history / unavailable_invalid_dt
- **ego_state.source**：can_bus / can_bus_pose_only / pose_diff

---

# 第二部分：中期记忆"增删查改 + 沉淀"机制深度剖析

> 全部在 `OnlineDrivingLoop.step()` 的**写入路径**内（决策之后），绝不前移到检索之前——
> 这是"先读后写"因果约束的红线。memory_off 完全不进入此路径（公平对照）。

## 2.1 增（Create）— 价值准入门控（阶段 2，`admission.py`）

- **触发时机**：每帧决策完成、构造 `MidTermMemoryRecord` 之后、`add_record` 之前。
- **输入**：本帧 `parsed`(behavior/risk/target_speed)、`scene_result`(scene_id/density/risk_factors/vehicles/pedestrians)、`ego_state`(speed/accel/yaw_rate)、`perception_objects`(oracle)、上一帧 behavior/ego（来自短期记忆）、`fallback_used`。
- **核心逻辑**：6 信号加权求价值分（权重和=1.0）：
  - `dynamics_surprise`(0.20)：自车动态突变（Δspeed/Δaccel/jerk/Δyaw_rate/目标速度变化）
  - `scene_salience`(0.25)：场景语义高价值（scene_id∈{intersection,lane_change,...} / 高密度 / 路口 / 行人 / 风险因子）
  - `perception_change`(0.20)：感知突变（对象数变化 / 最近距离缩小 / cut_in / 行人 / cyclist）
  - `decision_change`(0.15)：行为或决策变化（behavior 变 / risk 升 / target_speed 大变 / 轨迹大变 / fallback / 解析失败）
  - `memory_novelty`(0.15)：与已有中期记忆的最大相似度取反（1−maxsim）
  - `posthoc_outcome_value`(0.05)：后验价值（仅当前帧决策质量代理，**绝不读未来**）
- **决策**：
  - 命中任一 **18 类高价值事件**（lane_change/start/hard_brake/obstacle_avoidance/intersection/dense_traffic/pedestrian_interaction/cyclist_interaction/cut_in/merge/turn_left/turn_right/crosswalk/decision_change/occlusion/ghost_probing_risk/long_tail）→ **force store**，并填 `event_type/scene_tags/risk_tags`。
  - 否则若命中 **3 类低价值过滤**（stable_stop / normal_cruise / redundant_frames）→ **拒绝**。
  - 否则按 `score_threshold`(0.55) 判定：≥ 入库，< 拒绝；`force_store_threshold`(0.80) 强制入库。
- **输出**：`AdmissionResult{score, should_store, event_type, scene_tags, risk_tags, reasons, reject_reasons, policy_version}`。
- **数据格式**：结果写入 `MidTermMemoryRecord` 的 `admission_score/admission_reasons/admission_policy_version/event_type/scene_tags/risk_tags`；jsonl 同步 `memory_admission_*` 字段。
- **配置**：`config/memory.yaml -> mid_term.admission.{enabled, policy_version, score_threshold, weights, low_value_filters, high_value_events, thresholds}`。
- **约束**：`enabled=false + store_all_when_disabled=true` → 退化为逐帧全存（回归基线）。

## 2.2 删（Delete）— 容量价值淘汰（阶段 3，`eviction.py` + `value_scorer.py`）

- **触发时机**：`add_record` 写入后；当 `active 记录数 ≥ max_records × eviction_trigger_ratio(0.80)` 触发常规淘汰；`≥ × emergency_trigger_ratio(0.95)` 触发紧急淘汰（放宽保护）。
- **存量价值评分**（`MemoryValueScorer`）：综合 `admission_score + 事件高价值 + recency_score(近期性) + retrieval_utility(检索效用) + redundancy(冗余,负) + confidence + 冲突惩罚` → `memory_value_score`。
- **淘汰策略**（soft delete）：按 `memory_value_score` 升序淘汰，淘汰后 active 降到 `max_records × eviction_target_ratio(0.70)` 附近。
- **保护机制**：`protect_long_tail`(长尾事件)/`protect_high_risk`(risk_tags 非空)/`protect_recent_high_value`；每类 event_type 至少保留 `min_keep_per_event_type` 条（emergency 也保留）。
- **soft delete 实现**：`is_active=False` / `status=deleted` / `deleted_reason="capacity_eviction"` / `deleted_at=<ts>`（元数据保留，可审计）。
- **FAISS 物理压缩**（`MemoryCompactionManager`）：inactive 比例过高时 `rebuild_index`——用 `IndexFlatIP.reconstruct_n` 读出活跃向量，新建索引批量写入 + 重写 `.ids.json` + `mid_term_meta.json`（flat 索引不支持原生 remove，故 rebuild）。
- **输出**：被淘汰记录的 `is_active/status/deleted_reason`；FAISS 索引物理缩小。
- **配置**：`mid_term.capacity.{max_records, eviction_trigger_ratio, emergency_trigger_ratio}` + `mid_term.eviction.{strategy, protect_*, min_keep_per_event_type}` + `mid_term.compaction`。

## 2.3 查（Retrieve）— 价值感知检索（阶段 4，`mid_term_memory.search`）

- **触发时机**：每帧决策**之前**（检索阶段，read-before-write）。
- **流程**（漏斗）：
  1. **过滤**：剔除 `is_active=False` / `status∈{deleted,deprecated,superseded}` / 低置信记录。
  2. **候选池**：FAISS 取 top-N（>top_k，供重排）。
  3. **6 路相似度**：`final = 0.40·visual(FAISS余弦) + 0.10·text(Jaccard) + 0.20·scene(scene_id精确) + 0.10·weather + 0.10·nav + 0.10·state(速度/加速度/航向)`。
  4. **价值重排**：`value_aware = 0.8·相似度 + 0.2·memory_value_score`。
  5. **多样性约束**：`max_per_event_id` / `max_per_scene_token` 限流 + 近重复抑制（高相似去重）。
  6. **top-K**（默认 3）+ 更新命中统计（`hit_count++`/`last_retrieved_at`）。
  7. **事件优先**：`prefer_event_memory` 加成（event_memory 排序靠前）。
- **输出**：`{results:[{record, final_score, value_score, sub_scores}], stats:{candidate_count, active_candidate_count, filtered_count}}`。
- **数据格式**：jsonl 写 `retrieved_memory_ids/scores/value_scores/event_types/statuses` + 漏斗三 count。
- **配置**：`mid_term.weights.*` + `mid_term.retrieval.{enable_value_rerank, value_weight, max_per_event_id, ...}`。

## 2.4 改（Update）— 冲突感知软更新（阶段 6，`update.py`）

- **触发时机**：本帧记忆写入**之后**；对本帧与检索到的相似旧记忆做冲突检测。
- **冲突分类（5 类，按优先级）**：
  1. `context_mismatch`：情境不同（scene/weather/nav 差异大）→ **不冲突**，两者共存。
  2. `unsafe_new_evidence`：新证据不安全（高风险变道等）→ **不覆盖旧**，新记忆降权标记。
  3. `unsafe_old_memory`：旧记忆涉险（risk_tags 非空且过时）→ 新记忆取代旧（旧标 superseded）。
  4. `policy_conflict`：跨类别策略不同（同情境但 behavior 矛盾）→ 旧降权 + conflict_count++。
  5. `style_conflict`：同类别风格不同 → 两条都保留（风格多样性）。
- **软更新动作**（不物理删除）：降权(`status=low_confidence`) / 标记(`deprecated`/`superseded`) / `superseded_by` 指向新 / `conflict_count++` / `update_history[]` 追加 `{action, conflict_type, reason, at, by_new}` / 版本链(`previous_versions[]`)。
- **输出**：旧记忆的 `update_history/conflict_reasons/status`；jsonl `memory_update_enabled`。
- **配置**：`mid_term.update.{enabled, conflict_types, ...}`。
- **安全红线**：unsafe 新证据绝不覆盖安全旧记忆（防学到危险偏好）。

## 2.5 事件合并 — EventMemory（阶段 5，`event_memory.py`，连接"增"与"沉淀"）

- **触发时机**：准入门控 admit 后；连续高价值帧缓冲为一个事件。
- **事件结束条件**：高价值信号消失 / 达 `max_length` / scene 切换 / run 结束。
- **finalize 输出**（一条 event_memory 记录）：`event_id` + 关键帧（start/peak/end，peak=admission 最高帧）+ `anchor_sample_token`(=peak) + 4 摘要：
  - `ego_summary`：`"speed 4.1→4.5 m/s over 5 frames"`
  - `perception_summary`：`"max 9 objects, 4 vehicles, 2 pedestrians"`
  - `decision_summary`：`"STOP -> TURN_LEFT"`
  - `admission_summary`：事件类型 + peak admission + 帧数
- **短事件处理**：长度不足 `min_length` 的事件**丢弃**（避免噪声）。
- **检索优先**：event_memory 在检索时获 `prefer_event_memory` 加成。
- **配置**：`mid_term.event_memory.{enabled, max_length, min_length, prefer_event_memory}`。

## 2.6 沉淀（Consolidate）— 中期→长期候选（阶段 7，`consolidation.py` + `scripts/08`）

- **触发时机**：**离线**批处理（跑完 demo 后手动执行 `08_consolidate_long_term_candidates.py`），不在在线循环内。
- **流程**（不依赖 FAISS/VLM，只读 `mid_term_meta.json`）：
  1. 筛选高价值 active `event_memory` 记录。
  2. 按 `(event_type, risk_tags)` 分组。
  3. 组内 `count ≥ min_evidence_count`(默认 3) 且 `avg_value ≥ 阈值` → 生成候选规则。
  4. 候选含 `evidence memory_ids[]` / `confidence` / `safety_guard` / 三类分类（safety/strategy/style）。
  5. **危险偏好剔除**（高风险变道等不沉淀）；style 候选不得覆盖 safety 规则。
- **输出**：`outputs/long_term_candidates/candidate_rules.yaml`，`status=pending_review`。
- **关键约束**：**不自动覆盖**正式长期库 `data/knowledge/long_term_rules.yaml`——需人工审核晋升（保安全）。
- **配置**：`mid_term.consolidation.{min_evidence_count, min_avg_value, ...}`。
- **价值回退**：`memory_value_score` 缺失时回退用 `admission_score`。

---

# 第三部分：各阶段输入输出与数据格式（速查）

| 阶段 | 触发 | 输入 | 输出 | 数据格式 |
|---|---|---|---|---|
| 感知 | step ② | sample_token + image_path(mosaic) | DINOv2 .npy(768d) + scene_result(JSON) | features/<token>.npy + current_scene dict |
| 检索 | step ③（决策前） | query_feature + scene_id + ... | mid_term_results + stats | record列表 + score/value/event_type/status |
| 决策 | step ⑤ | N图 + prompt(场景/ego/nav/history/记忆/oracle) | behavior/trajectory/... | decision_output dict |
| 准入(增) | step ⑧（决策后） | parsed + scene + ego + oracle + 上一帧 | AdmissionResult | admission_score/should_store/reasons/event_type |
| 事件合并 | admit 后 | 连续高价值帧缓冲 | event_memory 记录 | memory_type=event_memory + 4 summaries |
| 写入(增) | admit 后 | mt_record + feature | _records[id] + FAISS add | MidTermMemoryRecord(68字段) |
| 淘汰(删) | add 后(超容量) | 全库 + memory_value_score | 被淘汰记录 is_active=False | deleted_reason/deleted_at + rebuild |
| 冲突(改) | 写入后 | 新记忆 + 相似旧记忆 | 旧记忆 update_history | conflict_count/status/superseded_by |
| 沉淀 | 离线 | mid_term_meta.json | candidate_rules.yaml | pending_review 候选规则 |

---

# 第四部分：感知信息如何被使用（完整数据流）

## 4.1 感知信息的来源（4 路）
1. **视觉图像**：nuScenes 6 相机 → surround_mosaic 拼成 2×3 单图（`outputs/mosaic/<token>.jpg`）。
2. **视觉特征**：mosaic/前视图 → DINOv2-base → 768 维向量（CLS token，L2 归一化）→ `outputs/features/<token>.npy`。
3. **Oracle 检测对象**：nuScenes GT `sample_annotation` → 6 相机投影 → `perception_objects[]`（含 boxes_2d/position_ego/velocity）。
4. **自车状态**：CAN bus（pose.json + vehicle_monitor.json）→ `ego_state`（speed/accel/yaw_rate/steering/throttle/brake）。

## 4.2 感知信息的使用去向（4 个消费者）

### A. 场景理解 VLM（把像素→结构化语义）
- 用：**视觉图像**（1 张 mosaic）。
- 产：`current_scene`（scene_id/vehicles/pedestrians/traffic_lights/...）。
- 该输出成为后续所有模块的"语义语言"。

### B. 决策 VLM（综合感知+记忆→动作）
- 用：**视觉图像**（N 张历史+当前）+ `current_scene`(文本段) + `ego_state` + `perception_objects`(oracle 段) + 三层记忆 + 导航/历史轨迹。
- 产：`decision_output`（behavior/trajectory）。
- Oracle 对象段在 prompt 里明确标注"GT 投影，非预测"，供决策参考精确目标运动学。

### C. 中期记忆（视觉特征做检索锚 + 感知信号做价值打分）
- 用：**视觉特征**（FAISS 检索的 query 与 key）+ `current_scene`(scene_text/scene_id) + `ego_state`(state 相似度)。
- 准入打分用：`perception_objects`(cut_in/行人检测) + `ego_state`(动态突变) + `current_scene`(场景显著性) + `decision_output`(决策变化)。
- 事件摘要用：`perception_objects`(对象计数/最近距离) + `ego_state`(速度趋势)。

### D. 短期记忆（最近帧上下文）
- 用：`image_path`(历史图喂决策 VLM) + `scene_understanding_result` + `ego_state`。
- 价值打分时取"上一帧 behavior/ego"做变化检测（`get_latest(2)`）。

## 4.3 关键设计：感知信息的因果性
- **Oracle 速度/加速度严格因果**：只沿 annotation `prev` 链回溯（当前+历史），**绝不读 `next`**（未来帧）；缺历史→`velocity=null` + `velocity_available=false`，**禁止伪造**。
- **感知在决策前完成**：感知（②）→ 检索（③）→ 决策（⑤），感知结果供检索与决策共用。
- **感知与记忆解耦**：感知失败（VLM 异常）→ 记录错误帧 + 规则兜底，不污染记忆库。

## 4.4 感知信息数据格式实例（scene-0001）
```
perception_objects[0] = {
  "category":"vehicle", "semantic_label":"car", "distance_to_ego":10.26,
  "position_ego":[8.2, 3.1], "speed":4.87, "velocity_available":true,
  "visible_cameras":["CAM_FRONT","CAM_FRONT_RIGHT"],
  "boxes_2d":{"CAM_FRONT":[120,200,800,700]},
  "kinematics_source":"annotation_keyframe_diff_2hz", "is_oracle":true
}
ego_state = {"speed":4.09, "acceleration":0.93, "yaw_rate":0.3299,
             "steering_angle":3.3493, "throttle":0.55, "brake":0.0, "source":"can_bus"}
current_scene = {"scene_id":"intersection", "traffic_density":"low",
                 "vehicles":[{"distance_m":25.0,"motion":"stationary"}],
                 "traffic_lights":[{"state":"red","controls_ego_lane":true}]}
```

---

> 本文档可作为论文"方法+数据"章节的素材库。所有字段/枚举/示例均与代码 schema 一致，
> 可直接用于配图标注、表格、附录数据字典。
