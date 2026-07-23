# 外挂式智驾 Agent 多层次记忆系统 V1.0 — 课题阶段性成果汇报大纲

> 用途：PPT 汇报 + GPT 生成科研配图。每节含【讲解】【配图提示词】【PPT 建议】三部分。
> 项目状态：离线研究 Demo（nuScenes 开环轨迹预测，非实车），核心正确性 = 在线循环"先读后写"防数据泄漏。
> 核心命题：让 VLM 驾驶决策**复用历史经验 + 显式驾驶知识**，通过 memory_on/off 对照量化记忆增益。

---

## 0. 一句话定位（封面页）

**外挂式（Plug-in）多层记忆系统**：一个与基础 VLA 规划器解耦的记忆"外挂"，给任意 VLM 驾驶 Agent
挂上"工作记忆 / 情景记忆 / 语义记忆"，使其在驾驶决策时**复用历史相似场景经验与显式驾驶规则**，
提升轨迹规划质量。V1.0 在 nuScenes 上离线验证，提供 memory_on vs memory_off 对照实验。

- **配图提示词（封面系统总览图）**：
  > 一张学术论文风格的系统总览示意图。白色背景，扁平极简。中央是一辆自顶向下俯视的自动驾驶汽车图标（蓝色），
  > 围绕它三层同心圆环：内环"短期记忆 Short-Term（工作记忆，秒级）"用浅蓝，中环"中期记忆 Mid-Term
  > （情景记忆，分钟级，FAISS）"用紫色，外环"长期记忆 Long-Term（语义记忆，永久规则）"用绿色。
  > 左侧输入箭头标注"感知输入：六视角环视图 + Oracle 检测"，右侧输出箭头标注"决策输出：行为+轨迹"。
  > 整体用中文标签，箭头为深灰，风格干净专业，适合论文 figure。
- **PPT 建议**：标题 + 一句话价值主张 + 总览图；副标题"V1.0 · nuScenes 离线验证 · memory_on/off 对照"。

---

## 1. 项目整体设计执行逻辑

【讲解】系统以 `OnlineDrivingLoop.step()` 为核心，**逐帧严格在线**处理，11 步时序：
①感知（DINOv2 特征 + 场景理解 VLM）→ ②三层记忆检索（只用 [0,i-1] 帧）→ ③组装多图输入 →
④决策 VLM → ⑤解析/兜底 → ⑥写 jsonl → ⑦更新短期记忆 → ⑧中期记忆准入门控 + 写入。
**红线**：第 i 帧检索时索引只含历史帧，当前帧在决策完成后才写入——彻底消除批处理 demo 常见的
data leakage。整体三阶段：`prepare_nuscenes`（加载+关键帧采样）→ `enrich_keyframes`（补 ego/轨迹/
mosaic/oracle）→ `OnlineDrivingLoop.run`（逐帧在线）。

- **配图提示词（在线循环流水线图）**：
  > 学术论文 figure 风格的横向流水线图，白底扁平。从左到右一排圆角矩形方框，用深灰箭头串联：
  > [nuScenes 数据] → [关键帧采样 enrich] → [感知：DINOv2 + 场景VLM] → [三层记忆检索] →
  > [决策 VLM] → [解析] → [写 jsonl] → [更新记忆（短期push / 中期门控写）]。
  > 在"检索"与"更新记忆"之间用红色虚线标注"决策在此之后"，并加红色警告图标写"先读后写：
  > 第 i 帧只见历史帧，防 data leakage"。方框按功能上色（感知=蓝、记忆=紫、决策=橙、输出=绿）。
  > 中文标签，底部时间轴箭头表示"逐帧 t₀→tᵢ"。
- **PPT 建议**：这张图作为"方法总览"主图。讲解时强调"在线循环 + 先读后写"是正确性根基。

---

## 2. 数据集与感知数据处理

【讲解】
- **数据集**：nuScenes v1.0-trainval（取 Part1，85 个场景，约 34000 关键帧样本）。关键帧 2Hz。
- **自车状态**：优先 CAN bus 真值（`pose.json` + `vehicle_monitor.json`：速度/加速度/偏航角速率/
  方向盘/油门/刹车），失败回退 ego_pose 差分。
- **视觉特征**：DINOv2-base 提 768 维向量（CLS token，L2 归一化），供中期记忆 FAISS 检索。
- **六视角环视（surround_mosaic）**：6 相机（CAM_FRONT_LEFT/FRONT/FRONT_RIGHT/BACK_LEFT/BACK/BACK_RIGHT）
  拼成 2×3 单张图替代单前视，进入 VLM/特征/记忆全流程。
- **Oracle 感知对象**：基于 nuScenes **GT 标注（sample_annotation）投影**到 6 相机，生成结构化
  `perception_objects`（检测框/类别/位置/速度/加速度），明确标注**非模型预测、研究用真值**。

- **配图提示词（感知数据处理流图）**：
  > 学术 figure 风格，白底。左侧是 nuScenes 6 个相机原图小图标（排成 2×3 网格）。
  > 三条平行处理支路汇入右侧"决策输入"：
  > 上支：6 张图 → [PIL 拼接] → 一张 2×3 surround mosaic 大图（标注相机名）；
  > 中支：单张图 → [DINOv2] → 768 维向量（紫色条形）；
  > 下支：GT 3D 标注 → [相机投影] → Oracle 检测框 + 目标运动学（用小立方体+框表示）。
  > 另有一条"CAN bus"支路提供自车状态。中文标签，箭头深灰，方框浅色填充。
- **PPT 建议**：强调"多模态感知输入"（图/特征/检测框/自车状态）汇聚到 VLM。

---

## 3. 三层记忆架构梳理

【讲解】借鉴认知科学三阶段记忆模型：
| 层 | 对应认知 | 时间跨度 | 检索方式 | 智驾映射 | 实现 |
|---|---|---|---|---|---|
| 短期 | 工作记忆 | 秒级 | 顺序滑窗 | 最近 N 帧路况演化 | `deque(maxlen=10)` |
| 中期 | 情景记忆 | 分钟级 | 相似度检索 | 本次驾驶的相似路口经验 | FAISS IndexFlatIP + 6 路加权 + 价值门控 |
| 长期 | 语义/程序记忆 | 永久 | 规则匹配 | 学到的驾驶规则 | YAML 规则库 |

统一入口 `MemoryRetriever` 一次返回三层结果；三层可独立开关（`use_short/mid/long_term`），
memory_off 全关构成纯净基线。

- **配图提示词（三层记忆架构图）**：
  > 学术示意图，白底。三个垂直堆叠的水平条带，从上到下：
  > 顶层"短期记忆"（浅蓝，画一个含 10 格的滑动窗口，最近帧从右进、最旧从左出）；
  > 中层"中期记忆"（紫色，画一个 FAISS 向量库图标 + 6 个小条表示 6 路相似度权重融合）；
  > 底层"长期记忆"（绿色，画一摞 YAML 规则卡片）。
  > 右侧一个"MemoryRetriever 统一检索"圆角框，三根箭头从三层汇入。左侧标注认知科学对应
  （工作记忆/情景记忆/语义记忆）。中文标签。
- **PPT 建议**：用这张图建立"仿生记忆"叙事，与人类驾驶员认知过程对应。

---

## 4. 三层记忆保存的数据内容（实例）

【讲解】每层保存不同粒度数据（V1.0 真实字段）：
- **短期**（`ShortTermMemoryItem`，11 字段）：`frame_id / timestamp / image_path / scene_id /
  weather_id / nav_instruction / ego_state(含 CAN) / history_trajectory / scene_understanding_result`。
  例：`{scene_id:"intersection", nav:"left_turn", speed:4.09, source:"can_bus", ...}`。
- **中期**（`MemoryRecord`，68 字段，schema v0.2）：原始 12（record_id/scene_id/behavior/trajectory/
  ego_state/...）+ 37 metadata（来源 source_* / 视觉 visual_input_type+feature_dim / 价值 admission_score+
  memory_value_score / 使用统计 hit_count / 删除 is_active）+ 事件字段（event_id/summaries）。
  例（事件级）：`{memory_type:"event_memory", event_type:"intersection", admission_score:0.8,
  ego_summary:"speed 4.1→4.5 m/s over 5 frames", perception_summary:"max 9 objects, 4 vehicles, 2 pedestrians",
  decision_summary:"STOP -> TURN_LEFT"}`。
- **长期**（`LongTermRule`）：`rule_id / scene_id / weather_id / title / content / priority`。共 16 条规则
  + 13 条策略。例：`{rule_id:"RULE_010", scene_id:"intersection", title:"路口减速观察", priority:2, ...}`。

- **配图提示词（三层数据内容对比图）**：
  > 学术 figure，白底，三列并排卡片对比。左列"短期记忆"：一个小滑动窗口图标 + 标"11 字段，
  秒级，路径+场景摘要"。中列"中期记忆"：一个数据库图标 + 标"68 字段，事件级，含价值分/摘要/
  oracle 关联"。右列"长期记忆"：一摞规则卡片 + 标"YAML 规则，scene 匹配，永久"。每列下方贴一个
  真实数据 JSON 片段（用等宽字体小框）。中文标签，列与列之间用虚线分隔。
- **PPT 建议**：用真实 JSON 片段增强可信度；强调"越深层越抽象/持久"。

---

## 5. 场景理解模型 I/O（实例）

【讲解】独立的 VLM（Qwen-VL 系列，经 DashScope/OpenAI 兼容 API），**无状态**，每帧调用。
- **输入**：1 张当前帧图（surround 模式为 mosaic）+ 静态 prompt（要求严格 JSON）。
- **输出**：结构化 JSON：`scene_description / scene_id / weather_id / traffic_density / lanes[] /
  vehicles[](relative_position/distance_m/type/motion) / pedestrians[](intent/distance) /
  traffic_lights[](state/controls_ego_lane) / intersections / risk_factors[]`。
- **实例**（scene-0001 首帧实测）：
  ```json
  {"scene_id":"intersection","weather_id":"cloudy","traffic_density":"low",
   "vehicles":[{"relative_position":"-30°","distance_m":25.0,"type":"truck","motion":"stationary"}, ...],
   "traffic_lights":[{"state":"red","relative_position":"front","controls_ego_lane":true}],
   "risk_factors":["前方红灯需等待","后方车辆正在通过路口需注意避让"]}
  ```

- **配图提示词（场景理解 I/O 图）**：
  > 学术 figure，白底。左侧一张驾驶场景图（mosaic 或前视图），中间一个"VLM 场景理解"圆角框
  > （带脑图标），右侧输出一张结构化 JSON 卡片：用彩色小方块分桶展示 lanes/vehicles/pedestrians/
  > traffic_lights/intersections，每个桶标数量。底部标注"输入：1 图 + prompt → 输出：结构化 JSON"。
  > 中文标签，箭头从左到右。
- **PPT 建议**：强调"VLM 把像素转成结构化语义"，这是记忆与决策的共同语言。

---

## 6. 决策模型 I/O（实例）

【讲解】第二个独立 VLM（决策，可配不同模型），**有状态**（含上下文）。
- **输入**：N 张图（短期窗口 N-1 历史 + 当前帧）+ 动态 prompt（含：场景理解段 / 自车状态 / 导航 /
  历史轨迹 / 三层记忆段 / Oracle 感知对象段 / 图像布局说明）。
- **输出**：`behavior / behavior_reason / target_speed / risk_level / trajectory[{t,x,y,optional_v}] /
  safety_notes`。坐标系 ego-centric（x 前向、y 左向）。
- **实例**（scene-0001 红灯）：
  ```json
  {"behavior":"STOP","behavior_reason":"前方红灯控制自车车道，需减速停车等待；导航指令为左转，待绿灯亮起后执行",
   "target_speed":0.0,"risk_level":"medium","trajectory":[{"t":0.1,"x":0.4,"y":0.05,"optional_v":3.8}, ... 共25点]}
  ```

- **配图提示词（决策模型 I/O 图）**：
  > 学术 figure，白底。左侧"多模态输入"竖排：N 张缩略图（时间序列）+ 文本块（场景/自车/记忆/
  > oracle）。中间"决策 VLM"圆角框。右侧输出卡片：behavior 标签（彩色徽章 STOP）+ risk_level +
  > 一段 ego-centric 轨迹折线图（x 前向，减速到 0 的曲线）。底部标注"输入：图+上下文 → 输出：行为+轨迹"。
  > 中文标签。
- **PPT 建议**：突出"记忆如何注入决策 prompt"——这是系统价值落点。

---

## 7. 中期记忆的增删查改（CRUD）

【讲解】V1.0 中期记忆是完整的"价值门控事件经验库"，覆盖 CRUD 全生命周期（阶段 1-7）：
- **增（Create）— 准入门控**（阶段 2）：每帧决策后由 `MemoryAdmissionController` 判断是否入库。
  6 信号加权（动态突变/场景显著性/感知变化/决策变化/记忆新颖性/后验价值）算价值分；低价值帧
  （巡航/稳定停车/冗余）拒绝，17 类高价值事件（变道/急停/路口/cut-in/行人交互/...）强制入库。
- **删（Delete）— 容量淘汰**（阶段 3）：容量上限触发 `MemoryEvictionManager`，按 memory_value_score
  淘汰低价值（soft delete：`is_active=False`），高价值/长尾/高风险受保护；inactive 比例高时 FAISS rebuild
  物理压缩（IndexFlatIP 用 reconstruct_n 重建）。
- **查（Retrieve）— 价值感知检索**（阶段 4）：6 路相似度 + 价值重排（0.8·相似度+0.2·价值）+ 多样性约束 +
  过滤 inactive/deprecated。
- **改（Update）— 冲突感知更新**（阶段 6）：`MemoryUpdateManager` 检测新记忆与已有记忆冲突（5 类：
  策略冲突/风格冲突/情境不同/旧涉险/新不安全），软更新（降权/标记 deprecated/版本链），unsafe 新证据
  绝不覆盖安全旧记忆。
- **沉淀（Consolidate）— 阶段 7**：离线把高价值中期记忆按 (event_type,risk_tags) 分组，≥min_evidence_count
  时生成**长期记忆候选**（pending_review，不自动覆盖正式库）。

- **配图提示词（中期记忆 CRUD 生命周期图）**：
  > 学术 figure，白底。中央一个圆柱体数据库图标标"中期记忆库（FAISS + 元数据）"。围绕它四个箭头操作：
  > 上方"增 Create"：一帧数据 → [门控：6信号打分] → 入库（绿色✓）或拒绝（红色✗）；
  > 右方"查 Retrieve"：查询向量 → [6路相似度+价值重排] → top-K；
  > 下方"删 Delete"：[容量上限触发] → 低价值标 is_active=False → [rebuild 压缩]；
  > 左方"改 Update"：新记忆 vs 旧记忆 → [冲突分类] → 软更新/版本链。
  > 底部一个"沉淀 Consolidation"出口箭头指向"长期记忆候选"。中文标签，四操作用不同色。
- **PPT 建议**：这张是核心创新图；强调"不是无脑存，而是价值驱动的活记忆"。

---

## 8. 短期记忆原理

【讲解】`collections.deque(maxlen=10)` 滑动窗口，FIFO 自动淘汰最旧帧。维护最近 10 个关键帧的
连续上下文（场景演化趋势、速度变化）。给决策 VLM 提供两类输入：①最近 N-1 张历史图路径（多图上下文）；
②文本摘要（scene_id/weather/speed/description 趋势）。**全系统唯一显式遗忘机制**（FIFO）。瞬时上下文，
进程内不持久化。

- **配图提示词（短期记忆滑窗图）**：
  > 学术示意图，白底。一个水平的长条容器（蓝色边框），内含 10 个小方格代表 10 帧，从左(旧)到右(新)。
  > 右侧一个新帧（高亮）正要进入，最左侧一帧被弹出（半透明+向左箭头标"淘汰"）。容器下方标"deque maxlen=10，
  > FIFO"。每格内画一个小场景缩略图。中文标签。
- **PPT 建议**：用"传送带"比喻，简单直观。

---

## 9. 长期记忆保存与演化机制

【讲解】静态 YAML 规则库（16 规则 + 13 策略），按 `scene_id` + `weather_id` 严格匹配（默认 strict_scene_match，
"all" 通配被屏蔽），取 top-5 注入决策 prompt。**演化路径**（V1.0 已铺好半自动链路）：中期记忆高价值事件
→ 阶段 7 离线沉淀为**候选规则**（`outputs/long_term_candidates/candidate_rules.yaml`，pending_review，
带 evidence/confidence/safety_guard）→ **人工审核晋升**到正式 `long_term_rules.yaml`（V1.0 故意不自动覆盖，
保安全）。未来 v0.9 演进为知识图谱（Neo4j/NetworkX）替代 YAML。

- **配图提示词（长期记忆演化链路图）**：
  > 学术 figure，白底。从左到右一条演化链路：
  > [多次中期事件记忆] → [阶段7 离线沉淀] → [候选规则 pending_review（黄色卡片，带 confidence 评分）]
  > → [人工审核（放大镜图标）] → [正式长期规则库 long_term_rules.yaml（绿色卡片栈）]。
  > 右侧一个虚线未来箭头指向"知识图谱（v0.9）"。底部标注"不自动覆盖正式库，保安全"。中文标签。
- **PPT 建议**：突出"从经验到知识的自动沉淀 + 人工把关"双保险。

---

## 10. 中期记忆检索机制

【讲解】6 路加权融合相似度（V1.0 实际权重）：
`final = 0.40·visual(FAISS余弦) + 0.10·text(Jaccard) + 0.20·scene(scene_id精确) + 0.10·weather +
0.10·nav + 0.10·state(速度/加速度/航向加权)`。
取 top-K（默认 3）。阶段 4 升级为**价值感知检索**：候选池 top-N → 价值重排（0.8·相似度+0.2·价值分）→
多样性约束（同 event_type/同 scene 限流 + 近重复抑制）→ 过滤 inactive/deprecated/低置信 → 更新命中统计。
检索优先返回**事件级记忆**（event_memory，prefer_event_memory 加成）。

- **配图提示词（检索机制漏斗图）**：
  > 学术 figure，白底。一个从宽到窄的漏斗形状（4 层）：
  > 最上层"全库"（很多小点）→ 第2层"6 路相似度 top-N 候选池"（标 0.40/0.10/0.20/0.10/0.10/0.10 权重）
  > → 第3层"价值重排+多样性"（点变少，按价值染色）→ 最下层"返回 top-3"（3 个高亮点）。
  > 漏斗旁边小图标标"过滤：inactive/deprecated/低置信"。中文标签，漏斗用紫色渐变。
- **PPT 建议**：强调"不只是相似度，还有价值与多样性"——区别于纯向量检索。

---

## 11. 未来迭代方向与实现方式

【讲解】V1.0 是开环离线验证，后续路线（与 docs/future_work.md 一致）：
1. **CARLA 闭环仿真**：实现 `DynamicsPlanner.plan()`（轨迹→steering/throttle/brake），决策输出直接
   控制 CARLA 仿真车辆，形成"感知→决策→控制→反馈"真闭环；动态场景生成（行人横穿/前车急刹）；
   在线评测（碰撞率/偏离车道）。依赖：CARLA 0.9.13+ Python API。
2. **专属 VLM 微调**：接入 HuggingFace 本地 VLM（Qwen2-VL/LLaVA/InternVL），在 nuScenes + 记忆增强
   prompt 数据上微调（LoRA），让模型原生理解"记忆上下文"而非靠 prompt 拼接；YAML 一键切 API/本地。
3. **外挂到开源规划论文提升性能**：把本记忆系统作为**即插即用模块**，挂载到开源运动规划模型
   （如 PlanTF、UrbanDriver 等 nuScenes planning baseline）上，作为其"记忆增强前端"，在官方
   planning benchmark 上对比"基线 vs 基线+记忆"的 L2/碰撞率/舒适度指标，证明记忆外挂的通用增益。
4. （已铺）**知识图谱长期记忆**（v0.9）：Neo4j 替代 YAML，场景→行为→结果关系图谱。
5. **更复杂感知**：faiss-gpu、多摄像头 BEV 特征融合、LiDAR 点云。

- **配图提示词（未来路线图）**：
  > 学术 figure，白底。一条从左到右的时间轴/阶梯，4 个里程碑节点：
  > ①"CARLA 闭环"（画一个仿真城市+闭环箭头）；②"专属 VLM 微调"（画一个神经网络+LoRA 标签）；
  > ③"外挂开源论文"（画一个插件模块嵌入一个"baseline 规划器"方框，输出箭头标"性能↑"）；
  > ④"知识图谱"（画一个节点-边网络图）。每个节点下方一行小字说明。整体向上的阶梯表示"能力提升"。
  > 中文标签。
- **PPT 建议**：收尾页用路线图，强调"外挂式"通用性（能挂到任何规划器上）。

---

## 附：PPT 整体结构建议（约 15-18 页）

1. 封面（标题 + 总览图 + 定位）
2. 问题动机（为什么需要记忆：VLM 每帧从零决策的痛点）
3. 系统总览（第 1 节流水线图）
4. 数据与感知（第 2 节）
5-6. 三层记忆架构 + 数据实例（第 3、4 节）
7. 场景理解 I/O（第 5 节）
8. 决策 I/O（第 6 节）
9-10. 中期记忆 CRUD + 检索（第 7、10 节，核心创新）
11. 短期 / 长期记忆（第 8、9 节）
12. 实验设置（memory_on/off 对照、指标 ADE/FDE/L2@1s/2s/3s）
13. 阶段性结果（写入率/事件合并/检索质量——可用烟雾测试数据）
14. 未来方向（第 11 节）
15. 总结 + 致谢

## 附：使用说明

- 每节"配图提示词"可直接复制粘贴到 GPT（DALL-E / GPT-4o 图像生成）。建议统一加风格前缀：
  "生成一张学术论文 figure 风格的示意图，扁平极简，白色背景，中文标签，配色专业（蓝/紫/绿/橙），
  箭头清晰，适合放入科研 PPT。内容：……（粘贴上方提示词）"。
- 实例数据均来自项目真实运行（scene-0001 烟雾测试），可在汇报时切到前端 Streamlit 演示对应帧。
