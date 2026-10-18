# PDM-Closed 的 CARLA 适配层

这里直接调用官方 `PDMGenerator`、`PDMSimulator`（LQR + bicycle）、
`PDMScorer` 和 `PDMObservation`，没有将简化规则重命名为 PDM。
地图、目标、车辆状态及轨迹坐标由本项目适配，属于 **PDM-Closed core + CARLA adapter**，
不能称为原版 nuPlan benchmark，也不能直接比较其原版排行榜分数。

## 固定源码与安装

- [tuplan_garage](https://github.com/autonomousvision/tuplan_garage/tree/b51d5d04fac1bd4389653b9ab2ff73ea88f435a3)：
  `b51d5d04fac1bd4389653b9ab2ff73ea88f435a3`，包版本 `1.0.0`。
- [nuplan-devkit](https://github.com/motional/nuplan-devkit/tree/e9241677997dd86bfc0bcd44817ab04fe631405b)：
  `e9241677997dd86bfc0bcd44817ab04fe631405b`，源码版本 `1.2.2`。
  部分 PyPI 镜像没有 `1.2.2`，因此固定官方源码，不依赖镜像提供此版本。
- 启动时核验 `upstream_hashes.json` 中 80 个核心源码文件，允许 Windows 换行转换；
  同时读取安装来源 commit/checkout commit，明确不匹配时拒绝运行。
  日志记录实际来源、源码核验摘要、包版本和适配层版本。

从零新建环境的完整流程见 [CARLA README 第 5 节](../README.md#5-环境搭建)，
包括 Python、PyTorch、FAISS、模型缓存、API 和服务器连接。
下列步骤用于**已有基础依赖的环境**，可直接使用 `mulmem_carla`；不要求新建或克隆环境。
如果希望隔离旧实验，可自行先执行 `conda create -n mulmem_pdm --clone mulmem_carla`，
然后激活 `mulmem_pdm` 执行同样的安装步骤。

在服务器项目根目录执行：

```bash
conda activate mulmem_carla
python -m pip install "numpy==1.26.4" "opencv-python==4.10.0.84" "pytest>=7.4,<9" -r carla_bridge/planning/requirements-pdm.txt
python -m pip install --no-deps 'git+https://github.com/motional/nuplan-devkit.git@e9241677997dd86bfc0bcd44817ab04fe631405b'
mkdir -p .deps
# 如果该源码目录已存在，跳过 clone，继续核对固定版本和安装
git clone https://github.com/autonomousvision/tuplan_garage.git .deps/tuplan_garage
git -C .deps/tuplan_garage checkout b51d5d04fac1bd4389653b9ab2ff73ea88f435a3
python -m pip install --no-deps -e .deps/tuplan_garage
python -m pip check
python -m carla_bridge.planning.pdm_planner --check
python -m pytest carla_bridge/tests/test_pdm_planner.py carla_bridge/tests/test_pdm_selection.py -q
```

tuplan_garage 使用 editable 安装是因为该版本 `setup.py` 只列出了顶层包；
普通 wheel 安装可能漏掉 planner 子目录。不要直接执行它的历史 `requirements.txt`，
本项目清单已补齐核心的间接导入依赖和上游声明的包约束。nuPlan 的 observation 模块
会导入点云/绘图模块，因此即使不画图也需要 matplotlib、Pillow、OpenCV 和 pytest。
tuplan 1.0.0 的元数据要求 `scikit-learn==1.2.2` 和 `positional-encodings==6.0.1`，
安装它们是为了满足包依赖检查，不表示启用了学习型规划器。已有 scikit-learn 1.5.2
会被调整为 1.2.2。没有下载模型权重的步骤。
`--check` 不连接 CARLA、不初始化 VLM、不运行仿真。缺包或源码不符会直接报错，
不会静默切换到其它规划器。服务器不能访问 GitHub 时，可在联网机器下载对应
commit 的源码及适用于服务器 Python/Linux 的依赖 wheel，上传后按同样的
`--no-deps`/editable 方式从本地目录安装，再执行 `--check`；不要上传 Windows wheel。

## 规划与对照约束

1. CARLA 同步快照转换到右手系：世界 `(x, -y, -yaw)`；返回项目轨迹为车体原点的
   ego-centric `(x 前, y 左)`。保留真实 footprint 尺寸、轴距，并显式转换 rear axle/车体中心。
2. 路线与可驾驶区域来自 CARLA waypoint 和实际 lane_width。候选包含路线车道和
   地图允许的相邻同向车道；变道路径使用 30m 平滑过渡。禁止两车道跳转、逆向车道、
   禁止变道标线及交叉口内变道。地图不支持请求时日志说明原因。
   规划参考曲线利用 waypoint 真实航向做 Hermite 插值，避免约 2m 折线的切向跳变；
   原验收路点不变。同车道回正与变道分开：回正长度取两倍轴距与当前速度两秒行程的
   较大值，上限为配置的变道长度；实际变道仍使用完整变道长度。两种路径均从实测
   后轴位置/航向开始，经过同样的模拟与基础评分，避免小转弯误差持续外推 30m。
   `geometry.lateral_paths` 中的 `path_kind` / `transition_length_m` 记录本次实际取值。
   版本 5 为当前车道恢复上游 PDM 的平行路径采样：零偏移以及车道内 ±0.25m、±0.5m，
   并按最窄实际车道宽度、自车宽度和 0.1m 余量收紧；窄车道会减少/缩小候选。
   它们共享该车道的横向语义，不能冒充变道；元数据分别记录 `physical_lane_side`、
   `route_target` 和 `lane_offset_m`，例如 `center@+0.25:MAINTAIN`。
   每条路径均从实测后轴姿态平滑衔接，并经过原生完整车身模拟与相同评分。
   这给弯道提供车道内位置选择，不通过放宽道路评分容忍越界。
3. 所有条件恒定包含 STOP、YIELD、DECELERATE、FOLLOW、MAINTAIN、ACCELERATE、
   STOP_HARD 和 YIELD_GENTLE 共八种纵向候选。STOP 与 STOP_HARD 都对应 STOP 语义，分别按
   2m/s²、4m/s² 生成减速至零的速度曲线，均按基础分与语义分比较。
   MAINTAIN 使用限速的 85% 正常巡航，FOLLOW 使用 75% 并增加跟车时距，
   ACCELERATE 使用限速，YIELD 限制到 2m/s。DECELERATE 向当前速度的一半减速，
   当前速度低于 0.6m/s 时按 1.5m/s² 减速至零，不再以 0.3m/s 为最低巡航速度。
   新增 YIELD_GENTLE 同属 YIELD 语义，目标不超过 2m/s，使用 0.2m/s² 的原生 IDM
   加速度参数，让弯道静止起步时有更缓的建立转向过程；它保留原 4m/s² 制动参数。
   原七种策略及其参数不变，常规移动候选仍使用 2m/s²；不是统一降低行驶速度，
   也不强制选择缓起步。日志用唯一 `longitudinal_policy_id` / 索引区分策略，
   同时保存实际 IDM 加减速参数和 headway。两种语义开关始终生成相同策略集合。
   全部仍受配置默认 8m/s 上限与地图限速限制。这些减速度是计划值，实际执行需测量。
   非停车策略继续使用官方 IDM 的速度更新公式，适配层改为不超过 0.01s 的子步积分
   （更低目标速度时继续细分）、非负速度与梯形位移积分，避免原 0.1s Euler 更新在
   低速下振荡、倒退。改动发生在候选生成阶段，全部候选仍经官方模拟器和评分器评估。
4. 最终选择只按 `final_score = base_score + semantic_bonus` 排序。
   `base_score` 为官方 scorer 原始分：碰撞、道路与方向为乘性指标，进度/TTC/舒适性
   权重为 5/5/2。保留这些原生评分；删除适配层独立的碰撞、道路、方向、TTC 完整通过
   门槛和红灯车身相交门槛，也不在选中后调用 `PDMEmergencyBrake` 覆写选择。
   因此基础分为 0 的有限候选仍可参与选择，并可能因语义加分胜出。
   交通灯仍通过 `get_stop_waypoints()` 提供给原生 observation/IDM，影响候选生成；
   不再用车身与停止区域相交单独淘汰候选。初始碰撞 token 和原生风险指标保留作诊断。
5. `semantic_enabled=False` 时语义分为 0，最终分即基础分；选择阶段不再扣连续性分或强制回归参考车道。
   默认横、纵向匹配各加 0.8 分。相同快照的两组候选、基础分和 hash 相同。
   `selection_rule=pdm_base_plus_semantic` 标识本版。`safe` / `safety` 仅为风险诊断，
   `eligible` 只表示轨迹与基础分数值有效，不保证安全。日志保留逐候选基础分、语义分、
   最终分、`baseline_selected_id`、`selected_id`、`selected_zero_base_score`。
   只有无数值有效候选时返回空轨迹；执行前仍校验时间、位置、速度和动力学一致性。

### 连续规划、控制和舒适性诊断

本版不使用上一条规划历史为候选扣分；旧配置中的 `continuity_weight` 不再参与选择。
`planning_score = base_score`，语义开启时仅增加语义匹配分。选择阶段没有额外的停车偏好、
保持动作时长或回归门控。各组独立闭环产生不同状态后，不能要求全程候选 hash 相同。
当前 `provenance.adapter_version=5`，论文应明确这是 CARLA 适配的仅评分选择实验，
不是原版 PDM-Closed 基线；不同适配版本的结果应分开统计。

速度曲线重规划时保留 PID 反馈状态，对实测速度求微分，减少目标变化引起的控制突变；
积分在饱和或误差反向时受限。固定巡航保留速度死区，计划缓加/缓减继续跟踪小误差。
有有效前向路径的计划减速用速度闭环跟踪，静止后保持刹车；
轨迹缺失、过期、无前向目标及紧急失败仍直接全刹。

每轮输出增加 `control.jsonl`：逐 tick 写入实测速度、有符号纵/横向加速度、CARLA
`get_control()` 返回的最近应用的归一化油门/刹车/转向，以及导致该 tick 的上一条控制命令
（包含目标速度、计划加速度、PID 状态、轨迹末端距离、
规划序号、语义帧 ID 和下发时间）。直接全刹单独标记，不能误归因于之前的 PID 目标。
归一化控制量不是实测发动机/制动力，执行效果仍需结合实测速度和加速度判断。
`planner.jsonl` 同时记录附近 actor 的位置/速度、逐候选预测碰撞/TTC 违规时间和速度变化范围，
便于定位“预测要求刹车”与“执行器刹得过猛”。actor 坐标沿用规划器右手系 `(x,-y)`；
`control.jsonl` 的测量使用 CARLA 原生世界坐标。

另记录原生速度向量、有符号前向速度、实际及下发命令的挡位/倒车/手刹状态，以及
`world_snapshot` 的服务器帧号、累计时间和实际步长。它们帮助识别后溜、额外 tick
或执行时序异常；本地调度和指标计时仍保持原有逻辑。`control_context.json` 在每轮
启动时保存世界物理子步配置、车辆质量及各轮最大制动力矩/半径，只读取、不修改。
读取不到的诊断值标为 `null` 并记录原因，不以默认值冒充实际物理参数。

报告保留旧舒适性字段，并明确旧 jerk 只是加速度模长变化率；新增纵向加速/减速与横向
加速度峰值，平面加速度模长的 P95/RMS，平面向量及纵向 jerk 的峰值/P95/RMS，
以及急刹段数、静止时长和采样间隔。
使用相邻运行采样时间差计算；当前同步循环按固定 tick 累加时间，不直接读取服务器快照时间。
保留启动/碰撞/急刹峰值，不通过滤波降低指标。
静止统计包含必要让行，不能直接解释为无必要停车。

服务器先重跑相同 seed 的 `R05_multilane_eval_s101`，检查路线完成率 100%、安全指标与
场景有效性，再对照 `planner.jsonl`、`control.jsonl` 和视频看起步、让行、恢复、转弯过程。
至少同时报告 jerk 分布、急刹与停滞情况；本地数值和回归测试不能证明 CARLA 中已达到自然驾驶。

### 路线几何与规划失败诊断

#### R05 路口停滞的定位依据

用户提供的运行 `78ccccea0cfa456f`（提交 `406a8c7`）实际经过了起始路口，
在后续右转入口停滞；117.1s 时手动中断，路线完成 36.829%，585 次规划，
无碰撞和轨迹拒绝。不能将这轮解释为轨迹校验提前终止或背景交通一定堵死。

| 证据 | 原因与影响 |
|---|---|
| 起步阶段模型多次输出 STOP；后续也有 FOLLOW_ROUTE / MAINTAIN 而规划器仍停车 | 提示词误读与数值规划停滞是不同问题，只改提示词不能修复后者 |
| 65.1s 参考车道移动候选道路评分 0，左邻道候选评分较高并胜出 | 适配层道路漏面使候选排序偏离真实道路条件，车辆被主动选入左道 |
| 左道 `24:0:1` 可直行/左转，右转路线为 `24:0:2 → 706:0:1 → 42:0:2` | 相邻车道不一定能通往目标出口；原候选缺少完整导航连接 |
| 后期所选左道路径不断缩短，最终约 4.35m | 参考路线旁的邻道缺失被当作路径终点，原生 IDM 因虚构终点持续减速停车 |
| 原实车参数与执行侧固定轴距/轮角配置不同 | PDM 模拟与实际跟踪的转弯模型不一致，放大弯道偏差 |

本机使用同为 CARLA 0.9.15 的 Town05，OpenDRIVE SHA256 与归档服务器一致：
`7edc53b0be840e177598da7a4ed9571146d7b2595ce280d154c90f92e8a48851`。
通过空交通、固定正常行驶语义的真实物理闭环又定位出道路末端漏面和单条横向路径的
弯道局限。单帧反事实与完整路线是两级不同证据，不能把某帧能起步当作整条路线已通过。
当前修复保持原 prepared 路线和终点，未通过减少背景交通或改换地图掩盖问题。

固定路线的坐标不能逐点独立投影到最近 waypoint：路口入口、出口可能有数条连接段
几乎重合，得到的车道 ID 序列会在直行和转弯之间跳变，导致合法路线被截断。现在
由 `Waypoint.next()` 的真实后继关系恢复整条路线，保留多个分支，利用后续路点消除
重叠歧义；A*/沿路规划则直接保存原始 waypoint。恢复结果在运行中缓存，导航语义
与 PDM 共用这条拓扑路线。原场景的验收坐标和终点不变，已有场景无需重新生成。

例如本次 Town05 右转应为 `24:0:2 → 706:0:1 → 42:0:2`，独立最近点查询会误插入
`607:0:2` 和 `694:0:2`。原逻辑将参考路径截短后交给 IDM，产生了虚假的路径终点
停车约束。现在参考路线断连会明确报几何错误；真正终止的相邻车道仍保留有限路径
供减速停车。恢复要求每点三维误差不超过 0.75m，不横跳邻道、不切换桥面。
`geometry.reference_resolution` 记录恢复方法、最大误差和与独立投影不一致的位置，
`geometry.lateral_paths` 记录实际路径长度；候选 `drivable_area_failure` 记录首次道路
检查失败的相对时间及预测车身四角，坐标为规划器右手系。

可行驶区域与导航路线分开构建。对于路线经过的路口，使用 `get_junction()` 中全部
真实 Driving 连接车道的宽度与后继采样补齐道路，并按局部路线高度排除其它桥面。
仅增加实际路面，不用路口包围框/凸包填满人行道，也不将这些额外连接加入导航目标
或授予变道许可。`geometry.junction_drivable_area` 记录路口 ID、采样数及高度排除数。
这样右转车身经过真实直行路面时不会误判越界；真实越界仍反映在官方基础分和诊断中，
不再作为额外候选资格门槛。

路口内可能有多条连接道路在同一位置重叠，不能只凭最近 waypoint 的 road/lane ID
不同就认定车辆离开路线。ID 匹配失败时，增加局部路线几何复核：只检查含路口的
连续车道段，车辆中心必须位于车道内（距边缘至少 0.05m）、航向差不超过 30°、
高度差不超过 1m；多个候选距离相差不足 0.25m 时仍拒绝。诊断 `geometry.lane_match`
记录匹配方式、距离和实际用于匹配的路线车道 ID，保留原始地图查询 ID。
折线弯道的共同内部顶点也参与距离匹配，避免两侧垂足都超出线段时留下匹配空隙；
只有实际后继连接的内部顶点适用，路线真实端点不向外延长。
这只处理路口位置查询歧义；变道路径生成仍要求实际车道连接和地图许可。

局部车道现在从自车所在位置向前、向后建立连续片段；自车后方缺失邻道不再使
整条候选失效。跨 road/section/lane 的连接通过 `Waypoint.next()` 验证，避免仅按
“参考路线左侧/右侧”拼接不相连的道路。前方车道终止时保留有限长度的路径，
由 IDM 的路径终点约束减速，短于完整变道所需距离的车道不能成为新的变道目标。

选择阶段不增加必要回归门槛，`KEEP_LANE` 与参考路线候选仍按相同公式竞争。
版本 5 修正了候选本身的导航完整性：相邻车道若继续通向直行/左转，却不能进入本路线的
右转出口，空间候选必须包含在地图许可区域内返回参考车道的过程。不能把“参考路线
旁边的车道消失”误当作该物理车道终点，也不能把左转车道直接接到右转出口。
新邻道变道只有在驶入和返回都能完成时才生成；已在邻道的候选采用固定地图位置的返回段，
避免每 0.2s 重规划都重新要求从头开始 30m 变道。来不及合法返回时，明确记录失败并
保留有限停车路径，不伪造通行连接。这是**候选几何的变更**，不是新增选择分数或评分门槛。
`geometry.lateral_paths.*.route_connection.method=legal_return_in_spatial_proposal` 记录返回过程；
`KEEP_LANE` 表示沿当前空间候选行驶，该候选在出口前可能包含导航所需的合法返回。
相同快照下开关两侧使用相同的几何、候选和基础分。真实车道结束、后继断连和普通变道
许可继续用于构造可用路径，不拼接不存在的道路。
兼容日志中的 `route_recovery_active` 恒为 `false`，`route_eligible` 不再筛选；
`geometry.lateral_paths` 记录实际车道截断及变道许可诊断。

版本 5 同时补齐了实测发现的两类道路漏面：出口邻道在第一个参考路点之前已经存在的
真实路面，以及道路长度不是 0.5m 整数倍时末段不足 0.5m 的路面。
`road_area.py` 按 OpenDRIVE 的实际 road/laneSection 区间采样 Driving 车道，包含精确区间
端点，并处理 CARLA 浮点站点查询的边界。它只补实际车道宽度，不膨胀 polygon、不填满
路口包围框。`geometry.road_drivable_area` 记录采样方法及 `polygon_expansion_m=0`。
这个零膨胀字段只描述新增补面；历史稀疏车道带的 0.025m buffer 保持未变。
修复前，部分车身角点经 CARLA `project_to_road=False` 查询仍在 Driving 车道内，
适配层却给出道路评分 0；这会压低所有移动候选的原生基础分，间接使停车候选胜出。

末段弯道还存在独立的低速起步问题：固定 1s 的上游 LQR 预瞄，在真实轮角已回零的
静止姿态下，对所有移动候选预测出道路评分 0，而停车评分为 1。这里的失败预测角点
经全部真实 Driving 车道并集核对，确实超出路面约 1.72mm，不属于前述漏面；不能通过
膨胀路面或强制选中移动轨迹消除。仅降低加速度、缩短空间恢复距离、加密路径采样，
或保留停车前最后约 4.84° 的轮角，都未在该姿态得到可行移动候选。

直接将上游共用预瞄改为 2s，虽能在局部测试及两组完整路线中通过，却同时改变了
纵向参考速度和 LQR 纵向增益，削弱预测制动：原本能通过 TTC 的急刹候选被预测碰撞。
因此这次反事实方案已弃用；不能用路线完成掩盖制动回归，也没有修改原 TTC 测试来放行。

最终适配器 `lqr_tracker.py` 分别调用原生求解器：纵向保留原来的 1s 参考、增益及停车
判断，横向使用 2s 曲率预瞄。只分离预瞄仍不足以从该姿态起步，配合新增的原生
YIELD_GENTLE 候选，才出现道路评分为 1 的移动轨迹。没有向初始状态注入预转角，
也没有修改上游源码、扩大道路或覆盖评分。独立 CARLA 物理尾段测试从实测轮角 0°
出发，8.2s 完成剩余 10.61m，终点距 1.524m；41 次规划仅 3 次选择 gentle，随后
恢复常规候选，碰撞、轨迹拒绝和清理错误均为 0。它只证明该局部状态可以恢复。
冻结快照中 gentle 的原生有效进度约 1.514m，超过上游既有 0.1m 归一化阈值，
进度、TTC、舒适项均为 1，基础分为 1.0；停车进度为 0，基础分降为 7/12≈0.5833。
关闭语义影响时也有明确 0.4167 分差，不依赖浮点平局、额外停车惩罚或强制起步。

分轴后还修正横向线性化的制动边界：把 1s 制动加速度恒定外推到 2s，可能预测出
停车后的反向行驶，进而在弯道预测反向转向。适配器仅在进入原生横向求解器前将
有限负未来速度截为 0；不改纵向命令、不改真实车辆物理，也不吞掉 NaN/无穷值。
4m/s、半径 8m 的左右弯硬停车反事实均验证了方向恢复，正常加速输出逐值不变。
这是修正横向预瞄假设，不能描述成原生 bicycle 模型已强制速度非负。
诊断 `lqr_lateral_velocity_forecast` 记录实际使用的方法。

横向配置为 `carla.planning.lqr_lateral_horizon_s`，默认 2.0；必须是 0.1s 的整数倍、
至少两个采样且不超过规划时域。纵向固定 1s，不随横向参数改变。每帧保存
`lqr_longitudinal_horizon_s/samples` 和 `lqr_lateral_horizon_s/samples`。
独立回归逐值比较软/硬停车的整段纵向模拟与原生 1s 结果，并保持原 TTC 软 0 / 硬 1
的断言；更长横向预瞄也不保证更好，需要对其它场景重新验收。

几何构建异常现在也写入 `semantic_error` / `control_error`，包含当前/参考车道 ID、
候选排除原因和位置。无数值有效候选时，兼容的 `pdm_emergency_brake` 审计原因对应
`details.failure_reason=no_finite_candidate`，不表示旧的道路/碰撞硬门槛触发。
连续拒绝退出阈值、禁止写入失败决策记忆的规则保持生效。

本轮本地 `carla_bridge/tests` 共 726 项回归全部通过，使用固定 PDM 核心，且设置了
真实 Town05 OpenDRIVE 路径，实际运行了可选地图回归。覆盖仅评分选择、原生风险
诊断、有限车道停车、连接检查、语义对照一致性、制动预测和数值失败审计。
数值回归不能代替真实物理闭环，更不能代替服务器上的完整 VLM、交通与记忆实验；
后者仍需重跑 `R05_multilane_eval_s101`，按序完成路线 100% 并审查实际驾驶指标。

### 本地完整路线物理验证（2026-10-18）

最终修复在真实 CARLA 0.9.15 / Tesla 物理闭环中，分别完成语义影响开启、关闭两组
`R05_multilane_eval_s101`。保留原 Town05、约 922.087m 的有序路线、seed 101、
ClearNoon、60 辆背景车和 20 名行人；未替换成空场景或缩短验收路线。

| 项目 | 语义影响开启 | 语义影响关闭 |
| --- | ---: | ---: |
| 按序完成路线并到达终点 | 100% | 100% |
| 仿真时长 | 316.2s | 313.6s |
| 记录碰撞 / 轨迹拒绝 / 清理错误 | 0 / 0 / 0 | 0 / 0 / 0 |
| 平均速度 | 2.903m/s | 2.875m/s |
| 速度低于 0.2m/s 的累计时长 | 97.4s | 74.2s |
| 原始最大纵向减速度 | 26.85m/s² | 27.41m/s² |
| 原始最大绝对纵向 jerk | 268.05m/s³ | 298.96m/s³ |

两组运行期间功能源码保持冻结；场景、路线、天气、车辆与行人初始配置、车辆模型和
首帧候选池一致，规划参数仅 `semantic_influence` 不同，背景车辆实际运动。
场景配置、地图与路线哈希，以及 60 辆背景车的初始蓝图和位置也与原服务器记录
核对一致；原服务器的行人导航网格采样位置不完全相同，因此不属于原动态交通的
逐帧重放。两次独立物理运行在控制选择不同后，也会自然产生不同的交通交互。

这些是**固定 FOLLOW_ROUTE + MAINTAIN 的工程诊断**，没有调用真实 VLM、
加载视觉特征模型或读写记忆库。它们支持“修复后的规划与控制链能在原配置背景交通下
完成整条路线”，不能用于证明语义模型收益、记忆收益或服务器完整系统已经验收。
原始急减速由相邻真实速度差分复核，不能用滤波或候选舒适评分掩盖；当前仍有较长
等待，不能宣称驾驶效率、平顺性或零压线已经达标。

本地证据保存在 `outputs/pdm_stall_analysis/final8_traffic_on_v8/`、
`final8_traffic_off_v8/` 及 `final8_pair_readonly_audit.json`；每组包含报告、逐帧控制、
候选评分和实验配置。`final8_vs_original_setup_audit.json` 保存与原服务器配置的核对。
`outputs/` 为本地忽略目录，推送代码不会上传这些运行证据。

流畅性也不能用“没有异常退出”代替验证。若相邻更新出现 STOP / 加速 / YIELD 快速
切换，应对照逐候选基础分、语义分和风险指标，定位排序变化原因。还需对照实际速度、加速度、
控制量与规划速度剖面，检查运动预测、规划模型与 CARLA 执行差异。当前局部候选
筛选和 PID 跟踪尚不能据单元测试认定具备稳定、自然的驾驶表现。

适配层维持 0.1s 轨迹采样（上游 TTC 计算使用 10Hz 下标）；PDM 更新频率由闭环
`carla.planning.replan_interval_s` 控制，默认 0.2s。VLM 语义更新仍为 3s。
为满足 100% 路线完成，规划中心线在固定验收终点外沿真实道路适度延伸，验收路线
与终点不变，闭环在实际达到终点时结束，避免 IDM 因“路径尽头前车”提前停住。

## 实验边界

- 本模块是 GT actor/地图输入，动态目标沿官方常速度模型预测；不是纯视觉感知实验。
- 当前道路来自 CARLA OpenDRIVE Driving 车道区间的采样、局部 waypoint 车道带及
  实际路口连接，并非 nuPlan 地图 polygon。
  现有论文路线按约 2m 采样；任意稀疏路线尚不支持，不能将 6m 及以上点距视为可用配置。
  三维道路以局部道路高度过滤其它桥面目标后投影到平面，复杂立交、斜坡/坡顶、
  急弯及车道终止仍需服务器预跑。保守约束可导致安全停车，不能以单元测试替代路线验收。
- 自车 wheelbase 来自 CARLA wheel positions；CARLA 0.9.15 的轮坐标为世界厘米，
  先变换到 actor 坐标，计算实际 rear-axle 到包围盒中心的距离，不再默认 wheelbase/2。
  可通过 `rear_axle_to_center_m` 显式覆盖，诊断同时保存测量值与来源。
  CARLA/PhysX 的归一化转向指令对应内侧前轮角，PDM bicycle 和 Pure Pursuit 使用
  等效中心转角。共享 `SteeringModel` 根据实际前轮距、轴距和速度转向曲线作正反换算；
  Tesla 的物理内轮上限约 70°，等效中心上限约 57.326°，两者在诊断中分别保存。
  官方 LQR 绑定实际轴距，bicycle 绑定实际等效中心角上限，执行侧将期望中心角逆换算为归一化指令，
  语义自车状态使用相同模型估计中心角。这个估计仍不是实测反馈。
  PDM 初始状态优先读取同步帧两个真实前轮角并换算为中心角，记录
  `steering_state.source=carla_measured_front_wheels`。仅 API 缺失时才明确警告并记录
  `command_ackermann_estimate` 回退；读取错误或异常数据直接报错，不静默替代。
  输入平滑/迟滞及轮胎侧滑尚未建模，记录 `input_latency_modeled=false`。
  此映射按默认 Tesla 的完整 Ackermann 几何验证，其它车辆仍需单独核查。
  内外轮定义可对照 [PhysX 3.4 的 computeAckermannSteerAngles](https://github.com/NVIDIAGameWorks/PhysX-3.4/blob/master/PhysX_3.4/Source/PhysXVehicle/src/PxVehicleUpdate.cpp)，
  本车另以实测两前轮角验证，没有把归一化指令当作当前实测转角。
  外部轨迹仍是 actor 原点坐标，新增 `optional_heading_rad` 表示相对捕获车身、左正的
  车身朝向。PDM 绑定后，Pure Pursuit 同时把轨迹与当前位姿换到后轴；不能以轨迹切线
  代替车身朝向，也不能只变换当前位姿。Tesla 的 actor→后轴约 −1.3864m，
  bbox 中心→后轴约 1.4156m，两者不可混用。绑定后缺失/无效朝向会明确报错并清轨迹；
  旧轨迹模式不要求此字段。以上均未修改 CARLA 车辆物理参数。
  CARLA 实际执行器/Pure Pursuit 与 PDM 内部 LQR/bicycle 模拟仍有差异，需要闭环验证。
- 安全评分基于模拟 rollout，最终返回上游生成轨迹及 IDM 速度剖面；现有物理轨迹校验
  在执行前继续生效。保留独立速度剖面，不把终点速度当作整段恒速。
- 本地测试包括 CARLA road/actor 测试替身及可选的真实 Town05 OpenDRIVE 道路，
  均调用固定 PDM 核心。设置 `CARLA_TOWN05_XODR` 为本机 `Town05.xodr` 的绝对路径，
  可运行 `python -m pytest carla_bridge/tests/test_route_waypoint_resolution.py carla_bridge/tests/test_junction_area.py -q`；
  该地图测试不需要启动 CARLA 服务端，不包含车辆物理和动态交通。这些测试不代表已完成
  服务器 Town03/04/05/10HD 的路线验收。服务器仍以路线完成率 100% 为成功条件。
- 真实 Town05 连续右转回放保留窄车和 4.8×2.2m、40° 的假设宽车两组。
  后者曾因单条横向路径而停滞，版本 5 加入车道内候选后也能完成完整弯道，
  已取消其历史 `xfail`，继续保留通行断言和道路评分检查。
  这组假设参数不能冒充服务器实车参数；默认 Tesla 实测为约 4.79178×2.16345m、
  最大轮角约 70°、轴距约 3.00465m、后轴到包围盒中心约 1.41564m。
  合成执行、静态冻结帧反事实和真实物理闭环是三类不同验证，均不能代替服务器上的
  完整 VLM、交通与记忆实验，也不能证明其它地图和所有车辆已经验收。

### 已知的纵向物理限制

路线完成不等于舒适性合格。独立 CARLA 0.9.15 / Tesla 直路诊断在没有 PDM、PID、
VLM 或背景交通参与时，约 5m/s 施加固定 `brake=0.2` 持续 0.5s，仍测得约
19.88m/s² 的最大减速度；单 tick 制动释放后也有明显延迟响应。归零轮制动扭矩、
关闭发动机制动和改变轮胎摩擦的反事实说明轮刹/轮胎动力学是关键因素；没有内部
轮速与接触力记录，具体抱死机制尚未确证。不能把这些尖峰解释成统计公式错误。

同一速度剖面的原生 Ackermann 和两种加速度闭环对照均未同时改善停车跟踪与峰值，
因此没有将失败的控制器替换进生产。生产继续使用原纵向 PID，车辆物理和原始指标
不变；其实际减速度不能由 PDM 轨迹中的加速度约束来保证。服务器还需单独验证
原始加速度、jerk、停车距离及恢复起步，当前不能宣称已实现自然、舒适驾驶。

另外，`get_waypoint(project_to_road=False)` 的严格查询失败不等于精确的路面并集越界；
道路接缝处仍可能有最近站点误差。车身四角查询、候选预测道路分和真实压线/车道侵入
是不同指标，不能相互替代或通过修改道路容差得到“零越界”。

## 第三方来源与许可

上游源码作为外部依赖安装，没有复制进本项目，也不受本项目 MIT 许可重新授权。
tuplan_garage 的作者为 Daniel Dauner、Marcel Hallgarten、Andreas Geiger、Kashyap Chitta，
采用 [Apache-2.0](https://github.com/autonomousvision/tuplan_garage/blob/b51d5d04fac1bd4389653b9ab2ff73ea88f435a3/LICENSE)；
nuPlan devkit 采用其 [Apache-2.0 许可](https://github.com/motional/nuplan-devkit/blob/e9241677997dd86bfc0bcd44817ab04fe631405b/LICENSE)。
研究引用请使用原作者论文 *Parting with Misconceptions about Learning-based Vehicle Motion Planning*
及 nuPlan 的原始论文，并明确说明本项目的 CARLA 适配和语义评分改动。
