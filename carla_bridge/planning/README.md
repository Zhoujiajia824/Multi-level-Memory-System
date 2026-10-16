# PDM-Closed 的 CARLA 适配层

这里直接调用官方 `PDMGenerator`、`PDMSimulator`（LQR + bicycle）、
`PDMScorer`、`PDMObservation` 和 `PDMEmergencyBrake`，没有将简化规则重命名为 PDM。
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
3. 所有条件恒定包含 STOP、YIELD、DECELERATE、FOLLOW、MAINTAIN、ACCELERATE
   和 STOP_HARD 共七种纵向候选。STOP 与 STOP_HARD 都对应 STOP 语义，分别按
   2m/s²、4m/s² 生成减速至零的速度曲线；两者都先通过当前安全检查再比较评分。
   MAINTAIN 使用限速的 85% 正常巡航，FOLLOW 使用 75% 并增加跟车时距，
   ACCELERATE 使用限速，YIELD 限制到 2m/s。DECELERATE 向当前速度的一半减速，
   当前速度低于 0.6m/s 时按 1.5m/s² 减速至零，不再以 0.3m/s 为最低巡航速度。
   已静止且可以安全接近/探查时应使用 YIELD，正常起步使用 MAINTAIN。
   全部仍受配置默认 8m/s 上限与地图限速限制。这些减速度是计划值，实际执行需测量。
   非停车策略继续使用官方 IDM 的速度更新公式，适配层改为不超过 0.01s 的子步积分
   （更低目标速度时继续细分）、非负速度与梯形位移积分，避免原 0.1s Euler 更新在
   低速下振荡、倒退。改动发生在候选生成阶段，全部候选仍经官方模拟器和评分器评估。
4. 官方 scorer 给出碰撞、道路、行驶方向、TTC、进度及舒适性指标。
   本适配层要求碰撞/道路/方向/TTC 全部通过后才能参与选择；初始真实碰撞不会因
   官方 observation 的已碰撞缓存而被忽略。交通灯使用 `get_stop_waypoints()`，
   红灯及黄灯停止线另外做候选硬检查，禁止变道绕过红灯。
5. `semantic_enabled=False` 时仅去掉语义分，其余候选、预测、评分和安全条件相同。
   日志有 `candidate_pool_sha256`、逐候选轨迹 hash/基础分/连续性扣分/语义分/安全项/最终分、
   `baseline_selected_id` 和 `selected_id`，可直接检查当次选择是否受到语义影响。
   语义无法复活不安全候选；全部失败或官方 emergency 触发时返回空轨迹，由闭环全刹。

### 连续规划、控制和舒适性诊断

`carla.planning.continuity_weight` 默认 0.15，可设为 0 关闭。连续性项只在通过安全和
路线约束的候选中参与排序：按绝对仿真时间对齐上次已选择轨迹，比较未来 1s 的横向位置、
航向和速度偏差，扣分不超过该权重。`planning_score = base_score - continuity_cost`，
然后按实验开关决定是否添加语义分；`base_score` 保留上游原始分数。
上一轨迹超过 0.5s、时钟回退或参考无效时不使用该项，不强制保持动作或限制紧急制动。

这是两组共用的短时规划执行状态，不查询或写入认知记忆库，`memory_off` 也使用相同机制。
同一快照重复进行语义开/关评估时冻结同一历史参考；独立闭环运行产生不同状态和历史后，
不能要求两个实验全程候选 hash 相同。新增 `provenance.adapter_version=2`，论文应明确
这是经过 CARLA 适配、数值积分和连续性评分修改的 PDM 核心，不是原版 PDM-Closed 基线。

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

报告保留旧舒适性字段，并明确旧 jerk 只是加速度模长变化率；新增纵向加速/减速与横向
加速度峰值，平面加速度模长的 P95/RMS，平面向量及纵向 jerk 的峰值/P95/RMS，
以及急刹段数、静止时长和采样间隔。
使用相邻运行采样时间差计算；当前同步循环按固定 tick 累加时间，不直接读取服务器快照时间。
保留启动/碰撞/急刹峰值，不通过滤波降低指标。
静止统计包含必要让行，不能直接解释为无必要停车。

服务器先重跑相同 seed 的 `R05_multilane_eval_s101`，检查路线完成率 100%、安全指标与
场景有效性，再对照 `planner.jsonl`、`control.jsonl` 和视频看起步、让行、恢复、转弯过程。
至少同时报告 jerk 分布、急刹与停滞情况；本地数值和回归测试不能证明 CARLA 中已达到自然驾驶。

### 相邻车道回归与规划失败诊断

路口内可能有多条连接道路在同一位置重叠，不能只凭最近 waypoint 的 road/lane ID
不同就认定车辆离开路线。ID 匹配失败时，增加局部路线几何复核：只检查含路口的
连续车道段，车辆中心必须位于车道内（距边缘至少 0.05m）、航向差不超过 30°、
高度差不超过 1m；多个候选距离相差不足 0.25m 时仍拒绝。诊断 `geometry.lane_match`
记录匹配方式、距离和实际用于匹配的路线车道 ID，保留原始地图查询 ID。
这只处理路口位置查询歧义，不放宽碰撞、可行驶区域或变道检查。

局部车道现在从自车所在位置向前、向后建立连续片段；自车后方缺失邻道不再使
整条候选失效。跨 road/section/lane 的连接通过 `Waypoint.next()` 验证，避免仅按
“参考路线左侧/右侧”拼接不相连的道路。前方车道终止时保留有限长度的路径，
由 IDM 的路径终点约束减速，短于完整变道所需距离的车道不能成为新的变道目标。

邻道终止、不连续、即将进入路口或失去回归变道许可时，计算回归边界。
边界进入 `lane_change_length_m + 4m + max_speed_mps × horizon_s` 的范围后，
若存在通过全部安全检查、规划窗口内沿路线前进超过 2m 且末速度超过 0.2m/s 的参考车道候选，
暂时禁止继续选择会错过该边界的邻道候选。`KEEP_LANE` 加分不能推迟必要回归。
若回归被阻挡，则保留其它安全候选；前方已知的回归边界也作为邻道路径终点，
供 IDM 减速停车。已经错过合法回归区间时不会强行跨实线或在路口内变道。

这是 CARLA 适配层新增的路线约束，不是原版 PDM 的功能。语义开/关都使用同样
规则，且规则只依赖当前地图、状态和候选，不读取历史决策或记忆库。`planner.jsonl`
新增 `route_recovery_active`、候选的 `route_eligible` / `eligible`、`geometry.lateral_paths`
中的截断位置和回归边界。`safe` 仍单独表示安全检查结果，`baseline_selected_id`
也在相同路线约束下计算；绘图时应使用 `eligible` 判断是否可参与最终选择。

几何构建异常现在也写入 `semantic_error` / `control_error`，包含当前/参考车道 ID、
候选排除原因和位置。无安全候选或上游紧急制动记录为 `pdm_emergency_brake`，
`details.failure_reason` 区分 `no_safe_candidate` 和 `upstream_emergency`，不再仅报
“空轨迹格式错误”。连续拒绝退出阈值、禁止写入失败决策记忆的规则保持生效。

本地回归使用固定 PDM 核心验证连续重规划、临近路口回归、有限车道停车、
连接检查、语义对照一致性及失败审计；仍需服务器重跑 `R05_multilane_eval_s101`
并达到路线完成率 100%，不能将本地测试通过视为原故障场景已经验收。

流畅性也不能用“没有异常退出”代替验证。若相邻更新出现 STOP / 加速 / YIELD 快速
切换，应先对照逐候选安全指标，区分语义偏好变化和安全可行集合变化；当上一条
轨迹已经不安全时，单纯增加动作保持时间会掩盖问题。还需对照实际速度、加速度、
控制量与规划速度剖面，检查运动预测、规划模型与 CARLA 执行差异。当前局部候选
筛选和 PID 跟踪尚不能据单元测试认定具备稳定、自然的驾驶表现。

适配层维持 0.1s 轨迹采样（上游 TTC 计算使用 10Hz 下标）；PDM 更新频率由闭环
`carla.planning.replan_interval_s` 控制，默认 0.2s。VLM 语义更新仍为 3s。
为满足 100% 路线完成，规划中心线在固定验收终点外沿真实道路适度延伸，验收路线
与终点不变，闭环在实际达到终点时结束，避免 IDM 因“路径尽头前车”提前停住。

## 实验边界

- 本模块是 GT actor/地图输入，动态目标沿官方常速度模型预测；不是纯视觉感知实验。
- 当前道路是由离散 waypoint 形成的局部车道带，并非 nuPlan 地图 polygon。
  三维道路以局部道路高度过滤其它桥面目标后投影到平面，复杂立交、斜坡/坡顶、
  急弯及车道终止仍需服务器预跑。保守约束可导致安全停车，不能以单元测试替代路线验收。
- 自车 wheelbase 来自 CARLA wheel positions，rear-axle 到中心默认约为 wheelbase/2，
  可通过 `rear_axle_to_center_m` 校准并写入日志；初始轮角由当前 steer 指令与最大轮角估计。
  CARLA 的实际执行器/Pure Pursuit 与 PDM 内部 LQR/bicycle 模拟存在差异，需要闭环验证。
- 安全评分基于模拟 rollout，最终返回上游生成轨迹及 IDM 速度剖面；现有物理轨迹校验
  在执行前继续生效。保留独立速度剖面，不把终点速度当作整段恒速。
- 当前本地测试使用假的 CARLA road/actor 接口加真实的固定 PDM 核心，不声称已完成
  服务器 Town03/04/05/10HD 的路线验收。服务器仍以路线完成率 100% 为成功条件。

## 第三方来源与许可

上游源码作为外部依赖安装，没有复制进本项目，也不受本项目 MIT 许可重新授权。
tuplan_garage 的作者为 Daniel Dauner、Marcel Hallgarten、Andreas Geiger、Kashyap Chitta，
采用 [Apache-2.0](https://github.com/autonomousvision/tuplan_garage/blob/b51d5d04fac1bd4389653b9ab2ff73ea88f435a3/LICENSE)；
nuPlan devkit 采用其 [Apache-2.0 许可](https://github.com/motional/nuplan-devkit/blob/e9241677997dd86bfc0bcd44817ab04fe631405b/LICENSE)。
研究引用请使用原作者论文 *Parting with Misconceptions about Learning-based Vehicle Motion Planning*
及 nuPlan 的原始论文，并明确说明本项目的 CARLA 适配和语义评分改动。
