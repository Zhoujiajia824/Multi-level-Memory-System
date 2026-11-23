# CARLA v2 场景与配对实验

本分支采用一个统一场景执行器，替换旧五模板及 scripted_traffic。每个场景仍通过 `--scenario-config 文件路径` 单独运行。主程序使用 `ep-driver-carla`，CARLA SDK 桥接使用 `carla37`。

**当前目录是可执行的候选集，不是已通过服务器物理闭环验收的正式测试集。** 22份 YAML 覆盖12个Town；真实CARLA 0.9.15 SDK离线OpenDRIVE已完成路线、演员布置、地图指纹和高度支持验证。服务器蓝图、车辆动力学、信号、传感器和多轮清理仍需验收。

场景坐标统一为 `carla-route-xy-bbox-1.0`：路线进度是冻结路点折线在XY平面的累计弧长，车辆/行人进度使用世界包围盒中心；高度只用于道路分层与几何检查。路点中的 `s_m` 始终是OpenDRIVE道路参数，不能与路线累计进度混用。候选资产已将早期3D累计弧长值一次转换为同路段XY弧长，转换记录保存在各路线JSON；旧冻结实例与验收报告不能继续复用。

VLM与EKF候选为车道范围内全部交通对象，加当前BEV矩形内全部车辆、自行车、行人；MPC势场及碰撞优化目标为车道范围内全部对象，加模型最多3个关注对象。八向最近代表主要用于检索，不截断前两类集合；质量与执行前审计独立检查全部预测候选。输入仅含当前真值、因果观测历史、自车公开导航和公共道路规则，不判断遮挡。NPC脚本、触发时刻、角色意图、任务判据和独立评价仅保存到私有审计文件，不进入当前观测资产或SceneSnapshot。模型原帧ID使用不含场景名称的哈希命名空间；实验目录保留可追溯映射。

## 场景目录

| YAML | 地图 | 路线/m | 家族 |
|---|---|---:|---|
| `town01_follow_stop_go.yaml` | Town01 | 403.0 | following |
| `town02_pedestrian_zone.yaml` | Town02 | 340.5 | pedestrian_crossing |
| `town03_roundabout.yaml` | Town03 | 400.9 | roundabout |
| `town03_signal_left.yaml` | Town03 | 361.4 | signal_junction |
| `town03_signal_right.yaml` | Town03 | 361.5 | signal_junction |
| `town03_signal_straight.yaml` | Town03 | 362.0 | signal_junction |
| `town04_overtake_return.yaml` | Town04 | 587.5 | overtake_return |
| `town05_lead_hard_brake.yaml` | Town05 | 358.7 | emergency_braking |
| `town05_roadworks.yaml` | Town05 | 476.7 | roadworks |
| `town05_roadworks_wait.yaml` | Town05 | 476.7 | roadworks |
| `town05_urban_mixed.yaml` | Town05 | 174.0 | multi_vehicle |
| `town06_cut_in_left.yaml` | Town06 | 450.0 | cut_in |
| `town06_cut_in_right.yaml` | Town06 | 450.0 | cut_in |
| `town06_ramp_exit.yaml` | Town06 | 794.5 | ramp_exit |
| `town06_ramp_merge.yaml` | Town06 | 817.6 | ramp_merge |
| `town07_priority_crossing.yaml` | Town07 | 351.7 | priority_junction |
| `town07_rural_curve.yaml` | Town07 | 600.5 | rural_curve |
| `town10hd_queue.yaml` | Town10HD | 500.9 | multi_vehicle |
| `town11_grade_follow.yaml` | Town11 | 650.0 | rural_curve |
| `town12_continuous_interaction.yaml` | Town12 | 650.0 | multi_vehicle |
| `town13_roadworks.yaml` | Town13 | 500.4 | roadworks |
| `town15_pedestrian_zone.yaml` | Town15 | 355.7 | pedestrian_crossing |

信号路口直行、左转、右转及左右切入分别配置。Town03环岛路线经过中心环行区再驶出；Town06汇入有独立主路车辆路线；Town07交叉口通过真实路网检索确认接近区域没有交通信号灯。环岛、汇入与无信号路口的 `public_rules` 是公开实验路权，各组一致，不把周车未来脚本传给规划器。路线终点、道路序列和具体出口已经锁定。

`town05_urban_mixed` 是60秒短城市交互候选：自车加3辆交通车、2名正常步速行人；双车道跟车及从左车道合法超越右车道慢车，宽弯左转进入单车道后过街让行，再遵守原生STOP并左转汇入主路，保留约32米终段恢复行驶。两个左转的车道中心线半径约12.4米、9.7米。此处汇入是城市左转进入主路，不是高速匝道；左侧超越不要求为超车变道。前方同车道车和右侧慢车在首个路口直行分流；主路车与行人按已观测自车进度启动，动作从下一物理帧生效。主路车以其自身实测路线进度判定事件完成，完成后仍继续行驶；所有交通车的路线终点都远离自车终点。超车不作为到达的必做条件。该场景仍需真实车辆、行人和STOP闭环验收，60秒是运行上限，不是成功保证。

学校式场景称为低速行人区；使用范围内真值、不判断遮挡，不能将这套结果解释为遮挡感知能力。`_Opt`地图可另编译同任务兼容性实例，不计为独立场景数量。

## 时钟、输入与录像

- 世界50Hz，控制段、事件、历史与结束条件使用整数物理帧。规划频率在 `config/carla.yaml` 的 `planner_hz` 选择1、2或10；正式主表固定同一频率。
- 六路RGB每物理帧实际采集50Hz，逐帧收齐后再推进；每五帧选择同帧六视角供10Hz消费与录像，原图每秒归档；表示/模型源帧1Hz。没有深度相机。同步和异步都保留图像屏障，真实墙钟吞吐需另测。
- 正式实验使用同步模式：算法计算期间世界暂停。异步保留为开发/单独时延实验，不混入同步结果。
- `--video --no-world-plan` 录制展示拼图，可在显示副本叠加轨迹；原始RGB不被世界debug线污染。旁观脚本只能观察，不能tick或修改世界。
- `video_recording.json`报告实际编码与解码状态，`video_frames.jsonl`记录图像帧和仿真时间；缺帧单列。录像完整不等于驾驶成功。
- RGB的原始帧号与时间戳必须同时匹配当前世界。若旧帧号携带当前时间戳，立即报告 `rgb_metadata_conflict` 并保留原始头部，不将其当普通旧图丢弃后等待30秒，也不改帧号或额外推进找图。CARLA 0.9.15的异步GPU回读分别生成帧号与后续流头部，时间戳/位姿相同不足以证明像素同帧。这项检查准确定位异常，不代表修复了服务器渲染端的元数据生成问题。
- 世界准入区分地图原生网格与SDK生成道具：只有 `static.prop.mesh` 严格符合0.9.15原生注册属性（仅有有效 `mesh_path`、`mass`，无父对象）才记录为地图对象并保留；空角色名本身不是豁免依据。其他道具、车辆、行人及已有同步世界仍阻止接管。分类证据见 `carla_world_ownership.json`，轮末清理仍只销毁本轮自建对象。

## 开发阶段单独运行

先执行环境和目录命令：
```bash
conda activate ep-driver-carla
cd /data/zjj/code/EP-Driver/EP-Driver-CARLA
```

下面每段都是独立命令，三轮使用三个参数实例；最长时限读取各自YAML。初次检查可添加 `--duration 3`，但短检查不能评价整个交互任务。无需设置SCENE或COMMON_ARGS变量。

### town01_follow_stop_go

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town01_follow_stop_go.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town01_follow_stop_go
```

### town02_pedestrian_zone

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town02_pedestrian_zone.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town02_pedestrian_zone
```

### town03_roundabout

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town03_roundabout.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town03_roundabout
```

### town03_signal_left

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town03_signal_left.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town03_signal_left
```

### town03_signal_right

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town03_signal_right.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town03_signal_right
```

### town03_signal_straight

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town03_signal_straight.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town03_signal_straight
```

### town04_overtake_return

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town04_overtake_return.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town04_overtake_return
```

### town05_lead_hard_brake

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town05_lead_hard_brake.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town05_lead_hard_brake
```

### town05_roadworks

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town05_roadworks.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town05_roadworks
```

### town05_roadworks_wait

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town05_roadworks_wait.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town05_roadworks_wait
```

### town06_cut_in_left

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town06_cut_in_left.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town06_cut_in_left
```

### town06_cut_in_right

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town06_cut_in_right.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town06_cut_in_right
```

### town06_ramp_exit

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town06_ramp_exit.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town06_ramp_exit
```

### town06_ramp_merge

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town06_ramp_merge.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town06_ramp_merge
```

### town07_priority_crossing

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town07_priority_crossing.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town07_priority_crossing
```

### town07_rural_curve

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town07_rural_curve.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town07_rural_curve
```

### town10hd_queue

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town10hd_queue.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town10hd_queue
```

### town11_grade_follow

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town11_grade_follow.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town11_grade_follow
```

### town12_continuous_interaction

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town12_continuous_interaction.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town12_continuous_interaction
```

### town13_roadworks

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town13_roadworks.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town13_roadworks
```

### town15_pedestrian_zone

```bash
env CUDA_VISIBLE_DEVICES="" python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/town15_pedestrian_zone.yaml \
  --runtime-mode sync --exclusive-world --allow-world-reset \
  --method A --fast off --reflection off --no-retrieve --no-write-memory \
  --episodes 3 --seed 1 --seed-mode increment \
  --video --no-world-plan \
  --episodes-dir outputs/carla/v2/development/town15_pedestrian_zone
```

## 冻结同一实例进行A/B/C比较

同seed不是充分的配对证据。先冻结一次参数和完整路线，三组都加载该JSON；`instance_hash`相同才表示相同初始条件与行为规则。不同自车行为会使闭环周车轨迹不同，这是正常交互。

```bash
python scripts/run_carla.py \
  --scenario-config config/carla_scenarios/town01_follow_stop_go.yaml \
  --seed 1 --freeze-instance outputs/carla/instances/follow_seed1.json
```

冻结操作完全离线，不连接CARLA或模型，也不会覆盖同名实例。把上面各场景命令中的 `--scenario-config ...` 改为 `--instance-config outputs/carla/instances/follow_seed1.json` 即可重用这个实例。冻结实例下 `--episodes 2` 表示同实例重复，不会重新采样；CLI seed不覆盖冻结seed。

三组开关如下，其他参数保持一致：

| 组别 | 必需开关 |
|---|---|
| A 基础规划 | `--method A --fast off --reflection off --no-retrieve --no-write-memory` |
| B 基础＋模型 | `--method B --fast server --reflection off --no-retrieve --no-write-memory` |
| C 基础＋模型＋检索 | `--method C --fast server --reflection off --retrieve --no-write-memory --memory-read-dir /data/zjj/frozen_memory` |

B/C使用同一模型服务、提示词、调用频率与下游规划。A也保留既定低频表示、独立参照规划与诊断。正式C在批次开始和结束计算记忆目录内容指纹，变化时标记对比无效；主实验不在线写入。建库/调参集不能含正式测试episode或相邻帧。不同episode帧不是独立统计样本。

## 技术验收与正式门禁

候选场景可以直接开发运行。正式实验前，先固定测试划分：复制YAML，设 `split: test`，再冻结相应实例；此后不要修改实例。用同一个冻结实例至少运行两次开发验证，保持录像开启和world-plan关闭。两次短烟雾只能验证初态与基础设施，不能替代完整场景覆盖。

```bash
python scripts/validate_carla_experiment_runs.py \
  outputs/carla/admission/episode_repeat_1 \
  outputs/carla/admission/episode_repeat_2 \
  --output outputs/carla/admission/follow_seed1_validation.json
```

上例两个路径须替换为真实episode目录。报告把 `technical_smoke_passed` 与 `scenario_coverage_passed` 分开：前者检查冻结哈希、SDK、50Hz连续物理帧、初态/灯态、完整测量、清理以及视频；后者要求至少一条开发运行从实际物理记录重算后经过全部有序路线检查点，并实际触发、完成该实例的所有必需事件。视频逐帧校验六路RGB源帧、时间戳、物理tick和10Hz间隔，视频文件本身也纳入证据哈希。

如果这两轮没有覆盖全部任务，可以用相同冻结实例的独立开发验证运行提供证据，在上述命令中增加 `--coverage-witness 实际开发轮目录`。该证据不要求使用被测MPC，也不要求 `strict_success`；**不能从A/B/C正式结果挑选成功轮用于准入，更不能据正式失败修改或替换测试实例。** 所有传入运行的失败原因都保留；缺少覆盖时正式门禁关闭，不把碰撞或停滞改成环境无效。仍需审阅视频确认物理交互符合任务语义。

正式运行在同一冻结实例命令上增加：
```text
--experiment-phase formal --validation-report outputs/carla/admission/follow_seed1_validation.json
```

门禁会重新读取并验证原始证据文件，不能只把passed改为true。报告及证据目录应一起保留/迁移。场景、桥接、初始化、规则和评价源码按统一LF文本计算指纹，相机与地图几何配置也必须保持一致；改动这些内容需重新验收。MPC权重和模型选择属于对比方法配置，会记录在每轮配置中，不作为场景准入源码限制。正式主表必须test划分、同步、明确A/B/C、关闭反思和写入，禁止临时覆盖时限、换地图或开启世界debug线。未获得服务器证据前，不提供伪造验收报告。

## 结果与论文口径

- `resolved_instance.json`：地图、锁定路线、seed实际采样值、初态、行为、信号与实例哈希。
- `carla_initialization.json`：正式t0演员状态、SDK、地图、相机预热和灯态。
- `physical_evaluation.jsonl`：50Hz实测评价证据；不受MPC八向对象选择限制。
- `evaluation.json`：到达、合规完成、违规/接触、时距、TTC/PET、停滞、舒适性、规则覆盖及基础设施状态。
- `summary.json`、批量JSON：驾驶结果与初始化/运行/清理错误分开；超时不算成功。MPC质量分不是任务成功率。
- `carla_cleanup.json`：自有演员清理与世界设置恢复核验；保留碰撞后清理失败这两个事实。

独立评价的 `strict_success_status` 可区分成功、失败和规则覆盖不足。未知规则不能当作零违规，也不能从对比结果中静默剔除。先报告所有尝试的运行可靠性，再报告有效实例上的驾驶性能。事件未发生需区分合理提前让行、未驶入事件区和真正脚本失效。

## 新增场景

1. 复用某条已锁定路线时，复制YAML并修改name、family、演员、事件窗口和variations；参数使用米、米/秒、物理tick(50tick=1秒)。每个变体只采样显式允许参数，值写入实例。
2. 新地点需要编译新路线：使用安装CARLA0.9.15 SDK的解释器、真实XODR，明确起点、分岔选择和任务出口。例如：

```bash
/home/zjj/miniconda3/envs/carla37/bin/python scripts/compile_carla_route.py \
  --xodr /data/zjj/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01.xodr \
  --map Town01 --start 1,1,40 --length-m 300 \
  --branches 具体道路ID:车道ID,下一分岔道路ID:车道ID \
  --task "指定路口右转后到目标出口" --output config/carla_routes/my_route.json
```

其中起点和branches必须填当前地图真实值；工具会给出未声明分岔的候选并停止，不会自动选一个方向继续。服务器打包地图没有XODR文件时，使用本地同版本资源或导出的真实OpenDRIVE。工具只编译公开拓扑，之后仍需演员布置和服务器验收。
3. YAML的 `experiment.route.file` 指向新JSON，NPC独立路线可放在 `actors[].placement.route.file`。前车路线应越过自车终点，避免NPC在任务终点停住形成伪堵塞。
4. 运行 `python scripts/verify_carla_catalogue.py` 检查冻结协议；本机SDK环境可加 `--maps-dir 地图目录` 做真实拓扑、actor placement和高度检查。不同机器地图指纹不一致时重新核验路线资产，不跳过哈希检查。
