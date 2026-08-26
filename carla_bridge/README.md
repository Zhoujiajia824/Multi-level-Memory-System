# CARLA 闭环集成（carla_bridge）

把多层次记忆系统（`src/vla_memory`，**不改一行**）接入 CARLA 0.9.15：用 CARLA 实时
感知（六视角图像 / GT 障碍物 / 天气 / 导航 / 自车状态）替代离线 nuScenes，决策轨迹
实时回控 CARLA，实现**自定义环境下的闭环驾驶**，并支持驾驶视频录制回看。

> ⚠️ 研究用离线/仿真 Demo，**不可用于实车**。

---

## 1. 设计要点

| 项 | 说明 |
|---|---|
| **记忆系统零改动** | 复用 `OnlineDrivingLoop`（setup/step/close），CARLA 数据经 `KeyframeBuilder` 组装成与 nuScenes 同构的 `kf` dict 喂入 |
| **1Hz 全量认知** | 每 1s sim 做一次完整 DINOv2 + 场景VLM + 三层记忆检索 + 决策VLM（与 nuScenes keyframe 一致） |
| **~5Hz raw 捕获** | 控制阶段以 ~5Hz 捕获原始感知（图像+ego+障碍物，不跑 VLM）push 进短期记忆，队列时刻最新 |
| **20Hz 回控** | Pure Pursuit(横向) + PID(纵向) 跟踪决策轨迹，同步模式 |
| **同步模式** | VLM 思考期间 CARLA 冻结（不 tick），wall-clock 慢但 sim 时间连续正确 |
| **驾驶视频录制** | 仅在 CARLA 推进 tick 取帧（跳过 VLM 冻结期），视频时长 = 仿真实际驾驶时间 |
| **CARLA 全优势** | 自定义地图/天气/交通流/车辆/路由，全 Python API |

---

## 2. 架构总览

```
CARLA server (CarlaUE4.exe, 端口2000, 手动启动, 同步模式)
        │   传感器/GT ↓   carla.VehicleControl ↑
        ▼
carla_bridge.closed_loop  ──每 1s sim──>  keyframe_builder
        │                                    │  组装 kf dict（与 nuScenes 同构）
        │                                    ▼
        │                 src.vla_memory.pipeline.OnlineDrivingLoop.step(kf)  【不改】
        │                                    │  DINOv2 + 场景VLM + 三层记忆 + 决策VLM
        │                                    ▼  decision = {behavior, trajectory(ego-centric)}
        │  ◀──  trajectory  ────────────────
        ▼
trajectory_tracker (Pure Pursuit + PID)  ->  carla.VehicleControl  ->  vehicle.apply_control
        │
        ├── metrics: 碰撞/违规/路线/舒适度 逐 tick 采集
        └── video: 每个 CARLA 推进 tick 取一帧（跳过 VLM 冻结）写视频
```

**单周期时序（每 1s sim）：**
```
[1 捕获]  tick -> 6 相机 mosaic + ego + GT 感知 + nav         (推进 0.05s)
[2 认知]  loop.step(kf) -> 决策轨迹                           (sim 冻结, wall 5-30s)
[3 控制]  20Hz × 20 tick：Pure Pursuit+PID 回控 CARLA          (推进 1.0s)
          其中 ~5Hz raw 捕获 push 进短期记忆；其余 tick drain 相机防堆积
[4 记录]  逐 tick 指标采集 + 视频取帧
```

---

## 3. 项目框架

```
carla_bridge/
├── README.md                       # 本文件
├── __init__.py
├── setup_env.sh                    # 一键建 mulmem_carla(3.9) 环境
├── closed_loop.py                  # 【主驱动】同步 tick + 1Hz 重规划 + raw 捕获 + 视频
├── run_carla_demo.py               # 【入口】CLI -> ClosedLoop
│
├── config/
│   ├── carla.yaml                  # 连接/同步/相机/ego/控制器/视频 配置
│   └── scenarios/                  # 每个场景一个 YAML
│       ├── straight_traffic.yaml   #   直行+交通流（Town04）
│       └── intersection_turn.yaml  #   路口连续转弯（Town05）
│
├── env/                            # CARLA 环境与场景
│   ├── carla_client.py             #   连接 + 同步模式 + tick + load_map + spectator 跟随
│   ├── scenario_manager.py         #   加载 YAML -> 地图/天气/ego/交通/路由
│   ├── traffic_spawner.py          #   Traffic Manager 车辆 + 行人
│   ├── weather_controller.py       #   天气预设（动态枚举，兼容 0.9.15）
│   ├── route_planner.py            #   A* 全局路由(start->end) + 派生 nav + 路线完成度
│   ├── walker_controller.py        #   行人过马路/路边行走
│   └── event_scheduler.py          #   长尾事件调度（按 ego 距离触发）
│
├── sensors/                        # 感知
│   ├── camera_manager.py           #   6 相机布置 + 捕获 mosaic + drain 防堆积
│   └── perception_provider.py      #   world.get_actors() GT -> PerceptionObject
│
├── state/                          # 状态与坐标
│   ├── coords.py                   #   CARLA↔ego-centric 变换（手性开关 YAW_SIGN）
│   ├── ego_state_provider.py       #   vehicle -> EgoState（速度/加速度直读）
│   └── history_buffer.py           #   滚动过去 5s 位姿 -> ego-centric 历史轨迹
│
├── control/                        # 低层控制
│   ├── pure_pursuit.py             #   横向：Pure Pursuit
│   ├── pid.py                      #   纵向：PID 速度跟踪
│   ├── trajectory_tracker.py       #   合成 -> carla.VehicleControl（每 tick）
│   └── scripted_vehicle.py         #   NPC 的 NOA 路由 + ACC 跟车
│
├── memory_adapter/                 # 记忆系统适配
│   ├── keyframe_builder.py         #   CARLA tick 数据 -> kf dict（喂 step()）
│   └── loop_runner.py              #   封装 OnlineDrivingLoop + add_short_term_item
│
├── metrics/                        # 闭环评测
│   ├── collector.py                #   碰撞/闯红灯/逆行/超速/路线/舒适度 采集
│   └── run_reporter.py             #   Markdown + JSON 报告
│
└── video/                          # 驾驶视频录制（独立小功能，默认关）
    └── recorder.py                 #   仅 CARLA 推进 tick 取帧，时长=仿真驾驶时间
```

---

## 4. 功能模块

### 4.1 `env/` 环境与场景
- **`carla_client.py`**：连接 CARLA（127.0.0.1:2000），设同步模式 + `fixed_delta_seconds=0.05`（20Hz）；`tick()` 推进；`load_map()` 切地图后重应用同步设置；`set_spectator_follow()` 把视角摆到 ego 后上方第三人称。
- **`scenario_manager.py`**：读场景 YAML，依次设地图/天气/ego/路由/交通流；`from_yaml` 类方法加载；`destroy()` 清理。
- **`traffic_spawner.py`**：Traffic Manager 生成 N 辆 autopilot 车辆 + M 个 AI 行人（best-effort）。
- **`weather_controller.py`**：**动态枚举** `carla.WeatherParameters` 预设（0.9.15 的 `MidRainyNoon`/`MidRainSunset` 命名不统一，硬编码会踩坑），支持预设名或自定义参数。
- **`route_planner.py`**：**A* 全局路径规划**（CARLA `GlobalRoutePlanner`，起点->终点，模拟地图导航）；从路由 `RoadOption` 派生 `nav_instruction`（straight/left_turn/...）；`progress_fraction()` 路线完成度。`end` 为空时回退沿路前进。
- **`walker_controller.py`**：行人生成 + 行为控制（`spawn_crossing` 过马路 / `spawn_roadside` 路边走），用 `controller.ai.walker`。
- **`event_scheduler.py`**：**长尾事件调度器**，按 ego 离起点距离触发事件（行人/ACC 车/障碍物/信号灯），每 tick 驱动 scripted 车。

### 4.2 `sensors/` 感知
- **`camera_manager.py`**：6 相机（前左/前/前右/后左/后/后右，2×3 布局）挂 ego；`capture()` 排空旧帧取最新 + 复用 `build_surround_mosaic` 拼 2×3；`drain()` 控制阶段丢弃帧防堆积；`warmup()` 首帧预热。
- **`perception_provider.py`**：`world.get_actors()` 取 GT（车/行人/交通灯），转 ego-centric，映射成 `PerceptionObject`（`is_oracle=True`，速度/加速度直读）。

### 4.3 `state/` 状态与坐标
- **`coords.py`**：**全项目坐标命门**。CARLA(x前/y右) ↔ ego-centric(x前/y左) 互转；`YAW_SIGN` 手性开关（实测镜像改 -1）；`trajectory_ego_to_global()` 决策轨迹转全局给控制器。
- **`ego_state_provider.py`**：vehicle 直读 transform/velocity/acceleration/control -> `EgoState`（`source="carla"`，优于 nuScenes 差分）。
- **`history_buffer.py`**：滚动过去 5s 全局位姿，用 `coords` 转 ego-centric 历史轨迹（**不复用 `TrajectoryBuilder`**，其 ego_y 实为"右"）。

### 4.4 `control/` 低层控制
- **`pure_pursuit.py`**：Pure Pursuit 横向，前瞻距离 `clamp(min+k·speed, min, max)`。
- **`pid.py`**：PID 纵向速度跟踪，throttle/brake 互斥，积分抗饱和。
- **`trajectory_tracker.py`**：捕获时刻把 ego-centric 轨迹转全局 waypoint，每 tick Pure Pursuit+PID -> `carla.VehicleControl`。`steer_sign` 修正左右。
- **`scripted_vehicle.py`**：NPC 的 NOA（路由跟随）+ ACC（前方同车道有车则减速），复用 Pure Pursuit+PID。

### 4.5 `memory_adapter/` 记忆适配
- **`keyframe_builder.py`**：组装 `step()` 所需 kf dict（sample_token/scene/image_path/ego/history/nav/perception）。
- **`loop_runner.py`**：薄封装 `OnlineDrivingLoop`；`add_short_term_item()` 供控制阶段 raw 捕获直接 push 短期记忆。

### 4.6 `metrics/` 评测
- **`collector.py`**：碰撞（collision sensor）、闯红灯、逆行、超速、路线完成度、舒适度（jerk/加速度）逐 tick 采集。
- **`run_reporter.py`**：写 Markdown + JSON 报告到 `outputs/carla_runs/`。

### 4.7 `video/` 驾驶视频录制
- **`recorder.py`**：挂一个 chase/front 相机到 ego，每个 CARLA 推进 tick 取一帧写视频（20fps）；VLM 冻结期不 tick 不取帧 -> **视频时长 = 仿真实际驾驶时间**；`flush()` 中断时保存。

### 4.8 主驱动与入口
- **`closed_loop.py`**：`ClosedLoop` 类，编排上述全部模块；含 fail-fast（API key）、SSL_CERT_FILE 自动修复、异常捕获打印 traceback、spectator 跟随。
- **`run_carla_demo.py`**：CLI 入口，加载 `carla.yaml` 作 overrides 深合并，跑 `ClosedLoop`。

---

## 5. 环境搭建

CARLA 0.9.15 绑定需 Python 3.9，与记忆系统的 mulmem(3.12) 不兼容，故新建统一环境：

```bash
bash carla_bridge/setup_env.sh
# 完成后：conda activate mulmem_carla
```

脚本：建 `mulmem_carla`(3.9) -> 装 `requirements.txt` + `faiss-cpu==1.9.0` + `carla==0.9.15` -> 验证 `import carla, faiss, torch, transformers` 与 `OnlineDrivingLoop` 均可用。

> mulmem(3.12) 保持不动，离线 nuScenes 流程照常在 mulmem 跑。

---

## 6. 运行操作指令

### 6.1 启动 CARLA 服务器（手动，另一终端）
```
D:\software\carla\WindowsNoEditor\CarlaUE4.exe
```
显存不够时用低画质：
```
cd D:\software\carla\WindowsNoEditor
.\CarlaUE4.exe -quality-level=Low -windowed -ResX=1280 -ResY=720
```
等窗口加载完、可交互（~15s）再跑下文命令。

### 6.2 设 VLM API Key（⚠️ Git Bash 必须用 `export`）
```bash
conda activate mulmem_carla
export DASHSCOPE_API_KEY=你的key      # Git Bash(MINGW64) 用 export；CMD 用 set；PowerShell 用 $env:
```
> `set` 在 bash 里不设环境变量，会触发 `EnvironmentError`。脚本有 fail-fast：缺 key 立即报错。

### 6.3 跑闭环
```bash
python -m carla_bridge.run_carla_demo --scenario straight_traffic --mode memory_on
```
- `--scenario`：场景名（`straight_traffic` / `intersection_turn`）或 YAML 绝对路径。
- `--mode`：`memory_on`（三层记忆）/ `memory_off`（对照基线）。

**memory_on vs memory_off 对比**（同场景跑两遍）：
```bash
python -m carla_bridge.run_carla_demo --scenario straight_traffic --mode memory_on
python -m carla_bridge.run_carla_demo --scenario straight_traffic --mode memory_off
```
报告在 `outputs/carla_runs/`，对比碰撞率/路线完成度等量化记忆增益。

### 6.4 驾驶视频录制
在 [config/carla.yaml](config/carla.yaml) 的 `carla.video` 块设 `enabled: true`：
```yaml
  video:
    enabled: true
    fps: 20
    width: 1280
    height: 720
    view: "chase"      # chase（后上方第三人称）| front（前视角）
```
跑完（或 Ctrl+C）视频在 `outputs/carla_videos/drive_<场景>_<模式>.mp4`，时长 = 仿真实际驾驶时间（跳过 VLM 冻结期）。

### 6.5 首测建议
1Hz 下 60s sim ≈ 60 次 VLM（每次 5-30s wall）。首测先把场景 YAML 的 `duration_s` 改成 `10`（约 3-5 分钟 wall），确认闭环通了再加长。

---

## 7. 配置说明

### 7.1 `config/carla.yaml`（关键项）
| 键 | 默认 | 说明 |
|---|---|---|
| `carla.host`/`port` | 127.0.0.1/2000 | CARLA 服务器 |
| `carla.synchronous`/`fixed_delta_seconds` | true/0.05 | 同步模式 + 20Hz |
| `carla.replan_interval_s` | 1.0 | 1Hz 全量捕获（与 nuScenes 一致） |
| `carla.max_duration_s` | 60.0 | 单场景最长 sim 时间 |
| `carla.cameras` | 6 相机 | 640×360，2×3 布局（P1 可调 transform） |
| `carla.controller` | — | wheelbase/max_steer/lookahead/PID 增益/`steer_sign` |
| `carla.video` | enabled:false | 视频录制开关与参数 |
| `memory_db_dir` | outputs/memory_db_carla | CARLA 独立中期记忆库（不污染 nuScenes） |
| `perception.mode` | surround_mosaic | 固定六视角环视 |

> carla.yaml 由入口 `load_config(overrides=...)` 深合并进主配置，**不改 `src/` 与现有 `config/`**。

### 7.2 `config/scenarios/*.yaml`
```yaml
scenario_name: "straight_traffic"
map: "Town04"              # CARLA 地图
duration_s: 60             # 单场景 sim 时长（秒）
seed: 42
weather: { preset: "ClearNoon" }
ego: { blueprint: "vehicle.tesla.model3", spawn: {x: null, y: null, yaw: null} }
traffic: { num_vehicles: 20, num_walkers: 5 }
route:
  start: {spawn_point: 0}        # 起点（= ego spawn）；也可 {x,y,z}；null=ego spawn
  end:   {spawn_point: 80}       # 终点：A* 全局规划到这里；null=回退沿路前进
  sampling_resolution: 2.0       # A* 采样间隔（米）
  length_m: 250                  # end 为空时用：沿路长度
  turns: []                      # end 为空时用：路口转向 straight/left/right
```

---

## 7.3 长尾事件系统（EventScheduler）

在场景 YAML 的 `events` 列表声明长尾事件，按 ego 离起点的距离（`at_distance` 米）触发：

```yaml
events:
  - {type: pedestrian_roadside, at_distance: 15, side: right, walk_ahead_m: 30}
  - {type: scripted_vehicle, at_distance: 40, blueprint: "vehicle.audi.etron", lane: left, speed: 8, acc: true, route_len_m: 120}
  - {type: pedestrian_crossing, at_distance: 70, side: right, cross_dist: 8}
  - {type: obstacle, at_distance: 100, prop: "static.prop.barrier", ahead_m: 20}
  - {type: traffic_light_force, at_distance: 130, state: red, duration: 15}
```

| 事件类型 | 说明 | 关键参数 |
|---|---|---|
| `pedestrian_crossing` | ego 前方路边生成行人，目标对侧（过马路/鬼探头） | `ahead_m`, `side`, `cross_dist` |
| `pedestrian_roadside` | 路边行人沿路走 | `ahead_m`, `side`, `walk_ahead_m` |
| `scripted_vehicle` | **NOA+ACC 交通车**（路由跟随 + 跟车减速） | `blueprint`, `lane`, `ahead_m`, `speed`, `acc`, `route_len_m` |
| `obstacle` | 静态障碍物突现 | `prop`, `ahead_m` |
| `traffic_light_force` | 强制前方信号灯状态 | `state`(red/green/yellow), `duration`, `ahead_m` |

- `at_distance`：ego 离起点多少米时触发。
- `scripted_vehicle`：spawn 在 ego 前方 `ahead_m`、`lane` 车道，沿车道走 `route_len_m`（NOA），速度 `speed`，`acc:true` 开启跟车减速（前方同车道有车则降到前车速度）。
- 障碍物蓝图可用 `bp_lib.filter("static.prop.*")` 列出全部（名字不对会 warning 跳过，不崩）。
- 示例：[longtail_city.yaml](config/scenarios/longtail_city.yaml)。

---

## 8. 坐标约定（最大坑）

CARLA 全局：x 前、y 右、z 上。项目 ego-centric：x 前、**y 左**。所有 CARLA 数据进出记忆系统统一走 [state/coords.py](state/coords.py)（含手性开关 `YAW_SIGN`）。

- 若车倒着开 / 转向镜像：把 `coords.py` 的 `YAW_SIGN` 或 `carla.yaml` 的 `controller.steer_sign` 改成 `-1`。
- 历史轨迹**不复用** `TrajectoryBuilder`（其 ego_y 实为"右"），由 `coords` 直接生成，保证与决策轨迹（y=左）一致。

---

## 9. 评测指标

原 ADE/FDE 不适用闭环（未来由自车决策产生）。`metrics/collector.py` 采集：

| 指标 | 说明 |
|---|---|
| 碰撞次数 | `sensor.other.collision` 触发 |
| 路线完成度 | `RoutePlanner.progress_fraction` |
| 闯红灯 / 逆行 / 超速 | 近红灯仍前进 / 与车道反向 / 超限速 |
| 最大速度 / 加速度 / jerk | 舒适度 |

报告：`outputs/carla_runs/carla_run_<场景>_<模式>.md` / `.json`。

---

## 10. 故障排查

| 现象 | 原因 / 解决 |
|---|---|
| `EnvironmentError: VLM API Key 未设置` | Git Bash 用 `export DASHSCOPE_API_KEY=...`（不是 `set`）；脚本 fail-fast 会在 setup 前报 |
| `FileNotFoundError: SSL_CERT_FILE` | conda 激活脚本指向不存在的 cacert.pem；`run_carla_demo.py` 已自动清除坏路径改用 certifi，无需手动处理 |
| `time-out waiting for the simulator` | CARLA 服务器没启动/卡死。`taskkill //F //IM CarlaUE4.exe` 后重启，等加载完 |
| 内存卡死 / CARLA 崩 | 6 相机已降到 640×360 + 控制阶段 `drain()` 防堆积；若仍崩，降 `carla.video` 分辨率或关视频 |
| VLM 调用久无日志 | 正常（5-30s/次），期间 sim 冻结。**别 Ctrl+C**，等 `[SCENE_UNDERSTANDING] raw_response` 出现 |
| 车倒着开 / 转向反 | 改 `state/coords.py` 的 `YAW_SIGN` 或 `carla.yaml` 的 `controller.steer_sign` 为 `-1` |
| CARLA 窗口看不到车 | `set_spectator_follow` 每 tick 跟随；若没跟，确认 CARLA 已加载完且 ego 已生成 |

---

## 11. 开发进度

- [x] P0 环境/骨架：`mulmem_carla`(3.9) 已建并验证（carla+faiss+torch+记忆系统导入 OK）；coords/carla_client/config
- [x] P1 感知接入：6 相机 + ego 状态 + GT 感知 + 天气
- [x] P2 记忆管线对接：keyframe_builder + loop_runner（复用 `OnlineDrivingLoop`，不改 src/）
- [x] P3 闭环控制：轨迹转全局 + Pure Pursuit + PID + `closed_loop` 20Hz 跟踪
- [x] P4 场景系统：YAML + Python 场景类 + 交通流 + 天气 + 路由 + nav 派生
- [x] P5 评测：碰撞/违规/路线/舒适度 + 报告（已接入 `closed_loop`）
- [x] 1Hz 捕获 + 控制阶段 ~5Hz raw 捕获（短期记忆时刻最新）+ 相机防堆积
- [x] spectator 第三人称跟随
- [x] 驾驶视频录制（跳过 VLM 冻结期，时长=仿真驾驶时间，中断保存）
- [x] fail-fast API key + SSL_CERT_FILE 自动修复 + 异常 traceback 打印
- [x] 导入冒烟测试通过（carla_bridge + src 全部导入 OK，坐标数学往返一致）
- [x] 长尾事件系统：EventScheduler + scripted 车(NOA+ACC) + 行人(过马路/路边) + 障碍物 + 信号灯控制
- [ ] P6 运行验证：启动 CARLA + 设 key 后跑 memory_on/off 闭环对比，调控制器/坐标手性
