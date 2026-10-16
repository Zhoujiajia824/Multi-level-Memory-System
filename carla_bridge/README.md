# CARLA 闭环集成（carla_bridge）

把多层次记忆系统（`src/vla_memory`）接入 CARLA 0.9.15：用 CARLA 实时
感知（六视角图像 / GT 障碍物 / 天气 / 导航 / 自车状态）替代离线 nuScenes。
默认链路为语义决策 → PDM-Closed 候选规划 → Pure Pursuit + 速度曲线 PID 回控。
支持通过 `--semantic-influence on|off` 做语义影响对照；安装依赖及使用方式见[第 13 节](#13-语义决策与-pdm-closed)。

首次配置按顺序阅读：[第 5 节：新环境或已有环境安装](#5-环境搭建) →
[第 12.0 节：生成与运行新场景](#120-从服务器开始生成场景再运行场景) →
[第 12.6 节：记忆库模式完整命令](#126-新场景与记忆库模式完整命令)。

> ⚠️ 研究用离线/仿真 Demo，**不可用于实车**。


## 服务器拉取与首轮验收

以下命令在服务器的**仓库根目录**执行。此分支包含之前漏入库的 `carla_bridge/env/`，
新机器拉取后无需再手工补目录。若服务器已有同名未跟踪文件或本地配置修改，先备份或保存，
不要使用强制覆盖命令。Git 检查到冲突时会停止更新。

```bash
git fetch origin
git switch carla-integration
git pull --ff-only origin carla-integration
git status --short --branch
git log -1 --oneline
conda activate mulmem_carla
# 已有环境补齐测试依赖；无需重建环境
python -m pip install "pytest>=7.4,<9"
```

**先检查环境，不连接仿真器或调用 API：**

PDM 是新增可选依赖，但已作为 CARLA 默认后端。首次更新先按第 13 节安装并预检；
缺依赖会在加载 CARLA 场景之前报错，不会静默回退到旧模型轨迹。

```bash
python -c "import sys, carla, faiss, torch, pydantic; from carla_bridge.closed_loop import ClosedLoop; print('Python:', sys.version); print('CUDA:', torch.cuda.is_available()); print('IMPORT_OK')"
python -m carla_bridge.run_carla_demo --help
python -m pytest -q -p no:cacheprovider
```

测试使用本地 CARLA Python 绑定和 FAISS，不启动 CARLA、不下载模型、不调用 VLM。
已有 Pydantic 兼容性警告不等同测试失败。服务器需单独执行以上检查。
使用已有环境时不必重新运行创建环境脚本；新环境见下文第 5 节。

**再检查运行配置：**

- 按服务器原有方式启动 CARLA 0.9.15，确认场景使用的 Town04 / Town01 地图可加载。
  `carla_bridge/config/carla.yaml` 默认连接 `127.0.0.1:2000`，异机部署需修改 host。
- 保持 `synchronous: true`。同一个 CARLA world 同时只运行一个闭环驱动客户端。
- 检查 `config/api_models.yaml` 的服务地址、模型名、`api_key_env`，在运行终端设置对应密钥。
  默认 `DASHSCOPE_API_KEY`：Bash 用 `export DASHSCOPE_API_KEY='你的key'`，
  PowerShell 用 `$env:DASHSCOPE_API_KEY='你的key'`。
- DINOv2 默认只读本地模型文件；`CUDA: False` 不代表导入失败，但推理性能需另行评估。
- 当前视频默认开启；显存不足时在 `carla.video` 中关闭视频或降低分辨率。

**检查手工上传的 DINOv2 缓存：** 默认路径相对仓库根目录解析，结构为：

```text
.cache/huggingface/
└── models--facebook--dinov2-base/
    ├── blobs/
    ├── refs/
    └── snapshots/<revision>/
        ├── config.json
        ├── preprocessor_config.json
        └── model.safetensors 或 pytorch_model.bin
```

上传时保留完整文件及符号链接目标。`--dinov2-cache-dir` 指向 `huggingface` 缓存根目录，
`--dinov2-model-dir` 指向包含配置、图像处理器配置和权重的模型目录或具体 snapshot。
离线预检会实际加载模型并执行一次前向推理：

```bash
python scripts/00_prepare_models.py --offline
# 缓存在其他目录时
python scripts/00_prepare_models.py --offline --cache-dir /path/to/huggingface
```

CARLA 配置默认 `feature_extractor.local_files_only: true`，缺文件立即报错，不等待 Hugging Face。
入口自动启用原生故障堆栈和及时日志输出，**无需 `HF_HUB_OFFLINE=1`、`-X faulthandler`、`-u` 前缀**。
VLM API 调用仍需网络。覆盖模型位置或显式允许下载的方式为：

```bash
python carla_bridge/run_carla_demo.py --scenario straight_traffic --dinov2-cache-dir /path/to/huggingface
python carla_bridge/run_carla_demo.py --scenario straight_traffic --dinov2-model-dir /path/to/dinov2-model
# 仅在需要联网下载时使用；已指定 model-dir 时始终读取该目录
python carla_bridge/run_carla_demo.py --scenario straight_traffic --allow-model-download
```

**验收运行以完整规划路线完成率 100% 为准。** 先运行直行完整路线，再验证转弯和记忆模式：

```bash
python carla_bridge/run_carla_demo.py --scenario straight_traffic --mode memory_on
# 直行：指定库积累经验；重复此命令继续积累
python -m carla_bridge.run_carla_demo --scenario straight_traffic --memory-bank outputs/memory_banks/server_straight
# 同库只读：验证能检索且不修改库
python -m carla_bridge.run_carla_demo --scenario straight_traffic --memory-bank outputs/memory_banks/server_straight --memory-access read_only
# 转弯：独立场景库，重点观察左右转与轨迹末端制动
python -m carla_bridge.run_carla_demo --scenario intersection_turn --memory-bank outputs/memory_banks/server_intersection
```

每轮记录新增的 `<run_id>`，按以下项目检查：

| 检查项 | 文件或观察点 |
|---|---|
| 路线验收 | `manifest.json` 的 `status=route_completed`、`outcome.success=true`、`outcome.route_completed=true`、`outcome.route_completion=1.0`；报告 `metrics.route_completion=1.0`，清理完成且退出码为 0 |
| 图像同步 | 决策 JSONL 中的 `carla_frame_id` / `sim_time_s`；报告中的 `capture_failures` |
| 记忆是否参与 | 决策 JSONL 的 `retrieved_memory_ids`、`memory_retrieval_enabled`、`memory_write_enabled` |
| 驾驶效果 | 完成路线之外，单独检查碰撞次数、违规和视频中的左右转、回正及制动行为 |
| 决策轨迹 | 决策 JSONL 的 `decision_validation`；异常轨迹被拦截后制动，不进入中期或事件记忆 |
| 异常定位 | 运行目录的 `logs/`，连同服务器 CARLA 日志一起保留 |
| 多轮隔离 | 两次运行目录不同；明确指定的共享库路径保持一致 |

路线必须按顺序通过，且当前车辆距终点不超过默认 2 米；仅驶近某个后续路段不能跳过前面的路线。
`carla.route_tracking_tolerance_m` 默认 5 米，允许中途正常变道；
`carla.route_completion_tolerance_m` 默认 2 米，控制终点到达判定。
场景 `duration_s` 与全局 `carla.max_duration_s` 默认均为 300 秒仿真时间，取较小值作为保护上限，
长路线可同时调大；达到上限是 `timeout`，不会当作路线完成。10 秒冒烟仅可辅助排查启动问题，不算验收。
本地回归测试不能验证引擎原生崩溃是否消除，仍需服务器完整路线运行与正常清理验证。

记忆准入有价值门控，普通直行场景可能没有记录入库；不能仅凭库为空判定写入失败。
报告中的闯红灯等指标采用启发式判定，回归评估时应结合视频。
如出现问题，保留对应运行目录、场景配置及 CARLA 日志；**无需分享 API Key**。

### 运行状态与定位

| 状态 / 报错 | 处理方式 |
|---|---|
| `route_completed` | 已按序完成规划路线；继续核对报告、碰撞指标和退出码 |
| `timeout` | 到达保护时限仍未完成路线，退出码非 0；先检查实际进度，再决定是否增加场景与全局时限 |
| `sensor_failure` | 连续采集失败；检查相机日志、渲染负载与是否存在其他 tick 客户端，再评估 `capture_timeout_s`，不要通过额外 tick 绕过同步 |
| `stuck` | 有效决策要求继续行驶却持续低速且无进展；目标为停车的决策不累计卡死计时，仍受总体超时保护 |
| `invalid_trajectory` | 连续 3 次决策未通过物理校验；检查 JSONL 的 `decision_validation`，单次拒绝即制动且不入库 |
| `cleanup_failed` / `report_failed` | 资源清理或报告保存失败；即使路线已完成也非零退出，检查 manifest 的 `outcome.cleanup_errors` |
| `memory_save_failed` | 检查库的写权限、磁盘空间和保存日志；本轮结果不能视为已成功积累 |
| `failed` / `interrupted` | 运行异常或主动中断；查看 manifest 的 `error_type` 与日志，清理会自动执行 |
| 库正在被其他运行使用 | 等当前使用者退出；不要删除活动锁文件强行并发 |
| ID 重复、维度不符、资料缺失 | 保留原库，先使用新库验证；不要覆盖旧库来消除错误 |

只有 `route_completed` 且完成清理和报告保存才以退出码 0 结束，其他结束状态均为非 0。
CARLA 默认禁用解析失败的规则轨迹兜底，保留原始模型失败供定位。

## 运行隔离与记忆库选择（当前入口行为）

在仓库根目录、`mulmem_carla` 环境运行。每次生成独立的
`outputs/carla_runs/<run_id>/`，包含 manifest、决策、报告、视频和日志。
manifest 记录实际配置、场景、代码提交号、工作区修改状态、记忆库来源和结束状态。
`run_id` 是短 UUID，帧编号包含它，重复运行同一场景不会覆盖产物。

```powershell
# 独立实验：本轮从空库开始，也在本轮持续学习
python -m carla_bridge.run_carla_demo --scenario straight_traffic

# 连续积累：多次运行都指定同一个库，加载历史并继续更新
python -m carla_bridge.run_carla_demo --scenario straight_traffic --memory-bank outputs/memory_banks/general

# 指定场景库：库的选择由参数明确指定，与场景文件名不自动绑定
python -m carla_bridge.run_carla_demo --scenario intersection_turn --memory-bank outputs/memory_banks/intersection

# 只检索已有库：不会新增、更新、淘汰或写回检索统计
python -m carla_bridge.run_carla_demo --scenario intersection_turn --memory-bank outputs/memory_banks/intersection --memory-access read_only

# 从历史库复制一个独立实验库（目标必须不存在；省略目标则使用本轮库）
python -m carla_bridge.run_carla_demo --scenario intersection_turn --memory-seed outputs/memory_banks/intersection --memory-bank outputs/memory_banks/experiment_a

# 无记忆对照：不读取、不写入库，也不使用短期记忆
python -m carla_bridge.run_carla_demo --scenario straight_traffic --mode memory_off
```

- `--no-short-term`：关闭本轮短期记忆；默认只读模式仍维护本轮短期上下文。
- `--no-memory-retrieval`：关闭中长期检索，但保留按写入策略积累记忆的能力；
  续写仍会加载旧库以保留历史记录。固定长期规则仍由原有 `long_term` 配置管理。
- 相对库路径以仓库根目录解析。未指定库时使用本轮 `memory_db/`，不会隐式使用旧的
  `outputs/memory_db_carla`；需要续用旧库时明确传入该路径。
- 可写库将本轮图像与特征保存到库内 `assets/`；只读/对照模式保存到运行目录。
  清理日志时保留记忆库及其 assets。旧版库已有的外部资料引用需继续保留；
  `--memory-seed` 会将有效的外部资料一起复制入新库，缺失资料则明确报错。
- 同一库的访问由进程锁串行保护（只读运行也持锁，避免读到保存一半的索引）。
  退出或进程结束后锁自动释放；库旁的 `.lock` 文件保留，不要手动删除正在使用的锁文件。
  该约定覆盖新版 CARLA 入口，不覆盖旧脚本或手工直接修改数据库。
- `read_only` 必须指定已有库。损坏、重复 ID 或模型维度不匹配的库会报错，避免静默覆盖。
- 图像按照 `tick()` 帧号同步，六路共享 `capture_timeout_s` 截止时间；缺帧则制动，
  连续 `max_capture_failures` 次失败结束运行。记录同时包含绝对记忆时间戳与仿真秒数。
- 控制器修正 Pure Pursuit 曲率，并按剩余轨迹选择前瞻点；无有效前向目标时制动。
  初始化失败和中断均执行清理，结束状态写入 manifest；无需额外 tick 等待图像。

验证：单元与模拟依赖回归测试无需启动 CARLA 或调用模型 API。
真实 CARLA 的相机超时阈值、弯道横向偏差和转向参数仍应通过闭环场景验证。

---

## 1. 设计要点

| 项 | 说明 |
|---|---|
| **复用记忆管线** | 复用 `OnlineDrivingLoop`（setup/step/close），CARLA 数据经 `KeyframeBuilder` 组装成与 nuScenes 同构的 `kf` dict 喂入 |
| **3s 全量认知** | 每 3s sim 做一次完整 DINOv2 + 场景VLM + 三层记忆检索 + 决策VLM（与 nuScenes keyframe 一致） |
| **每周期 5 次 raw 捕获** | 控制阶段每周期 5 次（≈1.67Hz）捕获原始感知（图像+ego+障碍物，不跑 VLM）push 进短期记忆，队列时刻最新 |
| **5Hz PDM 规划** | 基于最新 world 状态和最近语义，生成、仿真、评分轨迹候选 |
| **10Hz 回控** | Pure Pursuit(横向) + PID(纵向) 跟踪 PDM 轨迹及时间速度曲线，同步模式 |
| **同步模式** | VLM 思考期间 CARLA 冻结（不 tick），wall-clock 慢但 sim 时间连续正确 |
| **驾驶视频录制** | 仅在 CARLA 推进 tick 取帧（跳过 VLM 冻结期），按仿真帧率编码，丢帧时视频可能短于仿真时长 |
| **CARLA 全优势** | 自定义地图/天气/交通流/车辆/路由，全 Python API |

---

## 2. 架构总览

```
CARLA server (CarlaUE4.exe, 端口2000, 手动启动, 同步模式)
        │   传感器/GT ↓   carla.VehicleControl ↑
        ▼
carla_bridge.closed_loop  ──每 3s sim──>  keyframe_builder
        │                                    │  组装 kf dict（与 nuScenes 同构）
        │                                    ▼
        │                 src.vla_memory.pipeline.OnlineDrivingLoop.step(kf)
        │                                    │  DINOv2 + 场景VLM + 三层记忆 + 决策VLM
        │                                    ▼  decision = {lateral_decision, longitudinal_decision, ...}
        │       CARLA adapter → PDM generator → simulator → scorer + optional semantic preference
        │  ◀──  validated trajectory + time/speed profile  ────────
        ▼
trajectory_tracker (Pure Pursuit + PID)  ->  carla.VehicleControl  ->  vehicle.apply_control
        │
        ├── metrics: 碰撞/违规/路线/舒适度 逐 tick 采集
        └── video: 每个 CARLA 推进 tick 取一帧（跳过 VLM 冻结）写视频
```

**单周期时序（每 3s sim）：**
```
[1 捕获]  tick -> 6 相机 mosaic + ego + GT 感知 + nav         (推进 0.1s)
[2 认知]  loop.step(kf) -> 横纵向语义 -> PDM轨迹              (sim 冻结, wall 5-30s)
[3 控制]  10Hz × 29 tick：Pure Pursuit+PID 回控 CARLA          (推进 2.9s)
          每 0.2s PDM 根据最新 world 状态重规划；和下一捕获 tick 合计 3.0s
          其中每周期 5 次 raw 捕获 push 进短期记忆；其余 tick drain 相机防堆积
[4 记录]  逐 tick 指标采集 + 视频取帧
```

---

## 3. 项目框架

```
carla_bridge/
├── README.md                       # 本文件
├── __init__.py
├── setup_env.sh                    # 一键建 mulmem_carla(3.9) 环境
├── closed_loop.py                  # 【主驱动】同步 tick + 3s 重规划 + raw 捕获 + 视频
├── run_carla_demo.py               # 【入口】CLI -> RunSession -> ClosedLoop
├── run_session.py                  # 产物隔离、记忆库选择、进程锁与 manifest
│
├── config/
│   ├── carla.yaml                  # 连接/同步/相机/ego/控制器/视频 配置
│   └── scenarios/                  # 每个场景一个 YAML
│       ├── straight_traffic.yaml   #   直行+交通流（Town04）
│       └── intersection_turn.yaml  #   路口连续转弯（Town01）
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
│   ├── trajectory_validation.py    #   决策轨迹物理合理性检查
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
- **`carla_client.py`**：连接 CARLA（127.0.0.1:2000），设同步模式 + `fixed_delta_seconds=0.1`（10Hz）；`tick()` 推进；`load_map()` 切地图后重应用同步设置；`set_spectator_follow()` 把视角摆到 ego 后上方第三人称。
- **`scenario_manager.py`**：读场景 YAML，依次设地图/天气/ego/路由/交通流；`from_yaml` 类方法加载；`destroy()` 清理。
- **`traffic_spawner.py`**：Traffic Manager 生成 autopilot 车辆；AI 行人的出生点和目标来自人行导航网格，初始化同步一帧后启动。清理时先注销所有交通车辆，再销毁 actor、恢复 TM 模式。
- **`weather_controller.py`**：**动态枚举** `carla.WeatherParameters` 预设（0.9.15 的 `MidRainyNoon`/`MidRainSunset` 命名不统一，硬编码会踩坑），支持预设名或自定义参数。
- **`route_planner.py`**：**自实现 A\* 全局路径规划**（不依赖 agents.GlobalRoutePlanner，起点->终点，模拟地图导航）；由相邻 waypoint 的 yaw 差派生 `nav_instruction`（straight/left_turn/...）；`progress_fraction()` 路线完成度。`end` 为空时回退沿路前进。
- **`walker_controller.py`**：`spawn_crossing` / `spawn_roadside` 通过 `WalkerControl` 沿指定线段行走，到达后停止；定点事件不调用 AI 寻路，不自动绕障。
- **`event_scheduler.py`**：**长尾事件调度器**，按 ego 离起点距离触发事件（行人/ACC 车/障碍物/信号灯），每 tick 驱动车和行人，不额外推进世界。

### 4.2 `sensors/` 感知
- **`camera_manager.py`**：6 相机（前左/前/前右/后左/后/后右，2×3 布局）挂 ego；`capture()` 丢弃旧帧并等待指定帧号的六路图像，再拼接；`drain()` 控制阶段丢弃帧防堆积；`warmup()` 首帧预热。
- **`perception_provider.py`**：`world.get_actors()` 取 GT（车/行人/交通灯），转 ego-centric，映射成 `PerceptionObject`（`is_oracle=True`，速度/加速度直读）。

### 4.3 `state/` 状态与坐标
- **`coords.py`**：**全项目坐标命门**。CARLA(x前/y右) ↔ ego-centric(x前/y左) 互转；`YAW_SIGN` 手性开关（实测镜像改 -1）；`trajectory_ego_to_global()` 决策轨迹转全局给控制器。
- **`ego_state_provider.py`**：vehicle 直读 transform/velocity/acceleration/control -> `EgoState`（`source="carla"`，优于 nuScenes 差分）。
- **`history_buffer.py`**：滚动过去 5s 全局位姿，用 `coords` 转 ego-centric 历史轨迹（**不复用 `TrajectoryBuilder`**，其 ego_y 实为"右"）。

### 4.4 `control/` 低层控制
- **`pure_pursuit.py`**：Pure Pursuit 横向，前瞻距离 `clamp(min+k·speed, min, max)`。
- **`pid.py`**：PID 纵向速度跟踪，throttle/brake 互斥，积分抗饱和。
- **`trajectory_tracker.py`**：捕获时刻把 ego-centric 轨迹转全局 waypoint，每 tick Pure Pursuit+PID -> `carla.VehicleControl`。`steer_sign` 修正左右。
- **`trajectory_validation.py`**：检查轨迹时间、有限数值及位移/速度合理性；拒绝结果触发制动，写入 `decision_validation` 审计，并禁止该决策写入中期或事件记忆。
- **`scripted_vehicle.py`**：NPC 的 NOA（路由跟随）+ ACC（前方同车道有车则减速），复用 Pure Pursuit+PID。

### 4.5 `memory_adapter/` 记忆适配
- **`keyframe_builder.py`**：组装 `step()` 所需 kf dict（sample_token/scene/image_path/ego/history/nav/perception）。
- **`loop_runner.py`**：薄封装 `OnlineDrivingLoop`；`add_short_term_item()` 供控制阶段 raw 捕获直接 push 短期记忆。

### 4.6 `metrics/` 评测
- **`collector.py`**：碰撞（collision sensor）、闯红灯、逆行、超速、路线完成度、舒适度（jerk/加速度）逐 tick 采集。
- **`run_reporter.py`**：写 Markdown + JSON 报告到 `outputs/carla_runs/<run_id>/`。

### 4.7 `video/` 驾驶视频录制
- **`recorder.py`**：挂一个 chase/front 相机到 ego，每个 CARLA 推进 tick 取一帧写视频（10fps）；VLM 冻结期不 tick 不取帧 -> **按仿真帧率编码，丢帧时视频可能短于仿真时长**；`flush()` 中断时保存。

### 4.8 主驱动与入口
- **`closed_loop.py`**：`ClosedLoop` 类，编排上述全部模块；含 fail-fast（API key）、SSL_CERT_FILE 自动修复、异常捕获打印 traceback、spectator 跟随。
- **`run_carla_demo.py`**：CLI 入口，加载 `carla.yaml` 作 overrides 深合并，跑 `ClosedLoop`。

---

## 5. 环境搭建

本节面向 Linux x86_64 服务器，使用本项目选定的 **Python 3.9 + CARLA 0.9.15** 组合。
环境名不是程序硬编码要求；下面统一使用 `mulmem_carla`，不要求额外创建 `mulmem_pdm`。
离线 nuScenes 的 Python 3.12 环境和根目录通用安装步骤不用于此 CARLA/PDM 路径。

### 5.1 取得代码，选择一种环境方式

已经拉取仓库的用户只需进入仓库。全新机器先安装 Git、Conda，并准备 CARLA 0.9.15 服务端
及 Town03/04/05/10HD 地图；`pip install carla` 只安装 Python 客户端，不安装仿真引擎和地图。
首次克隆代码：

```bash
git clone --branch carla-integration \
  https://github.com/Zhoujiajia824/Multi-level-Memory-System.git
cd Multi-level-Memory-System
```

**方式 A：直接复用已有环境（你当前的使用方式）。** 不执行创建环境或重装 PyTorch：

```bash
conda activate mulmem_carla
python --version
python -m pip freeze > /tmp/mulmem_carla_before_pdm.txt
```

接着进入 **5.3** 安装 PDM；已有有效 API Key、DINOv2 缓存和 CARLA 服务配置可继续使用。

**方式 B：从空环境开始。** 只有环境尚不存在时才执行：

```bash
conda create -n mulmem_carla python=3.9 pip -y
conda activate mulmem_carla
python -m pip install --upgrade pip setuptools wheel
```

若已有同名环境但想独立安装，将创建和激活命令的名称都换成 `mulmem_pdm` 即可。
后续步骤只依赖当前激活的环境，不依赖名称。

### 5.2 仅新环境需要：安装驾驶与记忆系统基础依赖

先安装配套 PyTorch/torchvision。以下使用 CUDA 12.6 构建，要求服务器驱动支持该运行时；
先用 `nvidia-smi` 核对驱动环境。版本组合及其他构建见
[PyTorch 官方历史版本命令](https://pytorch.org/get-started/previous-versions/#v280)。
已有能运行 DINOv2 的 PyTorch 环境可以跳过这一步。

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install \
  "numpy==1.26.4" "opencv-python==4.10.0.84" \
  "faiss-cpu==1.9.0" "carla==0.9.15" \
  "pyyaml>=6,<7" "pydantic>=2,<3" "pillow>=10,<12" \
  "tqdm>=4.66,<5" "transformers==4.57.6" \
  "openai>=1.6,<3" "requests>=2.31,<3" "pytest>=7.4,<9"
```

这里按 CARLA 运行链路安装基础依赖，PDM 的数值/地图依赖在下一步补齐。
不需要 nuScenes 数据集，也不需要安装 `nuscenes-devkit`。
不要再叠加根目录 `requirements.txt` 或 `pip install -e .`：它们面向另一套离线流程，
会引入不必要的依赖或不同的 Python 约束；本入口从仓库根目录直接运行。
固定 NumPy/OpenCV 是为了与 PDM 的 NumPy `<2` 约束保持一致，避免未限定的新版 OpenCV 要求 NumPy 2。

### 5.3 新旧环境都需要：安装固定版本 PDM

以下会安装或调整当前激活环境中的依赖。已有环境保留 PyTorch，显式对齐 NumPy/OpenCV：

```bash
python -m pip install \
  "numpy==1.26.4" "opencv-python==4.10.0.84" "pytest>=7.4,<9" \
  -r carla_bridge/planning/requirements-pdm.txt

python -m pip install --no-deps \
  'git+https://github.com/motional/nuplan-devkit.git@e9241677997dd86bfc0bcd44817ab04fe631405b'

mkdir -p .deps
# 已有此目录时跳过 clone，继续核对固定版本和安装
git clone https://github.com/autonomousvision/tuplan_garage.git .deps/tuplan_garage
git -C .deps/tuplan_garage checkout b51d5d04fac1bd4389653b9ab2ff73ea88f435a3
python -m pip install --no-deps -e .deps/tuplan_garage
```

每一步成功后再继续。tuplan 使用 editable 安装以保留其规划子模块；不要安装上游历史完整
`requirements.txt`。PDM 不需要训练权重或 nuPlan 数据集。
服务器不能访问 GitHub/PyPI 时，固定源码和对应 Linux/Python 版本 wheel 的离线转移方法见
[PDM 依赖说明](planning/README.md#固定源码与安装)。Hugging Face 离线配置只处理 DINOv2，
不能替代这些软件依赖的首次安装。

### 5.4 环境预检、模型缓存和 API

先检查软件依赖，再进行离线模型加载：

```bash
python -m pip check
python -c "import carla, cv2, faiss, torch, transformers; from carla_bridge.closed_loop import ClosedLoop; print('CUDA:', torch.cuda.is_available()); print('IMPORT_OK')"
python -m carla_bridge.planning.pdm_planner --check
python -m pytest carla_bridge/tests -q -p no:cacheprovider
```

`--check` 核验 PDM 来源，不连接 CARLA 或模型 API。测试使用模拟场景接口，不能替代服务器真实驾驶。
若 `pip check` 报已有包与 NumPy/Shapely 等冲突，保留输出定位，不将检查失败当作安装完成。
这份流程未声称已在你的服务器新建环境完整安装并验收，需要执行上述检查确认。

DINOv2 两种准备方式二选一：

```bash
# 服务器可访问 Hugging Face 时，首次下载
python scripts/00_prepare_models.py --download

# 服务器不可访问 Hugging Face 时，先上传完整缓存，再离线验证
python scripts/00_prepare_models.py --offline
```

默认缓存是仓库 `.cache/huggingface`。缓存结构及异地路径参数见本文顶部“检查手工上传的 DINOv2 缓存”。
独立模型目录可用 `python scripts/00_prepare_models.py --offline --model-dir /实际路径/dinov2`；
此后驾驶命令也要添加同一个 `--dinov2-model-dir /实际路径/dinov2`。

核对 `config/api_models.yaml` 中场景理解与决策模型的服务地址、模型名和 `api_key_env`。
默认密钥变量可以在 Bash 当前终端中隐藏输入：

```bash
read -rsp "请输入 DASHSCOPE_API_KEY: " DASHSCOPE_API_KEY
echo
export DASHSCOPE_API_KEY
```

默认入口不会自动读取 `.env`，新终端需要重新设置或沿用你已有的环境变量管理方式。
模型缓存离线不意味着 VLM API 也离线；模型调用仍需访问配置的服务。

### 5.5 连接仿真器，再进入新场景实验

在另一终端沿用现有部署启动 CARLA；全新服务器安装好引擎后，可将下面路径替换为实际安装目录：

```bash
cd /实际安装路径/CARLA_0.9.15
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic -carla-rpc-port=2000
```

示例固定为 Epic 画质；各实验组保持相同画质设置。离屏渲染仍提供相机图像，
不要启用会使相机返回空数据的 `no_rendering_mode`，见
[CARLA 渲染说明](https://carla.readthedocs.io/en/0.9.15/adv_rendering_options/)。

回到已激活 Python 环境的仓库终端验证：

```bash
python - <<'PY'
import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(120.0)
print("客户端:", client.get_client_version())
print("服务端:", client.get_server_version())
print("地图:", sorted(client.get_available_maps()))
PY
```

版本一致且所需地图齐全后，继续 [12.0 的 prepare 和首轮驾驶命令](#120-从服务器开始生成场景再运行场景)，
再选择 [12.6 的记忆库方式](#126-新场景与记忆库模式完整命令)。
历史 `setup_env.sh` 仍是旧基础环境脚本，未包含完整 PDM 安装，不作为本节流程的替代。

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
报告在 `outputs/carla_runs/<run_id>/`，对比碰撞率/路线完成度等量化记忆增益。

### 6.4 驾驶视频录制
在 [config/carla.yaml](config/carla.yaml) 的 `carla.video` 块设 `enabled: true`：
```yaml
  video:
    enabled: true
    fps: 10
    width: 1280
    height: 720
    view: "chase"      # chase（后上方第三人称）| front（前视角）
```
跑完（或 Ctrl+C）视频在 `outputs/carla_runs/<run_id>/carla_videos/drive_<场景>_<模式>.mp4`，按仿真帧率编码，丢帧时可能短于仿真时长（跳过 VLM 冻结期）。

### 6.5 首测建议
直接验证完整路线，以 `route_completed` 和 100% 路线完成率验收。默认 300 秒是仿真时间保护上限，
实际墙钟耗时还包括每周期两次 VLM 调用；不能用实际运行了多久替代路线完成判定。

---

## 7. 配置说明

### 7.1 `config/carla.yaml`（关键项）
| 键 | 默认 | 说明 |
|---|---|---|
| `carla.host`/`port` | 127.0.0.1/2000 | CARLA 服务器 |
| `carla.synchronous`/`fixed_delta_seconds` | true/0.1 | 同步模式 + 10Hz |
| `carla.replan_interval_s` | 3.0 | 全量认知与语义更新周期 |
| `carla.planning.backend` | pdm_closed | PDM 核心；vlm_trajectory 为旧模型轨迹 |
| `carla.planning.semantic_influence` | true | 语义候选偏好评分开关，与记忆开关独立 |
| `carla.planning.replan_interval_s` | 0.2 | 两次语义更新之间的 PDM 规划周期 |
| `carla.max_duration_s` | 300.0 | 全局仿真时间保护上限，与场景 `duration_s` 取较小值；到时未完成为 `timeout` |
| `carla.capture_timeout_s` | 2.0 | 六路相机共享等待上限（墙钟秒） |
| `carla.max_capture_failures` | 3 | 连续缺帧达到阈值时结束运行 |
| `carla.cameras` | 6 相机 | 640×360，2×3 布局（P1 可调 transform） |
| `carla.controller` | — | wheelbase/max_steer/lookahead/PID 增益/`steer_sign` |
| `carla.video` | enabled:true | 视频录制开关与参数 |
| `--memory-bank` | 本轮 memory_db/ | 指定需要复用的中期记忆库 |
| `perception.mode` | surround_mosaic | 固定六视角环视 |
| `feature_extractor.cache_dir` | .cache/huggingface | DINOv2 缓存根目录，相对仓库根目录解析 |
| `feature_extractor.local_files_only` | true | CARLA 默认离线加载，可用 `--allow-model-download` 显式允许下载 |
| `feature_extractor.model_dir` | null | 可指定完整本地模型或 snapshot 目录，此时始终离线 |

> carla.yaml 由入口 `load_config(overrides=...)` 深合并进主配置，由 RunSession 进一步设置运行目录和记忆访问策略。

### 7.2 `config/scenarios/*.yaml`
```yaml
scenario_name: "straight_traffic"
map: "Town04"              # CARLA 地图
duration_s: 300            # 路线未完成时的仿真时间保护上限（秒）
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
- 行人事件沿指定线段运动、到达后停止，不自动绕障；随机交通行人仍使用导航网格 AI。
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
| 路线完成度 | 按顺序追踪规划路线；完整经过路线且当前距终点不超过默认 2 米才为 1.0，同时 `route_completed=true` |
| 闯红灯 / 逆行 / 超速 | 近红灯仍前进 / 与车道反向 / 超限速 |
| 最大速度 / 加速度 / jerk | 舒适度 |

报告：`outputs/carla_runs/<run_id>/carla_run_<场景>_<模式>.md` / `.json`。

---

## 10. 故障排查

| 现象 | 原因 / 解决 |
|---|---|
| `No module named pytest` | 在 `mulmem_carla` 中执行 `python -m pip install "pytest>=7.4,<9"` |
| DINOv2 本地文件缺失 | 运行 `python scripts/00_prepare_models.py --offline`，检查缓存根目录、snapshot 和链接目标；可显式传入本地模型目录 |
| 行人导航或退出时原生崩溃 | 保留入口自动输出的故障堆栈和 CARLA 日志；核对客户端/服务端均为 0.9.15，并用完整路线及正常退出验证 |
| 决策被拒绝后车辆制动 | 检查 JSONL 的 `decision_validation` 和原始决策，排查单位、轨迹时间与目标速度，拒绝帧不会写入中期或事件记忆 |
| `EnvironmentError: VLM API Key 未设置` | Git Bash 用 `export DASHSCOPE_API_KEY=...`（不是 `set`）；脚本 fail-fast 会在 setup 前报 |
| `FileNotFoundError: SSL_CERT_FILE` | conda 激活脚本指向不存在的 cacert.pem；`run_carla_demo.py` 已自动清除坏路径改用 certifi，无需手动处理 |
| `time-out waiting for the simulator` | CARLA 服务器没启动/卡死。`taskkill //F //IM CarlaUE4.exe` 后重启，等加载完 |
| 内存卡死 / CARLA 崩 | 6 相机已降到 640×360 + 控制阶段 `drain()` 防堆积；若仍崩，降 `carla.video` 分辨率或关视频 |
| VLM 调用久无日志 | 正常（5-30s/次），期间 sim 冻结。**别 Ctrl+C**，等 `[SCENE_UNDERSTANDING] raw_response` 出现 |
| 转向异常 / 弯道偏离 | 当前默认 `steer_sign=-1` 已有回归测试；先核对轨迹、坐标和配置，保留视频，不要直接翻转符号掩盖问题 |
| CARLA 窗口看不到车 | `set_spectator_follow` 每 tick 跟随；若没跟，确认 CARLA 已加载完且 ego 已生成 |

---

## 11. 开发进度

- [x] P0 环境/骨架：`mulmem_carla`(3.9) 已建并验证（carla+faiss+torch+记忆系统导入 OK）；coords/carla_client/config
- [x] P1 感知接入：6 相机 + ego 状态 + GT 感知 + 天气
- [x] P2 记忆管线对接：keyframe_builder + loop_runner（复用 `OnlineDrivingLoop`）
- [x] P3 闭环控制：轨迹转全局 + Pure Pursuit + PID + `closed_loop` 10Hz 跟踪
- [x] P4 场景系统：YAML + Python 场景类 + 交通流 + 天气 + 路由 + nav 派生
- [x] P5 评测：碰撞/违规/路线/舒适度 + 报告（已接入 `closed_loop`）
- [x] 3s 捕获 + 控制阶段每周期 5 次 raw 捕获（短期记忆时刻最新）+ 相机防堆积
- [x] spectator 第三人称跟随
- [x] 驾驶视频录制（跳过 VLM 冻结期，时长=仿真驾驶时间，中断保存）
- [x] fail-fast API key + SSL_CERT_FILE 自动修复 + 异常 traceback 打印
- [x] 导入冒烟测试通过（carla_bridge + src 全部导入 OK，坐标数学往返一致）
- [x] 长尾事件系统：EventScheduler + scripted 车(NOA+ACC) + 行人(过马路/路边) + 障碍物 + 信号灯控制
- [x] 本地回归覆盖库读写、隔离、帧同步、行人生命周期、离线模型加载与异常轨迹拦截
- [ ] P6 服务器验证：完整路线完成率 100%、清理完成且退出码 0；验证原生崩溃修复及 memory_on/off 对比

## 12. 服务器论文实验套件 paper_v1

这是项目自定义实验套件，不是 CARLA Leaderboard 或其他公开基准的复现。
配置来源为 [paper_suite.yaml](config/paper_suite.yaml)，工具为 [paper_suite.py](paper_suite.py)。
服务器已确认安装 Town01/02/03/04/05/10HD 及各自的 `_Opt` 版本；本套件统一使用普通版本，
不混用地图层设置。Town06 未安装，旧 NOA2 不包含在本套件中。
地图特征参考 [CARLA 官方地图说明](https://carla.readthedocs.io/en/0.9.15/core_map/)。

### 12.0 从服务器开始：生成场景，再运行场景

新场景与旧的 `straight_traffic` 不同：仓库内的 `paper_suite.yaml` 是**生成目录**，
不能直接作为 `--scenario` 输入。必须先在服务器 CARLA 的真实地图上执行 `prepare`，
得到冻结路线和事件位置的独立 YAML，再将该 YAML 的路径传给驾驶入口。
这些生成文件位于被 Git 忽略的 `outputs/`，因此仅 `git pull` 不会得到它们。

以下命令在服务器 Bash 中执行。首次使用 PDM 先完成
[第 5 节安装](#5-环境搭建)，后续只需激活已有环境：

```bash
cd /data/zjj/code/vlm_memory_system/Multi-level-Memory-System
conda activate mulmem_carla
python -m carla_bridge.planning.pdm_planner --check
python -m carla_bridge.paper_suite list
```

保持 CARLA 0.9.15 在 `127.0.0.1:2000` 运行；准备过程中会切换四张地图，
同一实例不能同时跑驾驶任务。首次生成全部 60 个 YAML（10 类条件 × 6 个 seed）：

```bash
python -m carla_bridge.paper_suite prepare \
  --host 127.0.0.1 --port 2000 --output outputs/paper_v1
```

已有 `outputs/paper_v1/preflight.json` 及完整 `train/`、`eval/` 时，可直接复用；
不要每次驾驶前重新生成。需要新版本时用 `--output outputs/paper_v2`，后续场景路径也相应替换。
`prepare` 不调用模型；实际驾驶仍需当前终端中已配置的模型 API Key 和可用 DINOv2 本地缓存。
缓存不在默认位置时，为下方驾驶命令追加实际 `--dinov2-model-dir` 或 `--dinov2-cache-dir`。
无需添加 `HF_HUB_OFFLINE=1`、`-X faulthandler`、`-u` 前缀。

**推荐第一轮：Town05 常规场景，PDM + 语义，无记忆。**

```bash
mkdir -p outputs/server_console
set -o pipefail
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1/eval/R05_multilane_eval_s101.yaml \
  --planner pdm_closed --semantic-influence on --mode memory_off \
  2>&1 | tee "outputs/server_console/R05_on_$(date +%Y%m%d_%H%M%S).log"
```

前一轮正常结束后，保持同一场景、seed 和配置，仅关闭语义评分：

```bash
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1/eval/R05_multilane_eval_s101.yaml \
  --planner pdm_closed --semantic-influence off --mode memory_off \
  2>&1 | tee "outputs/server_console/R05_off_$(date +%Y%m%d_%H%M%S).log"
```

再运行对应的行人横穿长尾场景：

```bash
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1/eval/L05_crossing_eval_s101.yaml \
  --planner pdm_closed --semantic-influence on --mode memory_off \
  2>&1 | tee "outputs/server_console/L05_crossing_on_$(date +%Y%m%d_%H%M%S).log"
```

此轮的关闭语义对照，只将同一命令的 `--semantic-influence on` 改成 `off`，同时更换日志文件名。
常规/长尾是在比较场景条件，`on/off` 是在比较语义评分；不要把二者同时改变的两轮当作单因素对照。
开关两侧仍调用模型；`off` 只去掉语义偏好评分，语义解析失败仍按共同规则制动。

文件名格式为 `<条件ID>_<train或eval>_s<seed>.yaml`。例如更换成占道停车：
`outputs/paper_v1/eval/L05_blocked_lane_eval_s101.yaml`；更换评估 seed：
`outputs/paper_v1/eval/L05_blocked_lane_eval_s202.yaml`。
入口没有 `--seed` 参数，seed 来自 YAML。完整条件列表见下表。

### 12.1 条件与对照

| 条件 ID | 地图 / 目标路线长度 | 场景内容 | 配对常规对照 |
|---|---|---|---|
| R03_urban | Town03 / 800 m | 城市路网、混合交通 | — |
| R04_extended | Town04 / 1200 m | 较长路线持续驾驶 | — |
| R05_multilane | Town05 / 900 m | 多车道路网、混合交通 | — |
| R10_downtown | Town10HD / 700 m | 高细节城区日间驾驶 | — |
| L05_crossing | Town05 / 同上 | 预置路边行人按距离开始横穿 | R05_multilane |
| L05_blocked_lane | Town05 / 同上 | 预置同向占道停车，选取有合法同向绕行车道的位置 | R05_multilane |
| L10_night_rain | Town10HD / 同上 | 夜间降雨 | R10_downtown |
| L03_fog | Town03 / 同上 | 浓雾视觉条件 | R03_urban |
| C05_crossing_blocked | Town05 / 同上 | 先后遇到横穿行人与占道停车 | R05_multilane |
| C10_night_crossing | Town10HD / 同上 | 夜雨和横穿行人组合 | R10_downtown；另与 L10_night_rain 比较 |

默认每轮请求 60 辆背景车和 20 名背景 AI 行人，事件 actor 另计。初始化必须达到请求数量，
不足就报错并记录实际数量，不静默变成稀疏场景。可在准备前统一修改 catalog 中的交通数量。
同一 profile 的对照和干预共用冻结路线、起点和预留事件区域，固定同一 seed 对比。
常规路线不保证经过地图的特定地标，Town04 路线也不宣称全程高速；必须看预检路线及视频。
行人场景不宣称有视线遮挡；雨雾测试不宣称自动改变轮胎摩擦。
当前管线仍开启 CARLA GT 障碍物输入（oracle）；雨雾不会自动降低这些真值对象的可见范围。
因此这套设置评估的是视觉、真值感知和记忆共同参与的决策系统，不应报告为纯视觉感知鲁棒性实验。

### 12.2 在服务器准备真实路线

本地可先查看定义，不连接 CARLA：

```bash
python -m carla_bridge.paper_suite list
```

新代码同步到服务器后，保持 CARLA 服务启动，**停止正在运行的闭环任务**，再准备：

```bash
conda activate mulmem_carla
python -m carla_bridge.paper_suite prepare --output outputs/paper_v1
```

`prepare` 会依次加载四个地图，重置当前 world，结束时停留在最后加载的地图。
它不运行 VLM、不生成交通 actor；检查地图/版本/蓝图、完整路线长度、人行道横穿位置、
可绕行车道以及事件点是否被路线的其他部分重复经过。不是只把旧场景里的地图名换掉。
如没有候选路线满足条件，会列出失败原因，不输出一个伪装为可用的套件。
这时可调整 catalog 的起点候选、路线长度或事件搜索区间，使用一个新的输出目录重试。

全部几何检查通过后生成：

```text
outputs/paper_v1/
├── preflight.json                  # 路线坐标、事件坐标、地图哈希、版本、catalog 哈希
├── train/                         # 10 条件 × seeds 11、22、33
│   └── R05_multilane_train_s11.yaml
└── eval/                          # 10 条件 × seeds 101、202、303
    └── R05_multilane_eval_s101.yaml
```

工具不会覆盖已有目录。生成的 YAML 是独立可运行场景；不要手改其路线/事件坐标，哈希校验会拒绝。
改变实验设计应修改 catalog 后重新准备，并保留原版用于追溯。
预检仅验证地图几何，不等于 actor 一定能生成、事件一定能完成或车辆已通过路线。

### 12.3 首轮驾驶验收与记忆实验

先用训练场景做一轮完整驾驶验收：

```bash
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1/train/R05_multilane_train_s11.yaml \
  --mode memory_off
```

然后在训练 seed 下积累库，可以按 case/profile 分库，也可以在多个训练场景使用同一个 general 库：

```bash
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1/train/L05_crossing_train_s11.yaml \
  --memory-bank outputs/memory_banks/paper_v1/train_general
```

其他训练条件/seed 更换 `--scenario`，库路径不变则继续积累。普通场景、长尾场景都应纳入训练覆盖，
不保证每一帧入库；不要用验证条件补充训练库。

以上命令默认使用 PDM 且语义评分开启，也可显式追加
`--planner pdm_closed --semantic-influence on`。持续积累默认是 `--memory-access read_write`。
如果只积累横穿行人专用库，将库路径统一替换为
`outputs/memory_banks/paper_v1/train_crossing`，并使用相应 `train` 场景。
如果只积累、不检索历史记忆，在训练命令中追加 `--no-memory-retrieval`；
若也不使用本轮短期上下文，再追加 `--no-short-term`。
这里的 train 指记忆库积累，**不训练或微调 VLM/PDM 权重**。

评估时冻结训练库，同一个 eval 场景、seed 分别运行以下对照：

```bash
# 无记忆基线
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1/eval/L05_crossing_eval_s101.yaml --mode memory_off

# 完整记忆，已有库只读；短期上下文开启
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1/eval/L05_crossing_eval_s101.yaml \
  --memory-bank outputs/memory_banks/paper_v1/train_general --memory-access read_only

# 去掉短期上下文，隔离持久化记忆的作用
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1/eval/L05_crossing_eval_s101.yaml \
  --memory-bank outputs/memory_banks/paper_v1/train_general --memory-access read_only --no-short-term

# 仅短期上下文：同一个只读库保持加载，但关闭中长期检索
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1/eval/L05_crossing_eval_s101.yaml \
  --memory-bank outputs/memory_banks/paper_v1/train_general --memory-access read_only --no-memory-retrieval
```

`eval` 场景禁止可写记忆模式；需要在线持续学习时另用训练场景和独立库开展实验，避免混入冻结库评估。

已有专用库只检索、不增加记忆的完整 PDM 示例：

```bash
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1/eval/L05_crossing_eval_s101.yaml \
  --planner pdm_closed --semantic-influence on --mode memory_on \
  --memory-bank outputs/memory_banks/paper_v1/train_crossing \
  --memory-access read_only
```

`read_only` 要求该路径已是可用记忆库，不能跳过积累步骤指向不存在的目录。
对同一冻结库分别使用 `--semantic-influence on` 和 `off`，即可检验记忆建议是否通过语义评分影响选择。
两轮使用同一只读库，不要中途继续向它写入。

分开的 seed 用于不同交通初始化，**不是未见地图/路线泛化划分**；3 个 seed 是初始配置，
论文所需样本数和统计不确定性需根据试验结果确定。相同 seed 不保证不同 VLM 调用或不同硬件逐位一致。
交通管理器和行人导航的种子都显式设置，参见 [CARLA TM 确定性说明](https://carla.readthedocs.io/en/0.9.15/adv_traffic_manager/)。

### 12.4 记录与验收

- `experiment_setup.json`：真实地图哈希、路线长度、请求/实际交通数、初始 actor、天气、seed 和完整 case 配置哈希。
- `events.json`：事件初始化、沿路线触发距离/时刻、运动证据、是否实际接近、完成或失败状态；关键状态变化即写盘。
- `manifest.json`：记忆模式、代码版本、是否有未提交修改，以及最终 `outcome.experiment_events`。
- 决策 JSONL、视频和原有运行报告仍在本轮独立目录内。

事件 actor 在驾驶前创建，不在车前突然凭空生成。行人等待达到**沿路线的累计进度**阈值才开始运动，
占道车辆从起步时就已存在。没有实际生成、没有触发/运动证据、或未实际接近的必需事件，不能当作已测试。
论文场景即使完成路线，事件覆盖不满足时也以 `scenario_invalid` 非零退出；
成功需要路线 100%、事件有效、正常清理及报告落盘。碰撞、违规仍独立报告，不能用路线成功掩盖。
每轮保护上限是 900 **仿真秒**，仅新准备的 paper 场景覆盖原 300 秒全局上限；旧场景不变。

汇总所有尝试（包括初始化失败、超时、事件无效，不只保留成功运行）：

```bash
python -m carla_bridge.paper_suite summarize --summary-csv outputs/paper_results_v1.csv
```

每行一轮运行，可用 case、seed、路线哈希、记忆模式和库路径配对。CSV 不覆盖已有文件。
比较不同记忆模式前，核对场景/路线哈希、初始 actor 布局、实际天气和交通规模；保留失败和缺失项，
区分基础设施无效、驾驶失败与成功，不将空白指标填成零，也不挑选最好的一轮当总体结果。

当前仍为单个 CARLA 实例的顺序实验。多张 GPU 不会自动合并显存或自动并行运行多个场景。
后续多实例并行需要独立的 CARLA/TM 端口和输出/库隔离；先验证单实例完整路线和事件效果。

### 12.5 单轮跑通后：顺序批量执行与收集反馈

先确认常规和长尾各一轮能完成，再批量运行。以下 Bash 命令运行 **eval 的全部 30 个 YAML ×
语义 on/off = 60 轮**，统一关闭记忆。它不训练记忆库，也不与同一 CARLA 实例上的其他任务并行。
每轮出现非零退出即停止，保留日志，防止服务故障后继续产生大量无效实验：

```bash
mkdir -p outputs/server_console
set -o pipefail
(
  shopt -s nullglob
  cases=(outputs/paper_v1/eval/*.yaml)
  if [ "${#cases[@]}" -eq 0 ]; then
    echo "未找到场景，请先执行 paper_suite prepare"
    exit 1
  fi
  for scene in "${cases[@]}"; do
    name=$(basename "$scene" .yaml)
    for influence in on off; do
      logfile="outputs/server_console/${name}_${influence}_$(date +%Y%m%d_%H%M%S).log"
      if python -m carla_bridge.run_carla_demo \
        --scenario "$scene" --planner pdm_closed \
        --semantic-influence "$influence" --mode memory_off \
        2>&1 | tee "$logfile"; then
        echo "已完成: $name / $influence"
      else
        echo "本轮失败，批次已停止，请检查: $logfile"
        exit 1
      fi
    done
  done
)

# 使用新文件名，汇总包含失败在内的全部 paper 运行
python -m carla_bridge.paper_suite summarize \
  --runs-dir outputs/carla_runs \
  --summary-csv "outputs/paper_results_$(date +%Y%m%d_%H%M%S).csv"
```

汇总扫描 `--runs-dir` 下所有 paper 运行，可能包含此前的试跑和重复运行；不会只选取本次批次。
正式统计应依据 run_id、case_id、seed、配置哈希和代码版本明确纳入哪些运行。
900 秒是每轮仿真时间保护上限；模型推理期间仿真暂停，因此现实耗时可能更长。

每轮独立目录 `outputs/carla_runs/<run_id>/` 的 `manifest.json` 应满足
`status=route_completed`、`outcome.success=true`、`outcome.route_completion=1.0`；
带事件的场景还要核对事件有效性。失败或未触发的事件不能因车走到终点就记为成功。
反馈时保留对应 `manifest.json`、`experiment_setup.json`、`events.json`（如有）、
`planner.jsonl`、`decisions_*.jsonl`、`carla_run_*.json` 和本节保存的控制台日志；
轨迹/变道表现有问题时附上该轮视频。准备阶段失败则提供 `prepare` 完整错误输出，
此时没有运行结果目录是正常的。

### 12.6 新场景与记忆库模式完整命令

先完成 `prepare`。以下 Bash 变量只是减少重复输入，在执行命令的同一终端定义：

```bash
SCENE_TRAIN=outputs/paper_v1/train/L05_crossing_train_s11.yaml
SCENE_EVAL=outputs/paper_v1/eval/L05_crossing_eval_s101.yaml
BANK_GENERAL=outputs/memory_banks/paper_v1/train_general
BANK_CROSSING=outputs/memory_banks/paper_v1/train_crossing
```

| 需求 | 场景划分 | 参数 | 是否修改持久库 | 本轮短期上下文 |
|---|---|---|---|---|
| 本轮从空库开始 | train | 不传 `--memory-bank` | 写入本轮独立库 | 开启 |
| 多轮持续积累通用库 | train | 相同库路径 + `read_write` | 是 | 开启 |
| 指定场景专用库 | train/eval | 明确传入专用库路径 | 由 access 决定 | 默认开启 |
| 只检索已有库 | eval | 已有库 + `read_only` | 否 | 开启 |
| 只积累、不检索历史 | train | `read_write --no-memory-retrieval` | 是 | 默认开启，可关闭 |
| 只用持久记忆、去掉短期 | eval | `read_only --no-short-term` | 否 | 关闭 |
| 去掉持久记忆检索、保留短期 | eval | `read_only --no-memory-retrieval` | 否 | 开启 |
| 从已有库复制后独立积累 | train | `--memory-seed` + 不存在的目标库 | 只修改新库 | 开启 |
| 无记忆对照 | train/eval | `--mode memory_off` | 不读不写 | 关闭 |

这里的检索开关针对中长期记忆检索；固定长期规则仍受原有 `long_term` 配置控制。
场景名不会自动决定使用哪个库，指定库不会自动筛选该库内部的某一种场景记录。
想得到专用库，应从空目录开始，只在对应训练场景向该库积累。

**本轮独立库：** 默认从空库开始，产物及记忆保存在本轮运行目录。重复执行会创建新库，不会续写上一轮。

```bash
python -m carla_bridge.run_carla_demo \
  --scenario "$SCENE_TRAIN" --planner pdm_closed --semantic-influence on \
  --mode memory_on
```

**持续积累通用库：** 相同路径会加载历史并继续积累。换成其他 `train` YAML 或 seed 22/33，仍使用同一路径。

```bash
python -m carla_bridge.run_carla_demo \
  --scenario "$SCENE_TRAIN" --planner pdm_closed --semantic-influence on \
  --mode memory_on --memory-bank "$BANK_GENERAL" --memory-access read_write
```

**单独积累横穿行人专用库：** 该库与通用库独立；重复此命令或更换对应训练 seed 可以续写。

```bash
python -m carla_bridge.run_carla_demo \
  --scenario "$SCENE_TRAIN" --planner pdm_closed --semantic-influence on \
  --mode memory_on --memory-bank "$BANK_CROSSING" --memory-access read_write
```

**只检索已有专用库，不增加记忆：** 必须先完成上面的积累。只读模式不写回条目、淘汰结果或检索统计，
但仍维护本轮短期上下文，并正常写运行日志、图像和报告。

```bash
python -m carla_bridge.run_carla_demo \
  --scenario "$SCENE_EVAL" --planner pdm_closed --semantic-influence on \
  --mode memory_on --memory-bank "$BANK_CROSSING" --memory-access read_only
```

**只积累、不使用历史检索或本轮短期上下文：** 如需保留短期上下文，删除 `--no-short-term`。
为续写保留旧条目，程序仍会加载已有库；“不检索”不等于“删除旧库”。

```bash
python -m carla_bridge.run_carla_demo \
  --scenario "$SCENE_TRAIN" --planner pdm_closed --semantic-influence on \
  --mode memory_on --memory-bank "$BANK_GENERAL" --memory-access read_write \
  --no-memory-retrieval --no-short-term
```

**持久记忆与短期上下文的分别对照：** 两轮都读取同一冻结库，不新增记忆。

```bash
# 保留持久记忆检索，关闭短期上下文
python -m carla_bridge.run_carla_demo \
  --scenario "$SCENE_EVAL" --planner pdm_closed --semantic-influence on \
  --mode memory_on --memory-bank "$BANK_CROSSING" --memory-access read_only \
  --no-short-term

# 保留短期上下文，关闭持久记忆检索
python -m carla_bridge.run_carla_demo \
  --scenario "$SCENE_EVAL" --planner pdm_closed --semantic-influence on \
  --mode memory_on --memory-bank "$BANK_CROSSING" --memory-access read_only \
  --no-memory-retrieval
```

**复制历史库做独立积累实验：** 源库保持不变；目标目录必须不存在，也不能与源目录相互包含。
下例首次复制后继续积累；后续续写目标库时去掉 `--memory-seed`。

```bash
python -m carla_bridge.run_carla_demo \
  --scenario "$SCENE_TRAIN" --planner pdm_closed --semantic-influence on \
  --mode memory_on --memory-seed "$BANK_CROSSING" \
  --memory-bank outputs/memory_banks/paper_v1/crossing_experiment_a \
  --memory-access read_write
```

**无记忆基线：** 不能再同时指定 `--memory-bank` 或 `--memory-seed`。

```bash
python -m carla_bridge.run_carla_demo \
  --scenario "$SCENE_EVAL" --planner pdm_closed --semantic-influence on \
  --mode memory_off
```

语义开关与上述记忆模式独立。将同一条评估命令的 `--semantic-influence on` 改为 `off`，
即可组成“有/无记忆 × 有/无语义评分”四组；`off` 保留模型调用及共同的解析失败制动规则。
四组应使用相同场景 YAML、seed、规划参数和冻结库快照，并保留重复试验及失败记录。
不要使用 eval 场景写库，也不要在只读对照之间继续更新训练库。

记忆准入存在价值门控，不保证每帧都入库；被规划/校验拒绝的决策不进入中期或事件记忆。
检查决策 JSONL 的 `memory_retrieval_enabled`、`memory_write_enabled`、`retrieved_memory_ids`，
以及 manifest 的 `memory_bank`、`memory_access` 和 `config.memory_access` 确认实际模式。
同一库由进程锁串行访问；备份或迁移时保留整个库目录和 `assets/`，
旧库若引用外部图像/特征还需保留这些文件，或用 `--memory-seed` 将可用资料复制入新库。

## 13. 语义决策与 PDM-Closed

CARLA 默认使用 `pdm_closed`。模型只输出横向、纵向意图及文字解释；轨迹位置、时间与速度
由规划器生成。离线 nuScenes 入口仍保留原有数值轨迹输出；CARLA 可通过
`--planner vlm_trajectory` 显式选择旧路径。

### 13.1 官方核心与 CARLA 适配

复用 [tuPlan Garage 的 PDM-Closed 核心](https://github.com/autonomousvision/tuplan_garage)：
候选生成、运动仿真、指标评分及紧急制动检测。上游原版运行于 nuPlan，本项目提供 CARLA 的
地图、障碍物、车辆状态与坐标适配，并增加语义评分与候选策略。因此论文应表述为
**使用官方 PDM 核心的 CARLA 适配版本**，不能将本项目实验分数称为官方 nuPlan PDM-Closed 基准成绩。

首次使用先按 [规划器依赖与来源说明](planning/README.md) 安装固定版本依赖；不需要下载
nuPlan 数据集或额外模型权重。依赖检查不连接 CARLA、不调用 VLM：

```bash
python -m carla_bridge.planning.pdm_planner --check
python -m pytest -q -p no:cacheprovider
```

缺依赖立即报错，不自动换成其他规划器。DINOv2 的离线缓存加载方式保持原有入口配置。
本地单元测试及官方核心合成场景测试不能替代服务器 CARLA 完整路线验收。

### 13.2 语义接口与控制周期

模型必须返回单个 JSON 对象，例如：

```json
{
  "lateral_decision": "FOLLOW_ROUTE",
  "longitudinal_decision": "DECELERATE",
  "behavior_reason": "前方行人接近车道，提前减速观察。",
  "risk_level": "medium",
  "safety_notes": ["保持路线，留出制动距离。"]
}
```

| 维度 | 允许值 | 含义 |
|---|---|---|
| 横向 | `FOLLOW_ROUTE`、`KEEP_LANE`、`CHANGE_LANE_LEFT`、`CHANGE_LANE_RIGHT` | 路线跟随、保持车道、请求向左/右合法变道 |
| 纵向 | `MAINTAIN`、`ACCELERATE`、`DECELERATE`、`FOLLOW`、`YIELD`、`STOP` | 保持、加速、减速、跟车、让行、停车偏好 |
| 风险 | `low`、`medium`、`high` | 用于解释与记忆上下文 |

模型不得给出 `trajectory`、`target_speed` 等数值控制字段；导航路线决定道路转向，
语义请求不能改写目的地或使禁止变道的路段变成合法变道。不可用的横向请求由候选诊断记录。
候选先经过统一安全筛选，语义仅改变合格候选的偏好评分。语义不是无条件执行命令。

默认每 **3 秒仿真时间**更新一次模型语义，每 **0.2 秒**使用最新 CARLA 状态重新规划，
控制器每 **0.1 秒**按轨迹时间插值速度。规划不额外推进 world tick。
无有效语义、规划异常或轨迹校验失败时制动；失败决策不会写入中期或事件记忆。
语义格式失败的共同处理规则对 `on`、`off` 都生效。

### 13.3 对照开关与记忆实验

```bash
# 模型语义参与 PDM 候选选择；默认即为 on
python -m carla_bridge.run_carla_demo --scenario straight_traffic --planner pdm_closed --semantic-influence on --mode memory_off

# 同样运行模型，但仅使用 PDM 基础评分选择候选
python -m carla_bridge.run_carla_demo --scenario straight_traffic --planner pdm_closed --semantic-influence off --mode memory_off

# 指定既有训练库只读检索，语义参与选择
python -m carla_bridge.run_carla_demo --scenario straight_traffic --planner pdm_closed --semantic-influence on --memory-bank outputs/memory_banks/pdm_train --memory-access read_only

# 旧版模型直接输出轨迹，用于历史方法对照
python -m carla_bridge.run_carla_demo --scenario straight_traffic --planner vlm_trajectory --mode memory_off
```

`--semantic-influence off` 保留模型调用、语义解析与日志，只关闭候选的语义评分，
因此不是“完全不调用模型”的计算成本基线。给定同一车辆/地图/交通状态，开关两侧生成
相同候选并使用相同安全检查；行驶动作不同后，两轮未来状态自然可能分化。

`--mode memory_off` 与语义开关是两个独立实验维度。做对照时固定场景、seed、路线、
规划参数、模型配置及评估库快照；不要先让一个实验组向同一库写入，再让另一个组读取新增内容。
持续积累仍使用 `--memory-bank ... --memory-access read_write`；评估使用只读库。
新旧方法的训练库建议分开保存，旧库不会被自动迁移或覆盖。

### 13.4 输出与验收

每轮 `outputs/carla_runs/<run_id>/` 中：

- `manifest.json` 保存完整规划配置、语义开关、记忆策略及最终路线结果。
- 决策 JSONL 的 `model_decision_output` 保留模型语义，`decision_output` 保存规划后的指令与轨迹，
  `planning_output` 保存候选诊断，`decision_validation` 保存执行前数值校验结果。
- `planner.jsonl` 记录每次 PDM 更新的仿真时间、关联语义帧、候选基础分/语义分/最终选择和校验结果。
- 记忆中的 `semantic_decision` 是模型建议，`planning_context` 是规划器选择；
  记录阶段为 `selected_before_execution`，不能据此断言已经成功完成该动作。
- `paper_suite summarize` 增加 `planner_backend`、`semantic_influence` 和规划配置哈希，方便分组。

验收仍要求**路线完成率 100% 且正常清理退出**，同时审查碰撞、违规和场景事件结果。
本实现使用 CARLA GT 障碍物与地图，PDM 局部预测和 CARLA 实际控制存在模型差异；
不能将通过候选检查解释为真实仿真中一定无碰撞。
