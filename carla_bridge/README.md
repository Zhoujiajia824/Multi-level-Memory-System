# CARLA 闭环集成（carla_bridge）

把多层次记忆系统（`src/vla_memory`）接入 CARLA 0.9.15：用 CARLA 实时
感知（六视角图像 / GT 障碍物 / 天气 / 导航 / 自车状态）替代离线 nuScenes，决策轨迹
实时回控 CARLA，实现**自定义环境下的闭环驾驶**，并支持驾驶视频录制回看。

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
| **10Hz 回控** | Pure Pursuit(横向) + PID(纵向) 跟踪决策轨迹，同步模式 |
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
        │                                    ▼  decision = {behavior, trajectory(ego-centric)}
        │  ◀──  trajectory  ────────────────
        ▼
trajectory_tracker (Pure Pursuit + PID)  ->  carla.VehicleControl  ->  vehicle.apply_control
        │
        ├── metrics: 碰撞/违规/路线/舒适度 逐 tick 采集
        └── video: 每个 CARLA 推进 tick 取一帧（跳过 VLM 冻结）写视频
```

**单周期时序（每 3s sim）：**
```
[1 捕获]  tick -> 6 相机 mosaic + ego + GT 感知 + nav         (推进 0.1s)
[2 认知]  loop.step(kf) -> 决策轨迹                           (sim 冻结, wall 5-30s)
[3 控制]  10Hz × 30 tick：Pure Pursuit+PID 回控 CARLA          (推进 3.0s)
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
| `carla.replan_interval_s` | 3.0 | 3s 全量捕获（= decision.trajectory.horizon_seconds） |
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
