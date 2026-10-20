# CARLA 闭环集成（carla_bridge）

把多层次记忆系统（`src/vla_memory`）接入 CARLA 0.9.15：用 CARLA 实时
感知（六视角图像 / GT 障碍物 / 天气 / 导航 / 自车状态）替代离线 nuScenes。
默认链路为 PDM-Closed 候选规划与条件语义顾问 → 所选原生 LQR 转向 + 速度曲线 PID 回控。
支持通过 `--semantic-influence on|off` 做语义影响对照；安装依赖及使用方式见[第 13 节](#13-语义决策与-pdm-closed)。

首次配置按顺序阅读：[第 5 节：新环境或已有环境安装](#5-环境搭建) →
[第 12.0 节：生成与运行新场景](#120-从服务器开始生成场景再运行场景) →
[第 13.6 节：策略与记忆完整命令](#136-选择性语义顾问与数值规划记忆完整命令)。
运行后的模型调用、图像、输入输出与视频定位见[第 13.11 节](#1311-模型调用审核视频定位与仿真轨迹绘制)。
全部入口开关、当前默认值与记忆组合速查见[第 13.12 节](#1312-运行开关总表与记忆使用形式)。

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
CARLA 禁用模型解析失败的规则轨迹兜底，保留原始失败供定位；`always` 失败时制动，
语义开启的 `selective` / `shadow` 建议不可用时保留基础 PDM，数值轨迹无效仍制动。

## 运行隔离与记忆库选择（当前入口行为）

在仓库根目录、`mulmem_carla` 环境运行。每次生成独立的
`outputs/carla_runs/<运行目录>/`，包含 manifest、决策、报告、视频和日志。
manifest 记录实际配置、场景、代码提交号、工作区修改状态、记忆库来源和结束状态。
新运行目录使用 `时间__场景文件名__记忆模式__规划器__语义开关__策略__数值记忆__短编号`，例如：

```text
20261019_143025__L05_crossing_eval_s101__memory_off__pdm_closed__sem_on__policy_selective__pm_full__7efbaf53
```

时间取运行机器的本地时间，场景文件名保留评估/训练划分与种子。
记忆模式为 `memory_off`、`memory_read_only` 或 `memory_read_write`；
关闭检索或短期记忆时，分别追加 `_no_retrieval`、`_no_short_term`。
PDM 的语义开关为 `sem_on` / `sem_off`，旧轨迹后端使用 `sem_na`。
场景名中的特殊字符会替换为下划线，最多保留 64 个字符。
旧配置未设置新字段时保留旧目录标签；Windows 路径较长时使用紧凑目录名，完整配置以 manifest 为准。
启动时终端打印完整输出路径，无须修改运行命令。

内部 `run_id` 仍为 16 位短 UUID，帧编号继续使用它；目录末尾取其前 8 位以区分重复运行。
manifest 额外保存 `run_directory` 和带时区的 `started_at`。
旧目录不会重命名，汇总脚本兼容新旧目录；已有记忆库仍按原路径指定。

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
| **条件或周期认知** | 默认 selective、语义 on 时按当前 PDM/GT 证据查询；always 或语义 off 保留模型调用时，每3s sim认知；model-inference off完全跳过 |
| **raw 捕获** | 旧周期认知路径每周期5次 raw 捕获供短期上下文；选择性路径在实际咨询时捕获模型图像，未咨询时排空相机队列 |
| **5Hz PDM 规划** | 基于最新 world 状态和最近语义，生成、仿真、评分轨迹候选 |
| **10Hz 回控** | 默认执行 PDM 已评分的 LQR 转向样本，PID 跟踪时间速度曲线；可显式选择旧 Pure Pursuit 横向对照 |
| **同步模式** | VLM 思考期间 CARLA 冻结（不 tick），wall-clock 慢但 sim 时间连续正确 |
| **驾驶视频录制** | 仅在 CARLA 推进 tick 取帧（跳过 VLM 冻结期），按仿真帧率编码，丢帧时视频可能短于仿真时长 |
| **CARLA 全优势** | 自定义地图/天气/交通流/车辆/路由，全 Python API |

---

## 2. 架构总览

```
CARLA server (CarlaUE4.exe, 端口2000, 手动启动, 同步模式)
        │   传感器/GT ↓   carla.VehicleControl ↑
        ▼
carla_bridge.closed_loop  ──条件查询/旧周期──>  keyframe_builder
        │                                    │  组装 kf dict（与 nuScenes 同构）
        │                                    ▼
        │                 src.vla_memory.pipeline.OnlineDrivingLoop.step(kf)
        │                                    │  DINOv2 + 场景VLM + 三层记忆 + 决策VLM
        │                                    ▼  decision = {lateral_decision, longitudinal_decision, ...}
        │       CARLA adapter → PDM generator → simulator → scorer + optional semantic preference
        │  ◀──  validated trajectory + time/speed profile  ────────
        ▼
trajectory_tracker (PID + 横向基础跟踪) -> native_tracking (PDM 所选 LQR 转向) -> vehicle.apply_control
        │
        ├── metrics: 碰撞/违规/路线/舒适度 逐 tick 采集
        └── video: 每个 CARLA 推进 tick 取一帧（跳过 VLM 冻结）写视频
```

**旧周期认知时序（always，或语义 off 但模型 on；每 3s sim）：**
```
[1 捕获]  tick -> 6 相机 mosaic + ego + GT 感知 + nav         (推进 0.1s)
[2 认知]  loop.step(kf) -> 横纵向语义 -> PDM轨迹              (sim 冻结, wall 5-30s)
[3 控制]  10Hz × 29 tick：所选 LQR 转向 + PID 回控 CARLA     (推进 2.9s)
          每 0.2s PDM 根据最新 world 状态重规划；和下一捕获 tick 合计 3.0s
          其中每周期 5 次 raw 捕获 push 进短期记忆；其余 tick drain 相机防堆积
[4 记录]  逐 tick 指标采集 + 视频取帧
```

默认选择性路径先按 10Hz 推进/控制、每0.2s更新PDM，再每0.5s检查是否需要咨询。
咨询仍同步冻结仿真，返回后更新短期建议；没有建议或建议失效时继续基础 PDM。
无模型路径不经过图中的模型管线。完整行为与边界见第13节。

---

## 3. 项目框架

```
carla_bridge/
├── README.md                       # 本文件
├── __init__.py
├── setup_env.sh                    # 一键建 mulmem_carla(3.9) 环境
├── closed_loop.py                  # 【主驱动】同步tick + 条件/周期认知 + PDM更新 + 视频
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
- **`planning/native_tracking.py`**：PDM 默认用实际所选 rollout 的短期 LQR 转向覆盖上述横向指令，纵向 PID 保持不变；旧 VLM 路点模式仍走 Pure Pursuit。
- **`trajectory_validation.py`**：检查轨迹时间、有限数值及位移/速度合理性；拒绝结果触发制动，写入 `decision_validation` 审计，并禁止该决策写入中期或事件记忆。
- **`scripted_vehicle.py`**：NPC 的 NOA（路由跟随）+ ACC（前方同车道有车则减速），复用 Pure Pursuit+PID。

### 4.5 `memory_adapter/` 记忆适配
- **`keyframe_builder.py`**：组装 `step()` 所需 kf dict（sample_token/scene/image_path/ego/history/nav/perception）。
- **`loop_runner.py`**：薄封装 `OnlineDrivingLoop`；`add_short_term_item()` 供控制阶段 raw 捕获直接 push 短期记忆。

### 4.6 `metrics/` 评测
- **`collector.py`**：碰撞（collision sensor）、闯红灯、逆行、超速、路线完成度、舒适度（jerk/加速度）逐 tick 采集。
- **`run_reporter.py`**：写 Markdown + JSON 报告到 `outputs/carla_runs/<运行目录>/`。

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
PDM 依赖清单同时包含 nuPlan 间接导入所需的 matplotlib/图像处理/pytest，
以及 tuplan 元数据要求的 scikit-learn 1.2.2、positional-encodings 6.0.1。
已有 scikit-learn 版本会按该要求调整；不能仅凭两个源码包安装成功就跳过依赖检查。
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
报告在 `outputs/carla_runs/<运行目录>/`，对比碰撞率/路线完成度等量化记忆增益。

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
跑完（或 Ctrl+C）视频在 `outputs/carla_runs/<运行目录>/carla_videos/drive_<场景>_<模式>.mp4`，按仿真帧率编码，丢帧时可能短于仿真时长（跳过 VLM 冻结期）。

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
| `carla.replan_interval_s` | 3.0 | always／语义off保留模型时的认知周期；selective/shadow语义on采用条件查询 |
| `carla.planning.backend` | pdm_closed | PDM 核心；vlm_trajectory 为旧模型轨迹 |
| `carla.planning.semantic_influence` | true | 语义候选偏好评分开关，与记忆开关独立 |
| `carla.planning.semantic_policy` | selective | selective条件介入／always旧版评分／shadow只记录假设选择；用于PDM后端 |
| `carla.planning.planning_memory` | full | off关闭数值经验／gate辅助咨询／full另提供参数候选；memory_off优先禁读写 |
| `carla.planning.model_inference` | true | false跳过双VLM、DINOv2、文本/视觉记忆和认知相机；要求PDM、语义off，以及memory_off或显式full数值记忆 |
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

报告：`outputs/carla_runs/<运行目录>/carla_run_<场景>_<模式>.md` / `.json`。

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
上述命令使用当前默认 selective。语义 on 时条件查询、建议失败回到基础PDM；off但模型on时
保留旧周期认知与失败制动，所以这是整体语义策略对照，并非只改变一个评分权重。
只比较旧版语义加分时，两边均追加 `--semantic-policy always --planning-memory off`；
新策略和快速纯PDM的明确矩阵见第13.6节。

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
它不运行 VLM、不生成背景交通；检查地图/版本/蓝图、完整路线长度、人行道横穿位置、
可绕行车道以及事件点是否被路线的其他部分重复经过。准备阶段会临时生成事件 actor
来验证坐标，每次生成后先推进一帧（同步模式）或等待新快照（异步模式），确认 actor
存在且存活后再销毁；真正的销毁或服务器通信失败会停止准备，不能当作几何不合格跳过。
这是因为 [CARLA 0.9.15 的 destroy 实现](https://github.com/carla-simulator/carla/blob/0.9.15/LibCarla/source/carla/client/Actor.cpp#L95)
会先检查快照中的 actor 状态，未初始化的探针也可能被报为“already dead”。若销毁返回
False，仅在另一张新快照确认 actor 已消失时接受清理完成；初始化后已死亡或消失的探针
仍判定为无效候选。准备阶段推进帧不改变运行期事件回调不额外 tick 的约束。
如没有候选路线满足条件，会列出失败原因，不输出一个伪装为可用的套件。
这时可调整 catalog 的起点候选、路线长度或事件搜索区间，使用一个新的输出目录重试。

全部几何及事件生成检查通过后生成：

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
预检验证空地图上的事件出生点；运行时背景交通可能改变可用空间，双端能生成也不等于
行人能穿过中间的护栏或顺利完成横穿。落地、横穿运动和车辆路线仍需要闭环验收。

#### 旧版横穿场景提示出生点碰撞时

`Spawn failed because of collision at spawn position` 表示固定点不能容纳该行人碰撞体，
不是模型或 PDM 依赖问题。旧生成器采用首个匹配的人行道边缘点并只加 0.3m 高度，
且未实际试生成。新版在横截面搜索两侧各有 0.5m 余量的候选，采用
[CARLA ScenarioRunner 0.9.15 横穿场景](https://github.com/carla-simulator/scenario_runner/blob/v0.9.15/srunner/scenarios/object_crash_vehicle.py#L409)
的地面高度加 1.2m 设置，用实际蓝图验证起点和终点；失败则在准备阶段继续有限搜索。
静止障碍车也必须通过生成检查。成功的坐标、蓝图和 `spawn_validation` 证据一起冻结到 YAML。
运行时不会随机挪点、换蓝图或跳过必需事件。这个高度选择仍需实际生成检查，不能保证任何地图点都可用。

**拉取代码不会修改已有 `outputs/paper_v1/*.yaml` 坐标。** 已经生成过旧场景时，
新代码同步后在空闲 CARLA 上创建新目录，再切换 `--scenario`：

```bash
conda activate mulmem_carla
python -m carla_bridge.paper_suite prepare --output outputs/paper_v1_spawncheck

mkdir -p outputs/server_console
set -o pipefail
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1_spawncheck/eval/L05_crossing_eval_s101.yaml \
  --planner pdm_closed --semantic-influence on --mode memory_off \
  2>&1 | tee "outputs/server_console/L05_on_$(date +%Y%m%d_%H%M%S).log"
```

保留旧目录用于追溯，不手改 YAML 或绕过哈希校验。语义开/关和记忆库各组使用同一套
新冻结场景；新旧事件坐标、预留位置及背景采样可能不同，结果不能当作同一几何条件直接配对。

### 12.3 首轮驾驶验收与记忆实验

先用训练场景做一轮完整驾驶验收：

```bash
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1/train/R05_multilane_train_s11.yaml \
  --planner pdm_closed --semantic-influence off --model-inference off \
  --planning-memory off --mode memory_off
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

每轮独立目录 `outputs/carla_runs/<运行目录>/` 的 `manifest.json` 应满足
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

这里的权限对中长期文本/视觉记忆与独立数值经验共同生效；`--no-memory-retrieval`
也关闭数值经验检索。固定长期规则仍受原有 `long_term` 配置控制。
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
可组成“有/无记忆 × 有/无语义影响”四组；默认selective的两侧查询调度和建议失败处理不同。
需要旧版无条件语义加分对照时，各组显式使用 `--semantic-policy always --planning-memory off`。
当前版本的车道成本/持续绕行默认开启；若要禁用这两项新数值机制，另加
`--lane-quality off --lane-maneuver off`。严格复现历史结果仍须使用当时的完整代码与控制配置。
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
候选生成、运动仿真和指标评分。上游原版运行于 nuPlan，本项目提供 CARLA 的
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

默认 `--semantic-policy selective` 使用六字段协议 v2；`always` 保留原五字段协议。
模型只提出短期建议，是否调用和采纳由规划顾问门控分别判断。例如无需介入时：

```json
{
  "lateral_decision": "FOLLOW_ROUTE",
  "longitudinal_decision": "MAINTAIN",
  "behavior_reason": "当前无需要补充的语义信息。",
  "risk_level": "low",
  "safety_notes": [],
  "intervention": {
    "needed": false,
    "reason_code": "none",
    "target_actor_ids": [],
    "target_stop_line_id": null,
    "activation": "immediate",
    "release": "condition_cleared",
    "valid_for_s": 1.0
  }
}
```

| 维度 | 允许值 | 含义 |
|---|---|---|
| 横向 | `FOLLOW_ROUTE`、`KEEP_LANE`、`CHANGE_LANE_LEFT`、`CHANGE_LANE_RIGHT` | 路线跟随、保持车道、请求向左/右合法变道 |
| 纵向 | `MAINTAIN`、`ACCELERATE`、`DECELERATE`、`FOLLOW`、`YIELD`、`STOP` | 保持、加速、减速、跟车、让行、停车偏好 |
| 风险 | `low`、`medium`、`high` | 用于解释与记忆上下文 |

模型不得给出 `trajectory`、`target_speed` 等数值控制字段；导航路线决定道路转向，
语义请求不能改写目的地或使禁止变道的路段变成合法变道。不可用的横向请求由候选诊断记录。
`always` 保留旧版无条件语义加分，当前数值分为 `PDM基础分 - 车道软成本`，
0基础分候选仍可能被旧语义分选中。
默认 `selective` 保留有限候选及原生基础分，只在当前证据授权、建议未过期且候选相对
数值最佳分的损失不超过预算时，提供有上限的语义加分。默认数值分损失预算和语义上限
各为 0.08；这限制顾问影响，不是恢复碰撞/道路/方向/TTC/红灯的全局硬筛选。
`shadow` 记录同快照下的假设选择，实际总执行同一数值PDM选择；车道居中/绕行数值模块也保持开启。
执行前数值与动力学一致性校验继续生效，两个数值开关见13.8。

适配版本 5 修正道路交界漏面和邻道导航连接。邻道通向直行/左转而本路线要求右转时，
候选需要包含合法返回目标车道的空间路径；无法完成驶入并返回的新变道不生成。
这改变普通候选几何，不改变上面的评分公式，语义开关两侧的普通候选使用相同构造；
13.7的模型恢复请求可另外增加局部路径，并对整池重新评分。
`KEEP_LANE` 候选在出口前也可能包含导航所需的返回段。详见
[路线几何与诊断说明](planning/README.md#路线几何与规划失败诊断)。

规划与执行共用实测轴距、前轮距和后轴参考点；归一化转向、内轮转角及 bicycle
等效中心角分别换算。PDM 初始轮角优先使用同步帧实测值。
LQR纵向保留原生1s制动预测，横向可用 `--pdm-lateral-preview speed_scaled` 按实测速度
在1–2s间调度；`fixed` 固定使用 `planning.lqr_lateral_horizon_s`（默认2s）。
原七种纵向策略之外新增一个缓起步 IDM 候选，保留常规加速和急刹策略参数；
是否选中仍由上述评分决定。分轴预测与新增候选的依据和局部复现见规划说明。

`always` 或语义off但模型on时，每 **3 秒仿真时间**更新模型语义；
`selective` / `shadow` 在语义on时按条件查询并限制查询频率。
每 **0.2 秒**使用最新 CARLA 状态重新规划，
控制器每 **0.1 秒**按轨迹时间插值速度。规划不额外推进 world tick。
语义on的 `selective` / `shadow` 无有效建议、格式失败或调用失败时继续使用基础 PDM；数值规划异常
或轨迹校验失败仍制动。`always` 保持原失败处理；失败决策不会写入中期或事件记忆。
模型仍同步查询，查询时仿真冻结；条件触发能减少查询次数，尚未实现异步推理。

### 13.3 对照开关与记忆实验

旧 `always` 提示词版本 5 位于 `config/decision.yaml` 的 `semantic.system_prompt` 和 `semantic.user_prompt`；
默认 `selective` / `shadow` 使用 `selective_system_prompt`、`selective_user_prompt`（schema v2、prompt v6）。
`pdm_closed` 不再使用 `config/api_models.yaml` 中要求输出路点的旧决策系统提示词；
`vlm_trajectory` 保持原系统契约。场景理解提示词仍在 `config/prompts.yaml`。
修改后重启驾驶进程使缓存配置生效；`[DECISION_PROMPT]` 日志记录实际系统/用户提示词哈希。

提示词以当前证据与通行条件选择动作，畅通时持续前进；六个纵向动作中性定义，
不使用长篇 STOP/YIELD 或低速探查示例引导。`behavior_reason` 只需一句短理由，
建议不超过 40 字，`safety_notes` 通常为 `[]`。v2 额外要求完整 `intervention`；
未提供对象、停止线或当前证据的建议不会从理由文字中推断授权。
历史轨迹显示相对时间，并从现有连续位姿计算有界的近静止时长，不把等待时间当作自动放行条件。
该运动窗口来自各记忆模式共同的传感器历史，不增加 memory_off 的检索或短期记忆。
路线锚点与真实 actor 分开显示；静止时长不能证明已在停止线完成停车，视觉估距也不是精确停止位置。
场景理解只报告观察证据，不从单帧臆测自车正在减速或直接发布停车命令。
版本 5 补充当前 CARLA 实测状态优先于单帧运动猜测；路面 STOP 文字不等于持续停车命令。
可选导航上下文来自当前地图和有序路线，注明快照时间，不把车道身份当作通行权。

验证时使用同一 prepared 场景、seed 和记忆配置，核对模型原始语义、PDM 最终选择、
停车后是否恢复以及碰撞/违规/完整路线结果。提示词与回归测试不能证明模型一定选择正确，
仍需服务器闭环验收；不要用计时器强制车辆起步。

```bash
# 模型语义参与 PDM 候选选择；默认即为 on
python -m carla_bridge.run_carla_demo --scenario straight_traffic --planner pdm_closed --semantic-influence on --mode memory_off

# 同样运行模型，但仅使用 PDM 基础评分选择候选
python -m carla_bridge.run_carla_demo --scenario straight_traffic --planner pdm_closed --semantic-influence off --mode memory_off

# 纯 PDM 快速运行：不初始化或调用模型，不需要 API Key 或 DINOv2 缓存
python -m carla_bridge.run_carla_demo --scenario straight_traffic --planner pdm_closed --semantic-influence off --model-inference off --mode memory_off

# 指定既有训练库只读检索，语义参与选择
python -m carla_bridge.run_carla_demo --scenario straight_traffic --planner pdm_closed --semantic-influence on --memory-bank outputs/memory_banks/pdm_train --memory-access read_only

# 旧版模型直接输出轨迹，用于历史方法对照
python -m carla_bridge.run_carla_demo --scenario straight_traffic --planner vlm_trajectory --mode memory_off
```

单独 `--semantic-influence off` 不代表关闭模型：仍保留每个认知周期调用的计算对照行为。
`selective` / `shadow` 在语义开关为 on 时才进入条件查询。彻底关闭调用需 `--model-inference off`。
比较语义影响时固定 `--planning-memory` 和库快照；`full` 的数值参数先验可扩展候选，
不能把同时改变数值记忆的两组解释成只有语义加分不同。

`--model-inference on|off` 是独立的模型调用开关，默认 `on`，对应
`carla.planning.model_inference`。组合如下：

| 语义影响 | 模型调用 | 运行方式 |
| --- | --- | --- |
| on | on | 按 semantic-policy 查询/采纳；shadow 实际不采纳 |
| off | on | 模型仍按认知周期调用，PDM 不接收模型语义分 |
| off | off | 跳过模型，可用 memory_off 纯 PDM，或 memory_on + full 数值记忆 |
| on | off | 不支持，启动前报错：没有模型输出可用于语义加分 |

纯 PDM 模式使用 `--planner pdm_closed --semantic-influence off
--model-inference off --mode memory_off`，此时不读写任何记忆。无模型另可使用
`--mode memory_on --planning-memory full` 启用独立数值经验库；不加载文本/视觉记忆，
不会生成模型决策记录。无模型时，旧 `vlm_trajectory`、语义on或memory_on但未显式选full
会在连接CARLA前报错；memory_off则始终禁读写，允许保留任意合法planning-memory配置值。
该模式不安装六路认知相机、不加载 DINOv2、不创建双 VLM 客户端，不检查模型 API Key。
仍使用 CARLA GT 地图与 actor、相同 PDM 候选生成/基础评分/执行前校验、0.2s 规划及
0.1s 物理控制，保留场景事件、碰撞与路线指标、卡死检测、正常清理和可选驾驶视频。
它不保证整轮立即结束：PDM 计算、CARLA 物理、交通等待和视频录制仍消耗时间。

在现有 `mulmem_carla` 环境快速运行原 R05 场景：

```bash
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1_spawncheck/eval/R05_multilane_eval_s101.yaml \
  --planner pdm_closed \
  --semantic-influence off \
  --model-inference off \
  --mode memory_off
```

恢复模型调用时将 `--model-inference off` 改为 `on` 或省略；模型调用开关关闭时，不可用
该轮结果评价提示词或真实模型语义的作用；数值记忆需单独对照。比较耗时时分别报告模型调用状态；
移除认知相机会改变计算负载，不属于仅改变语义加分的同计算条件对照。

`--mode memory_off` 与语义开关是两个独立实验维度。做对照时固定场景、seed、路线、
规划参数、模型配置及评估库快照；不要先让一个实验组向同一库写入，再让另一个组读取新增内容。
持续积累仍使用 `--memory-bank ... --memory-access read_write`；评估使用只读库。
新旧方法的训练库建议分开保存，旧库不会被自动迁移或覆盖。

### 13.4 输出与验收

每轮 `outputs/carla_runs/<运行目录>/` 中：

- `manifest.json` 保存完整规划配置、语义和模型调用开关、记忆策略及最终路线结果。
  新运行目录含 `policy_selective|always|shadow` 与 `pm_off|gate|full`；manifest 单独记录
  `semantic_policy`、`planning_memory`、实际有效记忆模式以及所选 schema / prompt 版本。
  Windows 输出路径可能使用紧凑目录标签，完整模式信息仍以 manifest 为准；Linux 保留完整可读目录名。
  纯 PDM 运行目录含 `model_off`，报告与 `planner.jsonl` 标记 `model_inference=false`，
  `semantic_decision` 和关联模型帧为 `null`，报告的全量认知间隔为 `null`。
- 决策 JSONL 的 `model_decision_output` 保留模型语义，`decision_output` 保存规划后的指令与轨迹，
  `planning_output` 保存候选诊断，`decision_validation` 保存执行前数值校验结果。
- `planner.jsonl` 记录每次 PDM 更新的仿真时间、关联语义帧、候选基础分/语义分/最终选择和校验结果。
  当前 `selection_rule=pdm_base_minus_numerical_cost_plus_semantic`，逐候选保存原生 `base_score`、
  `numerical_cost`、`planning_score`、`lane_quality` 和语义加分；无成本字段的旧调用仍保留
  `pdm_base_plus_semantic`。`eligible` 只表示数值有效，`safe` / `safety` 仅作
  风险诊断；`selected_zero_base_score` 标记是否选择了 0 基础分候选。选择阶段的必要回归门控已移除，
  `route_recovery_active=false`。`geometry` 保存实际车道连接、截断和变道许可诊断。
  几何失败也会写入 `semantic_error` / `control_error`；无数值有效候选时
  `failure_reason=no_finite_candidate`，兼容的校验原因仍为 `pdm_emergency_brake`。
  详细规则见 [PDM 规划说明](planning/README.md#路线几何与规划失败诊断)。
- `control.jsonl` 将每个物理 tick 的速度/加速度与前一条控制命令对齐，并保存服务器
  帧号/时间、带方向的前向速度及挡位/倒车/手刹状态。`control_context.json` 保存本轮
  实际物理子步配置和车辆制动参数，便于区分规划停车、控制响应和仿真推进异常。
- 记忆中的 `semantic_decision` 是模型建议，`planning_context` 是规划器选择；
  记录阶段为 `selected_before_execution`，不能据此断言已经成功完成该动作。
- `paper_suite summarize` 保存 `planner_backend`、`semantic_influence`、`model_inference`、
  `semantic_policy`、`planning_memory`、顾问查询/采用/拒绝/墙钟耗时和规划配置哈希，方便分组。
  旧记录缺失模型调用开关时按 `true`、缺失策略时按 `always`、缺失数值记忆时按 `off` 解释。
- `advisory.jsonl` 区分查询跳过、模型返回及失败，`planner.jsonl` 记录基础选择、实际选择、
  shadow 假设选择、当前目标证据、建议过期/释放及拒绝原因；`planning_experience_audit.jsonl`
  为数值经验读写/观测审计（具体文件以输出目录为准）。报告的“额外停车归因代理”仅标识语义
  改变候选且实测低速、基础候选仍有前进意图的时间，不是无必要停车真值或真实反事实通行结果。

`advisory.model_queries` 统计完整认知咨询轮数，每轮通常包含场景理解与驾驶决策两次模型请求，
不是底层 HTTP 请求计数；重试也不另算咨询轮数。旧 always 与语义off但模型on的认知周期同样计数、计时。
数值库健康检查与真实配对证据导入见 [执行经验库工具](planning/EXPERIENCE_MEMORY.md)。

验收仍要求**路线完成率 100% 且正常清理退出**，同时审查碰撞、违规和场景事件结果。
本实现使用 CARLA GT 障碍物与地图，PDM 局部预测和 CARLA 实际控制存在模型差异；
不能将通过候选检查解释为真实仿真中一定无碰撞。

定点论文事件生成失败时，本轮无效，不能跳过事件继续计为成功。`events.json` 保存
尝试的蓝图、坐标和 CARLA 原始生成错误。背景行人和车辆的出生点都会避开预留事件位置
8m；这会影响背景采样序列，新旧版本结果不能视为完全相同的交通快照，应在同一版本重跑
语义开/关对照。预留位置不保证没有地图静态障碍物；旧场景 YAML 若仍无法生成事件，
需要在服务器检查固定起点与碰撞几何，不应运行时随机挪动事件位置。

### 13.5 不调用模型的路线故障定位

`check_pdm_route` 使用真实 CARLA 车辆物理、相同 PDM 核心和执行控制器，固定输入
`FOLLOW_ROUTE + MAINTAIN`，不调用 VLM、不加载 DINO、不访问记忆库。
这是工程诊断，输出明确标记 `paper_vlm_experiment=false`，不能代替模型或记忆实验结果。
请使用空闲 CARLA 实例，关闭其它会推进该实例 `world.tick()` 的进程。
多实例时 Traffic Manager 默认使用各自实际 RPC 端口加 6000（例如 2000→8000、
2200→8200），可用 `carla.traffic_manager_port` 显式覆盖；不同实例不能共用一个端口。
绑定或同步初始化失败会明确报错，工具不自动抢占其它进程的端口。

先用原 prepared YAML 排除背景交通因素，再保留原交通数量复查：

```bash
# 空背景交通，基础 PDM 评分。保留原路线、终点、seed 和交通灯。
python -m carla_bridge.tools.check_pdm_route \
  --scenario outputs/paper_v1_spawncheck/eval/R05_multilane_eval_s101.yaml \
  --semantic-influence off --empty-traffic \
  --output-dir outputs/pdm_diagnostics/R05_empty_off

# 原 YAML 背景交通；固定正常行驶语义参与评分。
python -m carla_bridge.tools.check_pdm_route \
  --scenario outputs/paper_v1_spawncheck/eval/R05_multilane_eval_s101.yaml \
  --semantic-influence on \
  --output-dir outputs/pdm_diagnostics/R05_traffic_on
```

`--host` / `--port` 默认 `127.0.0.1:2000`；默认关闭渲染，`--rendering` 可启用。
不传 `--max-duration-s` 时使用 YAML 时限。每轮必须使用新的输出目录，工具拒绝覆盖旧结果。
包含脚本事件的 YAML 会明确报不支持，应使用完整驾驶入口验证，不能遗漏事件后计作成功。
按同样方式组合 empty/原交通 × on/off；这些对照用于定位地图、规划、控制和交通因素。
上面两条命令同时改变了交通和语义开关，不能直接相减来评价语义收益；比较开关时须保持
同一源码、场景、交通配置和参数，并分别运行 on/off。固定意图的实验也不能验证真实模型
是否仍误读 STOP 路面文字，后者要使用完整驾驶入口另行检验。

`report.json` 保存完整原始/有效场景、路线及源码哈希、实际交通出生清单、车辆参数、版本、
实际 Traffic Manager 端口、清理结果；`planner.jsonl` / `control.jsonl` 保存候选与逐 tick 执行。
只有按序完成路线 100%、到达终点、无碰撞、无异常轨迹、无运行错误且正常清理才通过。
运行中修改源码也不会计作通过。不要将“运行若干秒未报错”或绕过导航出口当成修复成功。
这个通过条件只验收路线可达性和上述运行检查，不验收压线、逆行、闯灯、舒适性或驾驶效率。
还应检查实际车辆的速度、加速度、车道侵入及交通违规；候选预测评分与实际执行指标也应分别报告。

此前2026-10-18版本的本地真实物理验证保留原 R05 路线和60车/20行人配置，固定语义的
on/off 两组分别在 316.2s / 313.6s 完成 100%，记录碰撞和轨迹拒绝均为 0。
这不是本轮新selective/full版本的验收，也不包含真实VLM或记忆验证，原始制动仍有明显尖峰；完整条件、结果及限制见
[PDM 修复与物理验证说明](planning/README.md#本地完整路线物理验证2026-10-18)。

### 13.6 选择性语义顾问与数值规划记忆：完整命令

在已有 `mulmem_carla` 环境、项目根目录运行。先完成 12.0 / 12.2 的场景生成与出生检查；
以下路径必须已经存在，不重新创建 conda 环境。评估场景只读，积累使用训练场景。

```bash
conda activate mulmem_carla
SCENE_EVAL=outputs/paper_v1_spawncheck/eval/R05_multilane_eval_s101.yaml
SCENE_TRAIN=outputs/paper_v1_spawncheck/train/R05_multilane_train_s11.yaml
BANK_SHARED=outputs/memory_banks/selective_v1/train_general
BANK_SCENE=outputs/memory_banks/selective_v1/R05_multilane
```

| 目的 | 语义策略 | 数值经验 | 记忆权限 |
| --- | --- | --- | --- |
| 纯 PDM 快速基线 | selective，语义 off、模型 off | off | memory_off，完全禁读写 |
| 旧版语义对照 | always，语义 on、模型 on | off | memory_off |
| 新选择性方法 | selective，语义 on、模型 on | off | memory_off，单独观察语义改动 |
| 仅观察建议 | shadow，模型 on | off | memory_off，实际总执行基础 PDM |
| 辅助查询门控的记忆消融 | selective | gate | 训练 read_write／评估 read_only |
| 完整数值记忆方法 | selective | full | 训练 read_write／评估 read_only |
| 无模型数值记忆 | 语义 off、模型 off | full | memory_on；不初始化文本/视觉记忆 |

参数优先级为配置默认值→显式CLI覆盖→访问权限约束；例如memory_off始终禁读写，
eval+memory_on仍必须read_only。shadow只关闭语义实际采纳，不自动关闭数值记忆：
若启用full，其实际基础选择仍在包含数值参数先验的候选池内进行；严格纯PDM基线应显式关闭记忆。
`semantic-policy` / `planning-memory` 属于PDM适配；旧 `vlm_trajectory` 不使用这些规划模块。

```bash
# 1. 纯 PDM 快速跑完整路线，无模型与记忆
python -m carla_bridge.run_carla_demo --scenario "$SCENE_EVAL" \
  --planner pdm_closed --semantic-policy selective --semantic-influence off \
  --model-inference off --planning-memory off --mode memory_off

# 2. 旧版无条件语义加分对照，保留五字段提示词/失败规则
python -m carla_bridge.run_carla_demo --scenario "$SCENE_EVAL" \
  --planner pdm_closed --semantic-policy always --semantic-influence on \
  --model-inference on --planning-memory off --mode memory_off

# 3. 新选择性顾问：先不使用任何记忆，检验语义策略本身
python -m carla_bridge.run_carla_demo --scenario "$SCENE_EVAL" \
  --planner pdm_closed --semantic-policy selective --semantic-influence on \
  --model-inference on --planning-memory off --mode memory_off

# 4. Shadow：可能按需调用模型并记录假设选择，实际始终执行基础 PDM
python -m carla_bridge.run_carla_demo --scenario "$SCENE_EVAL" \
  --planner pdm_closed --semantic-policy shadow --semantic-influence on \
  --model-inference on --planning-memory off --mode memory_off

# 5. 持续积累共享训练库；后续更换训练场景仍指定同一 BANK_SHARED
python -m carla_bridge.run_carla_demo --scenario "$SCENE_TRAIN" \
  --planner pdm_closed --semantic-policy selective --semantic-influence on \
  --model-inference on --planning-memory full --mode memory_on \
  --memory-bank "$BANK_SHARED" --memory-access read_write

# 6. 指定场景库：独立积累 R05，可与其它场景库分别管理
python -m carla_bridge.run_carla_demo --scenario "$SCENE_TRAIN" \
  --planner pdm_closed --semantic-policy selective --semantic-influence on \
  --model-inference on --planning-memory full --mode memory_on \
  --memory-bank "$BANK_SCENE" --memory-access read_write

# 7. 只读评估：读取训练库，不向库增加或修改经验
python -m carla_bridge.run_carla_demo --scenario "$SCENE_EVAL" \
  --planner pdm_closed --semantic-policy selective --semantic-influence on \
  --model-inference on --planning-memory full --mode memory_on \
  --memory-bank "$BANK_SCENE" --memory-access read_only

# 8. 只积累不检索；no-short-term 同时去掉本轮短期上下文记忆
python -m carla_bridge.run_carla_demo --scenario "$SCENE_TRAIN" \
  --planner pdm_closed --semantic-policy selective --semantic-influence on \
  --model-inference on --planning-memory full --mode memory_on \
  --memory-bank "$BANK_SHARED" --memory-access read_write \
  --no-memory-retrieval --no-short-term

# 9. gate 消融：辅助咨询/介入及失败动作的语义分折扣，不新增数值参数候选
python -m carla_bridge.run_carla_demo --scenario "$SCENE_EVAL" \
  --planner pdm_closed --semantic-policy selective --semantic-influence on \
  --model-inference on --planning-memory gate --mode memory_on \
  --memory-bank "$BANK_SCENE" --memory-access read_only

# 10. 不调用模型，仍使用 full 数值记忆；支持只有 planning_experiences.jsonl 的库
python -m carla_bridge.run_carla_demo --scenario "$SCENE_EVAL" \
  --planner pdm_closed --semantic-policy selective --semantic-influence off \
  --model-inference off --planning-memory full --mode memory_on \
  --memory-bank "$BANK_SHARED" --memory-access read_only
```

首次建立无模型数值库时，将第 10 条的场景换为 `"$SCENE_TRAIN"`、权限换为 `read_write`。
这只建立数值经验文件，不需要 `mid_term_meta.json`、API Key、DINOv2 或文本嵌入模型。
原三层文本/视觉库用于模型推理时仍需要其完整元数据和资产；无模型数值库不能冒充完整文本库。
`--memory-seed` 仍可在新目标库复制已有库，源与目标都加锁，源保持不变。

`memory_off` 优先于 `planning-memory full`，始终不读写经验；`--no-memory-retrieval`
同时关闭持久文本与数值经验检索，但不禁写。`read_only` 运行可在本轮输出目录写审计，
不能改变库文件。相同 bank 仍由独占 OS 锁保护，不并发写入；只读运行也需等现有使用者释放锁。

默认是规则化选择性门控与执行结果库，**尚未训练可学习路由器**。full 数值记忆的参数先验
进入候选后仍用当前 PDM 重新仿真评分；过去安全通过仅表示执行结果，不自动证明语义优于基础 PDM。
Shadow 的假设候选同样不是已执行对照结果。模型调用仍同步冻结仿真；少调用可以节省墙钟时间，
不能宣称已解决云端推理延迟。最终需要同场景/seed/版本/冻结库的独立配对实验，
同时报告 100% 路线、碰撞/违规、速度、静止、制动/jerk、查询次数和两类耗时。

### 13.7 模型请求候选扩展：原生 PDM 持续无法前进时脱困

选择性顾问现在不仅影响已有候选的偏好，也可以在原生规划持续退化时请求增加候选。
系统先验证连续退化证据，再调用模型；模型输出 `planner_degraded` +
`planner_request=expand_candidates`，数值模块生成更多当前车道的位姿回接路径，
与原路径一起重新运行 PDM 原生仿真、评分。没有让模型直接输出数值路点。

已有 `mulmem_carla` 环境、已生成的同一场景直接使用：

```bash
# 允许模型在原生算法持续退化时请求候选扩展
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1_spawncheck/eval/R10_downtown_eval_s101.yaml \
  --planner pdm_closed --semantic-policy selective \
  --semantic-influence on --model-inference on \
  --planner-recovery on --pdm-lateral-control native_lqr \
  --pdm-lateral-preview speed_scaled \
  --planning-memory off --mode memory_off
```

`L10_night_rain_eval_s101.yaml` 或其它已生成场景只需更换 `--scenario`。
`--planner-recovery on` 是允许开关，仍需模型明确请求和当前证据授权，不会启动即自动扩池。
恢复授权不会因单帧候选恢复就立即撤销；要核对持续恢复及实际进展，且始终受TTL和当前交通条件限制。

同场景做以下对照，其它配置、seed保持相同：

| 对照 | 参数修改 |
| --- | --- |
| 允许模型请求候选扩展 | 上述命令 |
| 仅对已有候选施加选择性语义偏好 | 只改 `--planner-recovery off` |
| 纯 PDM 快速基线 | 改为 `--semantic-influence off --model-inference off --planner-recovery off` |
| 观察模型恢复请求、实际执行原池 | 只改 `--semantic-policy shadow` |

上述四组均保持 `--pdm-lateral-control native_lqr --pdm-lateral-preview speed_scaled`，
避免把控制器和预瞄差异混入语义效果。
该开关独立于模型、语义、恢复和记忆开关；纯 PDM 也执行同样的 LQR 转向。
另用 `--pdm-lateral-control pure_pursuit` 可单独对照历史控制方式。
默认配置文件为 `native_lqr + speed_scaled`；自定义旧配置若没有 `planning.lateral_control`，仍兼容旧 Pure Pursuit。
LQR 模式执行所选原生模拟的下一时刻转角，每0.2秒更新，不是模型输出转角；
车辆物理、纵向 PID 和原始加速度/jerk 指标均未修改。
`speed_scaled` 低速保留两秒预瞄帮助起步，车速上升后逐步缩短至一秒，减少后续弯道跟踪偏移。
调度由实测速度、轴距确定，同一帧所有候选共用；模型不输出预瞄参数。
另用 `--pdm-lateral-preview fixed` 可单独对照固定两秒预瞄；旧自定义配置省略模式键仍为fixed。
具体原因、执行审计和验证边界见[横向执行与原生评分的一致性](planning/README.md#横向执行与原生评分的一致性)。

需要记忆时可沿用13.6的 `planning-memory`、bank和权限组合；
`memory_off` 仍完全禁读写，评估 `memory_on` 仍使用 `read_only`。
运行报告和CSV新增扩池、实际选中恢复候选、扩池但无正分前进候选的周期数。
这些数量不代表脱困成功，验收仍看实际路线100%、停车恢复、碰撞违规和原始运动指标。
恢复路径限制、协议、原池回退与逐帧字段详见
[规划模块说明](planning/README.md#原生规划持续退化时由模型请求扩展候选)。

### 13.8 车道居中、合法绕行与常规等待

这次修复覆盖数值候选和模型查询门控，并非只修改提示词。原生可驾驶道路评分不能等同于
车道居中，车道内偏移候选可能近似同分；变道过渡在每次重规划时从当前位置重新开始，
可能一直没有及时离开被挡住的车道；等待门控之前又漏掉单辆前车等红灯和低速移动跟车，
并每15秒复查模型。这些均可在真值数据准确时出现。

| 功能 | 当前行为 | 独立对照 |
| --- | --- | --- |
| 车道位置 | 对未来0.1–2秒实际预测车身相对地图中心/边界计有界软成本，优先区分基础分接近的候选 | `--lane-quality off` |
| 持续绕行 | 对持续慢速障碍提出固定地图锚点变道，实际选中后持续执行，超越并满足间距后合法返回 | `--lane-maneuver off` |
| 常规等待 | selective/shadow中，有当前信号依据的等待、单辆前车队列及正常移动跟车不周期咨询模型 | 检查 `advisory.jsonl` 的跳过原因与模型调用数 |

`--lane-quality`、`--lane-maneuver` 均默认 `on`，独立于语义、模型、恢复和记忆开关。
纯PDM也使用相同数值机制。公式为 `final_score = base_score - numerical_cost + semantic_bonus`；
原生 `base_score` 不改，数值成本总额不超过0.16，语义关闭时 `semantic_bonus=0`。
默认中心/边界成本权重0.04/0.12，普通新变道另有0.02启动成本，也受0.16总上限限制。
这不是把出界恢复成硬淘汰，也不保证所有实际轨迹都不压线；仍需审查实际车身与标线的关系。
合法变道的中心成本跟随连续机动参考，边界成本则检查源/目标真实车道并集的外边界；
不会用随参考移动的窄管道把合法跨虚线变道当成出界。
另一个已确认问题是地图车道表面的数值微缝：Town10HD两条合法相邻车道的真实共享线处，
直接并集后测到的边界距离仅0.52–2.96微米，软成本误把内部虚线当作外沿。
现在仅在软边界参考中作1毫米拓扑闭合，恢复到约3.5米外的真实外沿；
不修改原生PDM道路/可驾驶区域或基础分，也不因此放宽实线变道许可。

绕行必须有实际连通的同向邻道、标线与方向性许可，以及足够前后间距和合法返回空间。
普通变道和持续绕行共享标线许可检查，不能因CARLA宽泛的变道标志而跨越实线。
横穿行人、人行横道附近行人、红灯队列不作为普通绕行目标；无合法候选时允许跟车/等待。
新路径仍经过同一PDM生成、模拟和评分，提出候选不代表强制执行。
只有新增绕行路径使用带曲率限速的纵向策略副本，普通候选保留原有速度与IDM参数；
全池统一生成、模拟并评分。返回的首个合法空间锚点会缓存，避免起点随重规划前移。
越障后通过间距核验的返回提议，普通零偏移目标候选和专属绕行候选共用同一核验空间路径，
分别保留普通速度策略（默认最高8m/s）和专属限速策略（默认上限5m/s）。常速占据检查同时
覆盖限速/实测速度与普通最高速度两种到达时序，实际获选者才提交返回状态。
没有持续障碍时，实际获选的普通变道同样保存目标车道和空间路径；跨越中线后继续保留
原变道语义，避免被重新生成的单车道居中偏好拉回。到达实际目标车道且中心误差小于0.35米、
朝向误差小于0.15rad才确认完成，缓存最长20秒；TTL、时间/定位跳变或目标身份变化会清除。
该机制由 `--lane-maneuver` 一并控制，不强制选中返回或把预测结果记作已执行。
一次返道物理诊断对应的预测左前角曾越过对向实线约17厘米，原因是返回曲线的完整车身扫掠，
并非GT位置错误。现增加受实际车宽/车道余量约束的目标车道内±0.25/±0.5米候选，
每次都基于固定名义返回路径，不叠加上一轮偏移；保留原边界成本及权重，重新进行全池原生评分。
实际获选的偏移路径才更新机动，普通/专属候选分别记录真实选择来源、长度与几何哈希。
规划尾段沿真实地图后继扩展，评估路线和终点保持不变。always与其它模式共享当前信号/队列
上下文来判断绕行资格；这不改变always原来的周期模型调用。

在项目根目录、已有 `mulmem_carla` 环境中，以同一已生成场景作对照：

```bash
SCENE=outputs/paper_v1_spawncheck/eval/R05_multilane_eval_s101.yaml

# 数值规划基线：不开模型，两个修复均启用
python -m carla_bridge.run_carla_demo --scenario "$SCENE" \
  --planner pdm_closed --semantic-influence off --model-inference off \
  --planning-memory off --mode memory_off --lane-quality on --lane-maneuver on

# 选择性语义：数值层保持一致；正常红灯/队列等待不周期查询模型
python -m carla_bridge.run_carla_demo --scenario "$SCENE" \
  --planner pdm_closed --semantic-policy selective \
  --semantic-influence on --model-inference on \
  --planning-memory off --mode memory_off --lane-quality on --lane-maneuver on
```

单独评估车道成本时只改 `--lane-quality off`，评估持续绕行时只改 `--lane-maneuver off`；
两个都关用于旧数值机制对照，控制器、预瞄、种子、交通和记忆快照保持一致。
数值配置为 `carla.planning.lane_quality` / `carla.planning.lane_maneuver`；
机动阶段见 `planner.jsonl` 每条记录的 `lane_maneuver` / `ordinary_lane_intent`，逐候选成本位于
`candidates[*].lane_quality`，不是将模型的理由文字作为数值评分。
manifest、outcome、报告和汇总CSV同时记录 `lane_quality_enabled` / `lane_maneuver_enabled`；
CSV的 `execution_planner_version` 用于区分数值配置版本，研究对照应一并保存。

正常等待跳过原因为 `expected_signal_or_queue_wait`，正常移动跟车为
`ordinary_same_lane_following`。仅等得久或建议TTL到期不再触发咨询；
绿灯后仍异常堵塞、真实目标关系改变、对象侵入自车当前占据空间等情况会重新评估。
`wait_review_interval_s` 为旧配置兼容保留，不再每15秒复核。
该节的节流适用于语义on的selective/shadow；旧always和语义off但模型on仍按原认知周期调用。

当前经验版本为 `carla_pdm_execution_v4:<配置哈希>`，规范化后的两个数值模块配置纳入哈希。
旧版本经验保持原样但不跨版本检索，当前版本见 `manifest.json.outcome.execution_planner_version`；
库检查/导入应显式指定该完整版本，见[经验库工具说明](planning/EXPERIENCE_MEMORY.md)。

2026-10-19冻结代码后的真实CARLA专项结果，时长均为仿真时间：

| 专项 | 结果 |
| --- | --- |
| 居中 | 初始偏差0.45米，80米100%/14.9秒；5秒后最大中心误差0.3563米、最小车身余量0.2571米 |
| 车辆绕行 | 80米100%/14.8秒，完成超越并返回原车道；末态中心误差−0.1002米、车身余量+0.4706米 |
| 同车道行人绕行 | 行人真实速度0.5m/s，80米100%/14.6秒，超越并返回原车道；末态中心误差−0.1848米、实际车身余量+0.1972米，无模型调用 |
| 红灯单前车队列 | 20秒内20次查询检查均正常等待，模型请求0、绕行执行0，最终静止；模型/语义开关均on |

四项碰撞均为0、源码未改变、清理后无遗留测试actor。永久红灯是等待专项，不以路线完成计成功。
这些是局部工程检查，没有真实VLM，不能替代完整论文路线及动态交通验收。
原始证据位置见[规划模块说明](planning/README.md#车道位置软成本与持续绕行)。
纵向急制动与jerk仍是独立问题，局部到达终点和零碰撞不能证明舒适度已经解决。

### 13.9 R05变道期间的 `nonzero_origin` 退出

2026-10-20服务器运行在路线8.98%以 `invalid_trajectory` 结束。97次有效规划记录中，
6次被拒绝均为 `nonzero_origin`：生成轨迹的t=0位置距实测车身原点0.251–0.469米，
最后连续3次拒绝触发终止。39次顾问检查全部跳过，没有模型调用，也没有模型语义停车指令。

原因是适配器混用了两个空间轨迹：PDM生成器将t=0投影到固定变道参考路径上，
原生模拟器则从实际车辆状态开始跟踪并评分。持续变道不再每轮重置路径后，
真实车身与理想参考存在正常跟踪偏差，返回生成器参考就会触发0.25米起点检查。
先前短程专项未覆盖此条件，不能据此认为完整R05路线已验收。

现改为返回实际获选、实际评分的原生模拟轨迹，朝向也取同一rollout。
从后轴到捕获时刻车身原点使用同一个刚体变换，应用于全部未来点；不单独把首点清零，
不平移理想路径，也不放宽校验阈值。若rollout首状态与当前实测状态不一致仍明确报错。
IDM速度目标、纵向PID、PDM基础评分公式、候选选择和原生转向样本保持原有机制。

真实背景交通复测还暴露原生bicycle积分没有前进挡零速边界：实测约−23.49m/s²的制动
进入加速度滤波器后，预测速度会穿过零点并倒退，造成 `direction_reversal`。
现在所有候选统一使用前进挡积分：普通前进步与原生结果逐元素一致，仅在速度即将变负的
一步积分到实际停车点，剩余时间保持零速；停车后可再次起步。不裁剪已选轨迹，
不删除候选或恢复道路/碰撞硬否决。评分仍重新使用各候选的完整预测；
因此发生过零修正的候选分数可能随预测改善而变化，不能声称全部基础分数仍逐位相同。

`planner.jsonl` 新增 `planner_diagnostics.execution_trajectory`，包括
`geometry_source=native_pdm_scored_rollout`、`speed_profile_source=native_pdm_idm_target_velocity`、
`reference_origin_distance_m`、`execution_origin_distance_m` 和所选rollout哈希。
这一区别用于审计“参考路径偏差”和“实际返回轨迹是否正确锚定”，不能将参考路径偏差直接
当作车辆越界。此变更对纯PDM、选择性语义和shadow，以及两种横向控制模式统一生效。
数值执行经验版本随之升为v4，旧经验库保留但不跨版本检索。
`rollout_motion_model=forward_gear_native_bicycle_zero_crossing_stop` 标识上述数值车辆模型。

### 13.10 连续路口之间的短距离合法返回

修正起点和前进挡积分后的完整R05工程复测，在400秒内完成98.01%，2000次规划均通过轨迹
校验、碰撞0，但仍未达到100%验收。原始记录保留在
`outputs/diagnostics/r05_origin_full_route_final/`，该轮应计为失败，不能用零拒绝代替路线完成。

这里是另一个候选构造缺陷：自车在`27:0:1`左车道，参考路线从`27:0:2`右转进入`2353`。
原先固定30米的返回区间跨过前一个直行连接`2083`，因此整条返回方案被判为不合法；
随后IDM把被截断的左车道末端当成静止前车，只能不断蠕动或停车。实际地图在两个路口之间
仍有合法虚线区间，留出原有端点权限和出口余量后，可构造约16米的连续返回。

现在先完整保留已经合法的固定长度返回几何；仅在原返回穿越禁止区间时，搜索实际连续合法的
区间。在配置的返回长度以内，可缩短下限取两秒当前行驶距离、三倍车道宽度和两倍轴距的最大值，仍保留端点周围2米
的权限检查与出口前4米余量。空间不足时保留有限停车走廊，不能跨实线或穿越路口拼接变道。
这只是补足可供PDM评分的真实路径，没有加入强制返回参考车道的选择规则。

`planner_diagnostics.geometry.lateral_paths.*.route_connection`记录实际与配置返回长度、
最短长度及`legal_return_lane_keys`，可区分正常完整路径、合法短区间和确实无法返回的情况。
真实地图回归覆盖可完成的短区间、车速较高时不能强行使用的短区间，以及原有合法30米返回。

2026-10-20最终冻结代码的验证结果：

| 检查 | 结果 |
| --- | --- |
| 场景与地图 | 与服务器相同的R05配置、seed 101、Town05 OpenDRIVE哈希；922.087米原路线、60辆车和20名行人 |
| 完整物理路线 | 原生LQR横向及原速度曲线PID，100%按序完成并到达终点，311.1秒 |
| 规划与碰撞 | 1556次规划，轨迹拒绝0、碰撞0，运行期间源码未改变 |
| 返回动作 | 304.8秒实际回到`27:0:2`参考车道，随后经过`2353`出口连接 |
| 起点审计 | 33次理想参考起点偏差超过0.25米；实际返回轨迹起点帧误差最大约2.22e−16米，校验阈值仍为0.25米 |
| 清理 | 车辆、行人、行人控制器和碰撞传感器残留均0；服务器设置恢复，独立测试进程已关闭 |
| 回归 | 1735项通过，1项因Windows未启用符号链接而跳过 |

上述回归计数由全量测试和对应文件复核去重汇总：全量中有1个测试替身未接受新增的最短返回
长度参数，更新替身接口后，该文件25项全部通过，原断言未放宽。全量与复核XML均保留。
两次本地完整复测的实测状态在297.1秒才首次出现差异；前一轮98.01%的失败记录保留，
后一次通过并不替换前一次失败。两轮物理测试之间仅规划器候选构造源码改变，路线及交通配置相同。

最终完整报告与精简审计分别为`outputs/diagnostics/r05_complete_all_fixes/report.json`和
`verification_summary.json`，清理的独立观测为同目录`cleanup_observation.json`。
全量测试位于`outputs/selective_validation/r05_all_fixes_full.xml`，对应文件复核位于
`r05_geometry_fixture_recheck.xml`。

这是本地Windows CARLA 0.9.15的纯PDM工程回归，语义影响关闭，无VLM、DINO或记忆访问。
它确认本轮退出与路线停滞缺陷的修复，不能替代服务器选择性语义模式的复测，也不代表所有
场景、车道边界合规、交通规则及舒适性指标已经全部验收。

### 13.11 模型调用审核、视频定位与仿真轨迹绘制

本轮运行可自动生成一个离线审核页面，用于回答“哪一帧、什么位置、为什么调用模型、
看到了什么、模型输出了什么，以及建议是否改变了轨迹选择”。页面将调用列表、触发类型统计、
自车位置轨迹、原始输入图片、逐阶段请求详情和驾驶视频放在一起；点击调用记录可定位视频。
不调用模型的纯PDM运行也能生成页面，并在审计完整时显示0次调用。

在已有 `mulmem_carla` 环境、仓库根目录运行：

```bash
conda activate mulmem_carla
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1_spawncheck/eval/R05_multilane_eval_s101.yaml \
  --planner pdm_closed \
  --semantic-policy selective \
  --semantic-influence on \
  --model-inference on \
  --planning-memory off \
  --mode memory_off \
  --replay-audit on \
  --record-video on \
  --trajectory-visualization off
```

| 开关 | 用途与当前默认值 |
| --- | --- |
| `--replay-audit on/off` | 模型调用证据归档和审核页面；默认on |
| `--record-video on/off` | 驾驶视频和逐帧索引；当前 `carla_bridge/config/carla.yaml` 为on |
| `--trajectory-visualization on/off` | 在CARLA世界绘制候选及执行计划；默认off，使用PDM后端时生效 |

命令行优先于YAML。上面示例用于记录模型运行；需要观察世界轨迹时，将最后一项改为
`--trajectory-visualization on`。观测开关不修改数值规划、候选评分或语义选择规则；
但世界线会进入RGB相机，模型输入可能随之变化，因此正式模型对照实验保持世界绘制off，
只在可视化诊断时开启。归档、视频和绘制有I/O、编码及渲染开销，比较运行耗时时应保持观测配置一致。

#### 找到并打开本轮审核页面

启动后控制台打印 `[carla_bridge] 本轮输出目录: ...`。以下 `<RUN_DIR>` 指该完整目录，
例如 `outputs/carla_runs/20261020_...__R05_multilane_eval_s101__...`，不是整个 `outputs/`。
正常退出或由程序处理的中断进入清理阶段时，会生成：

```text
<RUN_DIR>/
  replay/
    index.html                   # 离线审核入口
    data.json                    # 页面使用的完整审核数据
    summary.json                 # 调用比例、阶段次数、触发类型与完整性提示
  model_calls.jsonl              # 一次咨询的开始、完成或失败事件（有咨询时生成）
  model_requests.jsonl           # 实际SDK dispatch及结果、失败与重试（有请求时生成）
  model_call_assets/             # 实际模型输入图片的运行内归档，含用到的历史记忆图片
  carla_videos/
    drive_....mp4
    drive_....frames.jsonl        # 视频帧号与实际CARLA帧、仿真时间对应关系
    drive_....events.jsonl        # 缺帧、过期或编码失败原因
    drive_....summary.json        # 视频帧数、缺失及最终补写统计
  control.jsonl
  planner.jsonl
```

建议将**整个本轮目录**下载到本机，再直接打开 `replay/index.html`。
页面内嵌审核数据，不需要CDN或在线服务；图片和视频通过相对路径加载，
只复制 `index.html` 会丢失这些素材。被实际送入模型的历史图片已归档到本轮目录，
无需额外复制原记忆库。

若浏览器限制本地文件或视频加载，在该目录所在机器启动静态服务：

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory "<RUN_DIR>"
```

在同一台机器浏览器打开 `http://127.0.0.1:8765/replay/`。若素材保留在服务器，
在服务器执行上面命令，再在本机新终端建立隧道：

```bash
ssh -N -L 8765:127.0.0.1:8765 zjj@<服务器地址>
```

本机浏览器访问相同地址即可。若MP4编码不被浏览器支持，可以用本机视频播放器查看原视频，
同时根据页面帧号和 `*.frames.jsonl` 对照；不要丢弃帧索引。

旧运行日志可在仓库中手动补建或重建页面：

```bash
python -m carla_bridge.replay.review "<RUN_DIR>"
```

补建不会调用模型，也不会改写驾驶日志。旧日志没有归档的完整prompt、输入图片、原始响应、
实际SDK请求和重试次数无法从结果反推；页面会显示未知或数据缺失，不能当作新审计的同等证据。
未经过清理的进程强制终止、磁盘写入失败等情况，也应查看完整性提示后再使用统计。

#### 人工观测与统计口径

先在页面上查看总调用帧占比与触发类型，再通过表格筛选、上一次/下一次调用按钮或时间线定位。
每次咨询详情显示仿真时间、真实CARLA帧、自车位置与速度、触发原因及实际使用的图片；
逐阶段展开可查看scene/decision的system prompt、完整用户prompt、发送给SDK的messages、
原始响应、请求参数、失败原因和各次重试。图片可能同时包括本帧视角图、拼接图与检索到的历史图，
应按“本次请求实际使用的全部图片”核对输入，而不只看前视图。

**实际调用帧占比**的分子是发生实际SDK dispatch的唯一CARLA帧数，分母是
`control.jsonl` 中 `world_snapshot.frame` 去重后的真实推进帧数。
同一帧的scene、decision及应用层重试只增加请求次数，不重复增加调用帧数。
例如1000个推进帧中有10个帧发生请求，比例为1%，即使这10帧合计发出了25次SDK请求。
请求计数覆盖应用可观察的SDK dispatch，不宣称覆盖SDK内部自动HTTP重试。
审计缺失或帧无法对齐时，页面显示未知及原因，不用0填充缺失值。

“咨询”“实际调用”“语义被采纳”是不同事件：顾问检查可能跳过；进入咨询后也可能在请求前失败；
模型有响应也可能未改变PDM的候选选择。页面分别统计SDK尝试、语义改变选择的规划次数和
可归因控制帧占比。语义改变选择以同一候选池的无语义基线及扩展候选审计为依据，
控制帧通过 `command.planner_sequence` 关联已接受计划；这不等于证明实际驾驶效果得到改善。

视频按仿真推进录制，不包含模型思考期间CARLA冻结的墙钟等待。
GPU相机回调可以迟到，录制器使用有界待匹配队列保留原帧时间与HUD，
按实际CARLA帧补写，不用相邻旧图冒充。最终缺帧会压缩视频时长，
因此**不能用“仿真时间×fps”直接推算视频帧**，应使用 `*.frames.jsonl`。
页面在调用帧存在时精确定位；图像缺失只能定位最近可用帧时，会提示近似映射。
视频HUD显示的是该帧采样时最近已开始的咨询；调用可能在采样后才开始，HUD会在后续tick更新，
精确触发瞬间以HTML审核中的实际请求帧及索引映射为准。

#### 在仿真环境观察候选和执行计划

启用 `--trajectory-visualization on` 后，按每次重规划绘制：

| 颜色 | 含义 |
| --- | --- |
| 蓝色 | 参与评分的候选轨迹 |
| 橙色 | 本次获选候选 |
| 绿色 | 通过轨迹校验、下发给控制器的执行计划 |
| 红色 | 被轨迹校验拒绝的计划 |

绿色线表示未来执行计划，不表示车辆已经实际走过的轨迹。
候选线、获选线和执行线可能重叠，最后绘制的线会遮住前面的部分。
世界线使用CARLA debug绘制。本机CARLA 0.9.15验证中已在RGB录像里看到世界线，
因此同时开启认知相机时，模型也可能看到这些标记。其他渲染模式的可见性仍需实际核对；
回放以保存的实际输入图片、视频和规划日志为依据。

配置位于 `carla_bridge/config/carla.yaml` 的 `carla.visualization.trajectories`：

```yaml
carla:
  visualization:
    trajectories:
      enabled: false
      max_candidates: 0      # 0画全部；正整数限制显示数量，不影响候选计算或选择
      point_stride: 3        # 绘制点抽样间隔
      lifetime_s: 0.25       # 短暂保留，随重规划刷新
      thickness: 0.06
      z_offset: 0.15         # 米；抬高以减少与路面重叠
```

全部候选绘制有性能成本；观察时可先开启，正式批量实验按需要关闭或限制显示数量。

本轮验证记录（`outputs/replay_validation/`，本地生成、不随Git分发）：

| 验证 | 结果与范围 |
| --- | --- |
| 相关回归 | 259项通过，覆盖调用归档、重试与中断、统计完整性、视频索引、规划及控制回归；`final_all.xml` |
| 页面交互逻辑 | 上述测试中调用Node脚本，14项用例通过，覆盖播放跟随、跳转、缺帧提示和无视频情况；不等于浏览器人工视觉验收 |
| 实际CARLA短程验证 | 本地0.9.15、Town05、60辆背景车与20名行人，最终样式运行4秒、40个推进帧、20次PDM规划，轨迹拒绝0次；已检查录像中的轨迹绘制和HUD |
| 视频缺帧验证 | 该短程最终编码24帧、缺失16帧、编码错误0；缺失显式记录，页面按真实帧索引定位；输入图归档独立于录像 |
| 零调用统计与清理 | 纯PDM显示0/40调用帧；结束后车辆、行人、行人控制器、传感器均无残留，同步设置恢复，独立测试服务已关闭 |

短程原始证据在 `outputs/replay_validation/carla_final_style_probe/`，清理独立观测为
`cleanup_observation.json`。在线模型输入输出通过模拟SDK的集成测试核对，
本轮没有在本地发起远程VLM请求，也没有重新跑完整路线验收；不沿用13.10节完整路线的通过结果。
本轮改动不会自动commit或推送，需明确授权后提交。

### 13.12 运行开关总表与记忆使用形式

以下覆盖 `python -m carla_bridge.run_carla_demo` 的全部命令行参数。
默认值按当前仓库的 `carla_bridge/config/carla.yaml` 与入口参数定义列出；
`--help` 中部分默认值为None，表示继承YAML，不表示关闭。显式CLI覆盖配置，
随后受后端兼容、记忆权限及论文train/eval约束限制。

#### 三个观测开关的区别

| 参数 | 开启后的行为 | 关闭后的行为 | 当前默认 |
| --- | --- | --- | --- |
| `--replay-audit on/off` | 归档实际模型输入图片、完整提示词、响应、触发原因、时间/帧/位置；运行结束生成 `replay/index.html`，联动视频并统计使用比例 | 不启用新增调用证据归档和自动审核页面；原有驾驶/规划日志仍保留 | on |
| `--record-video on/off` | 增加录像相机，输出驾驶MP4、逐帧索引、缺帧事件与HUD | 不初始化录像相机或录制MP4；调用审核仍可查看图片和文本 | on |
| `--trajectory-visualization on/off` | 用CARLA世界debug线绘制实际评分候选、获选候选、通过验证的执行计划或被拒绝计划 | 不绘制世界线；仍正常规划与保存规划日志 | off |

三个开关独立：审计不等于录像，录像不等于世界轨迹绘制，关闭审计也不等于关闭模型。
模型是否调用由 `--model-inference` 等规划开关决定。调用帧占比为“实际SDK请求的唯一CARLA帧数 / 全场景真实推进的唯一CARLA帧数”，
同一帧的场景理解、决策和重试不重复增加分子。视频缺帧按索引明确提示，输入图片独立保存。
世界线可能进入模型RGB输入，正式VLM对照实验推荐 `on / on / off`，人工观察候选轨迹时再把第三项设为on。

#### 场景、模型与规划开关

| 参数 | 可选值 / 当前默认 | 含义 |
| --- | --- | --- |
| `--scenario` | 场景名或YAML路径；默认 `straight_traffic` | 选择运行场景；论文场景须已完成准备/出生检查 |
| `--planner` | `pdm_closed`（默认）、`vlm_trajectory` | 数值PDM规划，或旧模型直接输出路点后端 |
| `--model-inference` | `on`（默认）、`off` | 是否初始化并调用场景理解/决策模型；off跳过认知相机、DINO和文本/视觉记忆 |
| `--semantic-influence` | `on`（默认）、`off` | 模型横纵向语义是否允许影响PDM选择；off本身不会关闭模型调用 |
| `--semantic-policy` | `selective`（默认）、`always`、`shadow` | 条件查询和介入；旧版周期查询/无条件语义加分；按需查询并记录假设选择但不实际采纳 |
| `--planner-recovery` | `on`（默认）、`off` | selective/shadow且语义on时，是否允许模型明确请求扩展脱困候选；shadow仅观察，不执行假设扩展 |
| `--pdm-lateral-control` | `native_lqr`（默认）、`pure_pursuit` | PDM横向执行控制；纵向均为PID |
| `--pdm-lateral-preview` | `speed_scaled`（默认）、`fixed` | LQR预瞄随实测速度调度，或固定预瞄对照；pure_pursuit不使用该LQR预瞄 |
| `--lane-quality` | `on`（默认）、`off` | 车道居中和车身边界的数值软成本，不是额外硬筛规则 |
| `--lane-maneuver` | `on`（默认）、`off` | 为持续慢障碍构造合法绕行/变道及返回候选，不保证一定选择绕行 |

`--model-inference off` 只允许PDM后端且语义off。此时使用 `memory_off`，或使用
`memory_on + planning-memory full` 的独立数值记忆；不能使用无模型的gate/text-only组合。
selective/shadow只有语义on时走条件咨询；语义off但模型on仍保留周期认知。
shadow只关闭语义采纳，不自动关闭full数值记忆。严格无模型、无记忆PDM基线：

```bash
python -m carla_bridge.run_carla_demo \
  --scenario outputs/paper_v1_spawncheck/eval/R05_multilane_eval_s101.yaml \
  --planner pdm_closed --semantic-influence off --model-inference off \
  --planning-memory off --mode memory_off \
  --replay-audit on --record-video on --trajectory-visualization off
```

#### 记忆开关及使用形式

| 参数 | 可选值 / 当前默认 | 含义 |
| --- | --- | --- |
| `--mode` | `memory_on`（默认）、`memory_off` | 总记忆开关；off关闭所有记忆读写和短期记忆，不关闭模型 |
| `--memory-bank` | 路径；默认未指定 | 指定可跨运行复用的库；省略则使用本轮目录下独立库，不会自动跨轮积累 |
| `--memory-access` | `read_write`（默认）、`read_only` | 允许积累或只读；只读必须指定已有、完整的bank |
| `--memory-seed` | 已有库路径；默认未指定 | 将已有库复制到尚不存在的新目标库后运行；源库不变，仅支持可写目标 |
| `--no-memory-retrieval` | 布尔旗标，默认不传 | 禁止持久文本及数值经验检索，但不禁止写入；不自动关闭本轮短期记忆 |
| `--no-short-term` | 布尔旗标，默认不传 | 禁止本轮短期上下文记忆；不关闭持久库检索/积累 |
| `--planning-memory` | `full`（当前YAML默认）、`gate`、`off` | full在辅助介入判断之外提供数值候选参数先验；gate仅提供介入/语义折扣证据；off关闭数值规划经验，仍可保留模型文本/视觉记忆 |

以下均为追加到同一运行命令的参数组合；需要模型记忆时保留 `--model-inference on`。

| 使用目的 | 记忆参数组合 |
| --- | --- |
| 完全无记忆 | `--mode memory_off --planning-memory off`，不要传bank/seed |
| 每轮独立积累 | `--mode memory_on --memory-access read_write`，不传bank |
| 持续积累共享库 | `--mode memory_on --memory-bank outputs/memory_banks/shared --memory-access read_write`；各轮复用同一路径 |
| 按场景独立积累 | 同上，将bank改为 `outputs/memory_banks/R05` 等不同路径；目录名只用于隔离，不自动筛选场景 |
| 指定库只检索、不增加记忆 | `--mode memory_on --memory-bank outputs/memory_banks/R05 --memory-access read_only` |
| 只积累、不检索持久记忆 | 可写bank组合追加 `--no-memory-retrieval`；连短期也不使用时再加 `--no-short-term` |
| 仅短期记忆 | `--mode memory_on --memory-access read_only --memory-bank <已有库> --no-memory-retrieval --planning-memory off` |
| 仅持久库、无短期 | 只读或可写bank组合追加 `--no-short-term` |
| 从已有库复制后独立改进 | `--mode memory_on --memory-seed <已有源库> --memory-bank <尚不存在的新库> --memory-access read_write` |
| 仅模型文本/视觉记忆 | 模型on、memory_on及bank权限组合，显式 `--planning-memory off` |
| 无模型、仅数值经验 | `--planner pdm_closed --semantic-influence off --model-inference off --planning-memory full --mode memory_on`，再选择bank和读写权限 |

`read_only` 禁止向持久库新增/修改记忆，但允许本轮短期上下文演化和运行审计落盘；
需要把短期也关闭时加 `--no-short-term`。论文 `paper_v1` 的eval场景在memory_on时强制read_only，
积累必须选择train场景。只读必须先有对应格式的训练库；无模型数值库不能用于初始化完整文本/视觉库。
完整的train/eval与多种记忆命令示例见13.6节。

#### 模型文件与帮助参数

| 参数 | 当前默认 | 含义 |
| --- | --- | --- |
| `--dinov2-model-dir` | 未指定 | 指定本地DINOv2模型或Hugging Face snapshot目录 |
| `--dinov2-cache-dir` | `.cache/huggingface` | 指定Hugging Face缓存根目录 |
| `--allow-model-download` | 默认不传，仅本地加载 | 显式允许DINOv2联网查找/下载；不控制远程VLM API调用 |
| `-h` / `--help` | 查看后退出 | 打印当前入口支持的全部命令行参数，不运行场景 |

连接地址/端口、仿真步长、规划频率/速度、视频视角/尺寸、轨迹绘制数量等数值项在
`carla_bridge/config/carla.yaml` 中配置，没有对应CLI参数。
该文件中的额外功能布尔项包括 `carla.synchronous`（同步推进）、
`carla.stuck_detection.enabled`（停滞检查）、`carla.route_injection.enabled`（导航参考注入）、
`perception.oracle_objects`（GT对象输入）、`perception.mosaic.label_subimages`（拼图视角标签）、
`carla.video.hud`（录像HUD）和 `fallback.allow_rule_fallback_for_format_error`（格式错误规则回退）。
它们属于实验配置，修改后应保留配置快照；上述CLI总表不意味着这些配置项可直接写成命令行参数。
