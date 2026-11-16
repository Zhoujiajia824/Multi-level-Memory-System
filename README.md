# EP-Driver 运行指令与参数

> 当前实现边界：正常行驶候选除软跟踪代价外，仍需满足窄跟踪带硬资格（允许误差为 `0.25m + 初始横向偏差 × exp(-路径进度/15m)`）；制动候选不使用此项硬资格。尚未实现“车道中心仅作为软偏好”的后续调整。CARLA物理仿真效果仍需服务器验证。

当前工作分支为 `EP-Driver-CARLA`，由 `EP-Driver` 创建。下文高速环境的克隆示例使用 `EP-Driver`；运行CARLA请使用远程同名分支 `EP-Driver-CARLA` 的独立工作区。配置和命令中的相对路径均以项目根目录为准。

## 0. 克隆分支、权重与个人配置

先安装 Git LFS，并确保 GitHub SSH 认证已配置：

```bash
git lfs install
git clone --branch EP-Driver git@github.com:Zhoujiajia824/Multi-level-Memory-System.git EP-Driver
cd EP-Driver
git lfs pull
git lfs ls-files
```

`models/dinov2-base/` 保存本项目使用的 DINOv2 Base 权重和模型配置。执行 `git lfs pull` 后应取得实际模型文件，不能仅留下 Git LFS 文本指针；检索和写入记忆默认使用该本地目录，无需再次从模型站下载。其他已部署的快模型与反思模型通过服务调用，不包含在此权重目录内。

首次克隆后创建个人模型配置。已有 `config/models.yaml` 时跳过此步骤，避免覆盖个人设置。

Git Bash / Linux：

```bash
cp -n config/models.example.yaml config/models.yaml
```

PowerShell：

```powershell
if (-not (Test-Path -LiteralPath 'config/models.yaml')) {
    Copy-Item -LiteralPath 'config/models.example.yaml' -Destination 'config/models.yaml'
}
```

模板中的两个模型均为 `"off"`。`config/models.yaml`、测试、研究文档、验证脚本、记忆库及运行输出不上传；每台机器的模型地址和个人配置自行维护。已发布入口为 `run_highway.py`、`check_environment.py`、`memory_admin.py`；本分支另新增 `run_carla.py`。

## 1. 环境

已有环境：

```bash
conda activate mulmem_highway
python -m pip install -r requirements.txt -c config/verified_dependencies.txt
python scripts/check_environment.py
```

新建环境：

```bash
conda env create -f environment.yml
conda activate nine_grid_highway
python -m pip install -r requirements.txt -c config/verified_dependencies.txt
python scripts/check_environment.py
```

`config/verified_dependencies.txt` 固定本机验证使用的核心依赖版本；本机实测 Python 3.12.13、CPU 推理。CUDA 服务器的驱动与 PyTorch 安装需单独验证。

## 2. 四场景：纯 MPC
### 当前基础规划：五次参考与安全优先MPC

参数版本 `development-1.1.0-safe-tracking`。常规保持车道中心；基础规划可以在合法、安全且有收益时左侧超车，右换道只主动用于公开导航需要。有效模型单目标决定五次参考的经过位置；不改变原帧Frenet语义，不吸附目标，不增加到达时间或停车要求。新目标立即替换旧目标，纵向经过后释放，实际状态再平滑归中。

先从当前场景筛选，再运行动态EKF：常规前30米、后10米、横向25米，八向每格最近一个、同ID去重。前向范围按 `v*reaction_s + v*v/(2*braking_mps2) + stopping_margin_m` 必要扩展；进入候选道路走廊的横穿/切入、短时纵向冲突和后方快车单列安全例外。模型关注可再加最多3个有效范围内对象。**常规集最多8个不等于总集合永远8或11个。**静态障碍保留几何但不运行EKF；路外静止物体不加入道路规划。完整场景仍供BEV、模型、检索及回放，不改检索九宫格的范围和语义。

`config/parameters.json` 的 `planning_objects` 使用 `frenet-eight-planning-2.0`，新增开发默认值：`adaptive_front: true`、`reaction_s: 1`、`braking_mps2: 4`、`stopping_margin_m: 5`、`conflict_horizon_s: 3`、`conflict_margin_m: 0.5`。这些是计算范围及粗筛设置，不是全场景安全证明。

MPC时域3秒、控制网格0.1秒、物理子步0.02秒。重规划频率可选1、2、10Hz，默认10Hz；同步模式计算完成后执行前10、5、1段控制。CARLA在 `config/carla.yaml` 改 `carla.planner_hz`，Highway在 `config/parameters.json` 改 `clock.planner_hz`。不要把1Hz误理解为第一项控制恒持1秒。

普通候选必须满足真实尺寸避碰、道路/实线/停止截面、限速、控制边界、控制变化率与跟踪带，并经0.02秒子步复核。未找到普通安全候选则重新计算并验收制动；日志分别保存 `normal`、`safe_brake`、`emergency_unverified`。最后一类按用户确定的实验策略仍执行，但仍可能碰撞，不是安全成功，也不作为合格经验。异常输入仍报错。求解名义预算冷启动2秒、后续1秒、最多10次SQP，不宣称实时达标；同步优先验证规划效果。本地CPU检查可出现数百毫秒单次求解，1、2、10Hz是仿真重规划节拍，整体墙钟耗时必须另测。

四个模型/记忆开关都关时默认仍在1Hz构建BEV、OCC、检索八向和真实DINO，并计算独立无语义参照MPC、质量分和轨迹BEV。`diagnostics.enabled / dino` 为显式诊断开关，默认均true；诊断不打开模型、不读写记忆。真实DINO权重须已下载，不能用随机特征替代。

```bash
python scripts/run_highway.py --scenario highway --seed 1 --duration 30 --fast off --reflection off --no-retrieve --no-write-memory --live --video
python scripts/run_highway.py --scenario merge --seed 1 --duration 30 --fast off --reflection off --no-retrieve --no-write-memory --live --video
python scripts/run_highway.py --scenario roundabout --seed 1 --duration 30 --fast off --reflection off --no-retrieve --no-write-memory --live --video
python scripts/run_highway.py --scenario intersection --seed 1 --duration 30 --fast off --reflection off --no-retrieve --no-write-memory --live --video
```

原生画面青色线为当前实际计划；轨迹BEV与原始检索图分开。`plans.jsonl`保存参考身份、活动目标、筛选与例外理由、约束残差、密集几何检查和执行分类；`summary.json`分别统计制动、紧急、实际碰撞、到达与异常。`timing.jsonl`分别记录EKF、MPC、表示、额外参照、质量、物理步进、显示录像和整体周期；配置10Hz不等于墙钟达到10Hz。新的有效目标可复用移位控制作为数值初值，必须重新积分、计价、验算；不能复用旧目标代价或旧状态轨迹。

本地确定性几何/协议和Highway检查不能代替真实CARLA动力学、真实模型或全部场景验收。未验证场景应查看紧急率、碰撞、违规、导航进度和计时，不以程序完成一轮认定规划成功。


近停车制动使用 `stop_control.py` 的离散释放包络，避免将加速度直接截为 `-v/dt` 造成jerk跳变。若当前速度与上一实际制动已导致jerk和非倒退无法共同满足，记录 `braking_release_conflicts` 并归类紧急；不把该冲突伪装为正常可行。

## 3. 配置两个模型

系统提示词在 `config/prompts.yaml` 中修改：`fast.system` 用于快模型，`reflection.system` 用于反思模型，两者独立。使用 YAML 多行文本 `system: |` 编写有长度限制的结构化说明；程序不对大括号作模板替换。最终回答必须且仅包含 `attention`、`local_goal`、`explanation`，协议版本为 `local-goal-frenet-1.0`。修改后重新启动生效，实际提示词原文及SHA256保存在每轮 `configuration.json` 的 `models` 中。

单目标回答格式（以下数值仅示意，仍须通过当前道路和交通校验）：

```json
{
  "attention": [],
  "local_goal": {"delta_s_m": 40.0, "d_m": 3.5},
  "explanation": {
    "情境": "输入中可验证的事实和关键交互。",
    "依据": "道路、导航、规则约束及目标选择依据。",
    "参考答案": "该目标表达的行驶意图及选择理由。",
    "适用条件": "结论成立的条件。"
  }
}
```

`delta_s_m` 从输入原帧自车投影开始沿冻结参考线向前，`d_m` 相对同一参考线左正右负，均为米；`d_m`不是相对自车的横移。新有效目标在下一次MPC立即替换旧目标；自车中心纵向经过原目标截面后全部释放，即使横向未完成。目标没有到达时间或目标速度，不是停车终点。解释目标所需的参考、合法候选车道及导航规则摘要由程序随原帧提供，不能用BEV直角坐标替代Frenet弧长。

指定另一份提示词文件（相对路径以项目根目录为准）：

```bash
python scripts/run_highway.py --scenario highway --prompts-config config/prompts.yaml --fast api --reflection api
```

从 `config/models.example.yaml` 创建个人配置后，编辑 `config/models.yaml`，分别填写 `fast` 和 `reflection` 的 `base_url`、`model`。`api` 表示官方 API，`server` 表示已部署的 OpenAI 兼容服务，`off` 表示关闭。两个角色可独立选择。请保持 `"off"` 的引号；不要将个人密钥写入文件或提交到仓库。

```yaml
fast:
  backend: "off"
  base_url: "https://服务地址/compatible-mode/v1"
  model: "实际模型名称"
  api_key_env: FAST_VLM_API_KEY
  timeout_s: 120
  max_tokens: 4096
reflection:
  backend: "off"
  base_url: "http://服务器地址:端口/v1"
  model: "实际部署模型名称"
  api_key_env: REFLECTION_VLM_API_KEY
  timeout_s: 300
  max_tokens: 4096
```

官方 API 的密钥通过环境变量设置，不写入配置文件。无鉴权的本地服务可以不设置密钥。

PowerShell：

```powershell
$env:FAST_VLM_API_KEY = '填写实际密钥'
$env:REFLECTION_VLM_API_KEY = '填写实际密钥'
```

Git Bash：

```bash
export FAST_VLM_API_KEY='填写实际密钥'
export REFLECTION_VLM_API_KEY='填写实际密钥'
```

只调用快模型，不检索、不写记忆：

```bash
python scripts/run_highway.py --scenario highway --fast api --reflection off --no-retrieve --no-write-memory --live
```

两个服务器模型，不检索、不写记忆：

```bash
python scripts/run_highway.py --scenario highway --fast server --reflection server --no-retrieve --no-write-memory --live
```

## 4. 记忆采集、检索和冻结库实验

检索或写记忆需要真实 DINOv2 Base 权重。默认使用仓库内的 `models/dinov2-base`，首次克隆后先执行 `git lfs pull`。使用其他本地目录时，在下列命令追加 `--dino-model "模型目录"`。纯基础诊断默认也加载DINO；只有显式关闭DINO诊断且检索、写入均关闭时才不加载。

当前表示版本为 `bev-occ-grid-2.1`：方向箭头仅用于自车、机动车、自行车及行人；静态障碍物继续绘制真实轮廓并参与规划。该变化也影响无字BEV和DINO特征，因此旧2.0库原样保留、明确排除混检，需要单独重建特征后再使用；OCC和八向公式未改。NAVSIM白底、浅灰道路及对象配色，不绘制车道边界线和分割线，仅保留深灰虚线中心参考路径；历史轨迹采用鲜亮的同色系饱和度渐变。目标参考线与沿线距离标记只进入模型图和回放，不进入DINO特征图或OCC。模型文本含当前状态、八向区域、目标参考摘要与图像解释，命中时附历史情境和参考答案；不发送完整地图边界、历史状态列表或检索四矩阵。目标参考摘要另携带输入帧、仿真时间、参考身份、当前横偏、候选车道及相关导航规则，供正确解释Frenet目标。未入格的可见对象仍提供。

新经验须声明 `local-goal-frenet-1.0` 并保存原帧目标参考。本次三通道表示升级另要求道路存在/可跨越属性、观测掩码及完整运动状态。缺少这些证据的旧记录明确排除，原库保留，不补零、不自动迁移。被排除原因可在轮初 `memory_snapshot.jsonl` 和 `configuration.json` 查看，或使用下方 `memory_admin.py list`。

先要求三通道证据均有效且八向分数至少0.2，再由BEV、OCC、八向各取最相似的一条；同一经验ID合并，最多返回3条，不补齐，不再加权融合。确认空场景可以是有效证据，遮挡未知不能当成空场景。以下可调参数在 `config/parameters.json` 修改，并随每轮配置保存；改变参数后与旧参数签名不同的经验不会静默混算。

| 检索参数组 | 开发默认值及含义 |
|---|---|
| `retrieval.minimum_grid_score` | `0.2`，先要求三通道证据有效、八向达到此分数，再按各通道独立选择 |
| `retrieval.per_channel_k / selection` | 固定 `1 / per_channel_top1`，BEV、OCC、八向各取第一名；同分按经验ID |
| `retrieval.merge_same_record` | 固定 `true`，同ID只向模型展示一次，保留全部命中通道；不补齐第二名，返回最多3条 |
| `grid_distance.cutoff_c` | `0.5`，部分匹配阈值；每个未匹配对象代价为其一半 |
| `grid_distance.min_joint_known_fraction` | `0.75`，八向共同明确比例下限 |
| `grid_distance.lambda_T / lambda_E` | `1 / 0.5`，道路和自车运动差异惩罚 |
| `grid_observation.empty_min_coverage` | `0.95`，无对象区域被确认空所需覆盖率 |
| `grid_observation.empty_unknown_area_tolerance_m2` | `0.25`平方米，空区覆盖计算的离散几何容差 |
| `occupancy_distance.min_common_known_fraction` | `0.5`，各类别共同观测覆盖下限 |
| `occupancy_distance.min_foreground_comparable_fraction` | `0.95`，本方已知前景在共同支持下的留存下限 |
| `occupancy_distance.empty_min_known_fraction` | `0.95`，整个OCC裁剪域确认双空所需本方覆盖率 |
| `occupancy_distance.min_union_area_m2 / area_tolerance_m2` | `0.01 / 0.01`平方米，面积有效性阈值及舍入容差 |

本版取消加权融合、融合分门槛和最终Top-K重排；旧 `weights / minimum_score / final_k` 配置会明确报错，请使用以上字段。`retrieval.jsonl` 保存各通道第一名及 `matched_channels`，不再生成 `fused` 分数。检索策略版本为 `bev-occ-eight-channel-top1-3.0`。已有 `bev-occ-eight-joint-2.0` 经验在表示、单通道指标、观测证据与DINO指纹均兼容时仍可只读使用，始终按新选择策略计算，不迁移或改写原库。

以上为开发起点，不是已优化的检索参数。禁跨邻道的对象只退出八向集合，仍进入完整场景、BEV、OCC、模型输入和物理避碰。

空库采集，不检索历史：

```bash
python scripts/run_highway.py --scenario highway --fast api --reflection api --no-retrieve --write-memory --memory-mode collect --memory-write-dir outputs/memory_db/collection_01 --episodes-dir outputs/episodes/collection_01 --live
```

边检索边积累，读写同一个自定义库：

```bash
python scripts/run_highway.py --scenario highway --fast api --reflection api --retrieve --write-memory --memory-mode collect --memory-dir outputs/memory_db/experiment_01 --episodes-dir outputs/episodes/experiment_01 --live
```

冻结库主实验，只读、不写：

```bash
python scripts/run_highway.py --scenario highway --fast api --reflection api --retrieve --no-write-memory --memory-read-dir outputs/memory_db/frozen_01 --episodes-dir outputs/episodes/frozen_eval --live
```

读固定库，修正经验写入另一个库：

```bash
python scripts/run_highway.py --scenario highway --fast api --reflection api --retrieve --write-memory --memory-mode formal --memory-read-dir outputs/memory_db/frozen_01 --memory-write-dir outputs/memory_db/new_reflections --episodes-dir outputs/episodes/online_eval --live
```

`collect` 接收通过验收的快答案或反思答案；`formal` 只接收通过验收的反思答案。没有有效模型答案时，开启写入也不会生成经验。检索结果供快模型使用；关闭快模型时，检索只会留下检索记录，不直接改变 MPC。每轮只读轮初快照，本轮新增经验从下一轮起才可检索。

## 5. 批量运行与 seed

连续 50 轮，每轮固定 seed=20：

```bash
python scripts/run_highway.py --scenario highway --episodes 50 --seed 20 --seed-mode fixed --duration 30 --fast off --reflection off --episodes-dir outputs/episodes/fixed_seed
```

连续 50 轮，seed 依次为 1 至 50，开启模型、检索和积累：

```bash
python scripts/run_highway.py --scenario highway --episodes 50 --seed 1 --seed-mode increment --duration 30 --fast api --reflection api --retrieve --write-memory --memory-mode collect --memory-dir outputs/memory_db/highway_01 --episodes-dir outputs/episodes/highway_01 --live
```

将 `--scenario` 改为 `merge`、`roundabout` 或 `intersection` 即可运行其他场景。库不会按场景自动拆分。轮末等待已接纳的反思任务完成，再启动下一轮；忙碌队列满时拒收的新任务不会补跑。

## 6. 参数表

| 参数 | 含义与默认值 |
|---|---|
| `--scenario` | `highway / merge / roundabout / intersection`，默认 `highway` |
| `--episodes` | 轮数，默认 1 |
| `--seed` | 初始随机种子，默认 1；相同种子不保证模型回答或受墙钟预算影响的数值求解结果逐位相同 |
| `--seed-mode` | `fixed` 每轮同种子；`increment` 每轮加 1，默认后者 |
| `--duration` | 每轮最长仿真秒数，默认 30；不等于真实运行耗时 |
| `--vehicles` | 覆盖场景初始交通车辆数；匝道和环岛受原生交通模板数量限制，超出会报错；不删除地图静态障碍物 |
| `--spawn-probability` | 路口每仿真秒生成交通的概率；0 关闭生成 |
| `--fast`、`--reflection` | 分别覆盖模型配置：`off / api / server`；不指定则沿用models.yaml，纯MPC请显式off |
| `--retrieve / --no-retrieve` | 检索开关，默认关闭 |
| `--write-memory / --no-write-memory` | 入库开关，默认关闭 |
| `--memory-mode` | `collect / formal`，默认 `collect` |
| `--memory-dir` | 读写路径简写，默认 `outputs/memory_db/default` |
| `--memory-read-dir` | 独立读取库，优先于 `--memory-dir` |
| `--memory-write-dir` | 独立写入库，优先于 `--memory-dir` |
| `--dino-model` | 本地 DINOv2 Base 目录或模型标识；默认 `models/dinov2-base`，不接受随机替代特征 |
| `--episodes-dir` | 运行产物根目录，默认 `outputs/episodes` |
| `--prefix` | 本次批量 ID 前缀，默认含微秒的时间戳；重复 episode ID 会报错 |
| `--live` | 显示原生pygame窗口；同步模型调用期间仿真暂停，不改变求解预算 |
| `--video` | 原生物理子步50FPS保存 `video.mp4`，可与 `--live` 同用，叠加青色当前规划轨迹 |
| `--config` | 算法参数 JSON，默认 `config/parameters.json` |
| `--scenarios-config` | 场景与路线 YAML，默认 `config/scenarios.yaml` |
| `--models-config` | 服务配置 YAML，默认 `config/models.yaml`；首次克隆须由 `config/models.example.yaml` 创建 |
| `--prompts-config` | 系统提示词 YAML，默认 `config/prompts.yaml`；含 `fast.system` 与 `reflection.system`，启动时固定读取 |

编辑 `config/scenarios.yaml` 后直接运行；使用其他文件时指定：

```bash
python scripts/run_highway.py --scenario highway --scenarios-config config/my_scenarios.yaml --live --video
```

高速场景配置示例（其他场景的段落保留原配置）：

```yaml
highway:
  env_id: highway-v0
  lanes_count: 4
  vehicles_count: 12
  initial_speed_mps: 25
  desired_speed_mps: 25
  destination: "1"
  native:
    initial_lane_id: null
    ego_spacing: 2
    vehicles_density: 1
    screen_width: 1200
    screen_height: 300
    scaling: 11
    centering_position: [0.3, 0.5]
    show_trajectories: false
    real_time_rendering: false
```

| YAML字段 | 范围和含义 |
|---|---|
| `initial_speed_mps / desired_speed_mps` | 各场景初速度、软参考速度，单位m/s；后者不是硬速度上限，实际速度还受进度代价、弯道与道路限速影响 |
| `vehicles_count` | 初始周车数；匝道/环岛原生模板最多4辆；0不删除静态障碍 |
| `lanes_count` | 只用于高速，正整数 |
| `destination` | 原生道路图中的自车导航目的节点；不可达时报错 |
| `initial_longitudinal_m` | 可选，自车初始车道上的纵向位置 |
| `exit_distance_m` | 最后一段导航车道起点起算的目标纵向位置；最多取该车道长度减5米，高速不使用 |
| `spawn_probability` | 只用于路口，每仿真秒的生成概率，0—1 |
| `native.initial_lane_id` | 只用于高速，0起始；null由原生种子选择 |
| `native.ego_spacing / vehicles_density` | 只用于高速，原生初始间距及车流密度，正数 |
| `native.incoming_vehicle_destination` | 只用于环岛，null或0/1/2，原生入环交通目的选项 |
| `native.screen_width / screen_height` | 原生窗口和视频尺寸，至少64的偶数；不改变BEV尺寸 |
| `native.scaling` | 原生画面像素/米，正数；不改变物理道路 |
| `native.centering_position` | 自车在画面中的比例位置，两个0—1数值 |
| `native.show_trajectories` | 原生历史轨迹显示；不是未来MPC轨迹 |
| `native.real_time_rendering` | 原生显示是否按仿真时间限速；不能加速求解器 |
| `connection_boundary_clip_radius_m` | 公开连接口的边线适配范围，属于地图/规则研究参数 |

命令行 `--vehicles / --spawn-probability` 优先于YAML。未知字段或试图覆盖固定物理频率、连续控制类型会报错，不会静默忽略。有效配置及哈希保存于每轮 `configuration.json`。算法、坐标、地图适配和判定规则变动需同步理论稿。

## 7. 查看运行结果和记忆

每轮目录保存 `summary.json`（终止原因及统计）、`run.log`、`timing.jsonl`、`model_inputs.jsonl`、`model_outputs.jsonl`、`retrieval.jsonl`、`memory_snapshot.jsonl`、`predictions.jsonl`、`plans.jsonl`、`quality.jsonl`、`execution_audit.jsonl`、`frames.jsonl` 及 `captures/`。`timing.jsonl`区分表示、模型等待、预测、主MPC、参照和评分，明确管线计时范围；summary区分首次与后续MPC耗时和超期次数。入库经验的资产同时保留实际预测、规划与参照结果；反思通过后另存原帧重新规划验收结果。未启用的功能不一定产生对应日志。批量统计保存在输出根目录的 `*_batch.json`。

PowerShell 实时查看日志：

```powershell
Get-Content -LiteralPath 'outputs/episodes/实际轮次目录/run.log' -Wait -Encoding UTF8
```

列出库中可检索记录及被排除的记录：

```bash
python scripts/memory_admin.py --memory-dir outputs/memory_db/experiment_01 list
```

查看某条经验（将 ID 替换为 `list` 输出中的实际 ID）：

```bash
python scripts/memory_admin.py --memory-dir outputs/memory_db/experiment_01 show "经验ID"
```

人工确认错误后停用，保留原始资产与审计：

```bash
python scripts/memory_admin.py --memory-dir outputs/memory_db/experiment_01 invalidate "经验ID" --reason "已人工确认的错误原因"
```

停用结果对随后建立的轮初快照生效；已经运行的轮次保持其冻结快照。需要立即排除该经验时，应结束当前轮后重新运行。

## 8. 单目标与势场参数

基础参考参数在`config/parameters.json`的`local_reference`组配置；单目标参数在`local_goal`组，答案协议在`models.answer_protocol_version`。原`guide_points`和`mpc.weights.guide_contour`不再接受。

| 单目标参数 | 初值及含义 |
|---|---|
| `local_goal.minimum_delta_s_m / maximum_delta_s_m` | `2 / 70`米，同时受原参考线覆盖限制 |
| `local_goal.maximum_lateral_slope` | `0.35`，几何粗筛的`abs(d-d_ego)/delta_s`上界 |
| `local_goal.passed_tolerance_m` | `0.001`米，真实自车中心的纵向通过容差 |
| `local_goal.projection_window_m` | `120`米，原参考线连续投影搜索窗 |
| `local_goal.attraction_weight` | `3`，单次首次到达吸引项的独立权重，开发起点 |
| `local_goal.longitudinal_scale_m / lateral_scale_m` | `30 / 3.5`米，目标处切向/法向归一化尺度 |
| `risk.classes.*.influence_m` | 机动车20、自行车10、行人10、静物12、不可跨越线1.5米，按实体净距，不是硬安全距离 |

## 9. CARLA 0.9.15：独立世界与第一批交互场景

本分支已提供CARLA入口、SDK桥与脚本场景，尚未在真实CARLA服务器完成实测。局部交互段要求同层、近直、非路口；后续自车导航允许沿公开连接经过路口和温和弯道，不要求整条180米都是直线。找不到合法交互走廊、足够导航长度、邻道或人行道时明确失败，不换模板、不自动缩短路线。每份场景YAML可以指定任意已安装的CARLA地图名称或资源路径，不限定Town05/06/10HD；已安装地图仍须满足该交互模板的道路条件。

主程序继续使用Python 3.12的独立环境；CARLA SDK在已有`carla37`（或安装了0.9.15 SDK的Python 3.8）环境中运行，通过子进程交换数据。不要把CARLA旧版SDK安装进主环境，不需要修改已有`carla`、`carla37`和模型服务环境。

轮位/参考点诊断：轮位按CARLA 0.9.15的世界厘米转换，包围框中心保留为actor局部米；坐标变换只修改临时副本。当前输出的`state_sources.wheel_geometry_version`为`carla-wheel-reference-1.1`。如实际轮位仍不合法，错误会列出前后轴局部位置、框中心、轴距和后轴到中心距离；不会使用车长或虚构尺寸代替。无需为本次修复更改场景YAML或重装环境。

先在服务器检查SDK解释器，不连接世界：

```bash
/home/zjj/miniconda3/envs/carla37/bin/python -c "import carla; print(carla.__file__)"
conda activate ep-driver-carla
cd /data/zjj/code/EP-Driver/EP-Driver-CARLA
python scripts/run_carla.py --list-scenarios
python scripts/run_carla.py --scenario slow_lead --dry-run
```

`--dry-run`只检查配置和输出运行清单，不导入CARLA SDK、不创建客户端、不加载DINO、不写运行产物。两个模型默认关闭，即使个人YAML配置为开启也不会隐式调用；启用时必须传`--fast server/api`或`--reflection server/api`。模型都关闭且个人配置不存在时可直接使用模板；启用模型时缺少个人配置会报错。

用户先在独立端口启动CARLA 0.9.15服务。该端口不得有其他人的车辆、行人或同步控制程序；关闭已有CARLA观察脚本中主动调用`world.tick()`的功能。程序不自动启动CARLA服务器，也不会接管已有同步世界。首次无模型冒烟运行：

```bash
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --host 127.0.0.1 --port 22000 --scenario-config config/carla_scenarios/slow_lead.yaml --runtime-mode sync --exclusive-world --allow-world-reset --duration 3 --fast off --reflection off --no-retrieve --no-write-memory --video --episodes-dir outputs/carla/smoke
```

`--exclusive-world`是每次真实运行的明确确认。当前空世界地图与`--map`不同时默认拒绝；确认该服务专用后，另外加`--allow-world-reset`才允许切换地图。两项开关不允许清理其他人的车辆或接管其他同步客户端。结束时只销毁本轮创建的对象并恢复本轮修改前的世界设置；被切换的地图不自动切回。

先检查当前服务地图上的场景选址（不生成演员、不改世界设置、不换地图、不推进仿真，不调用模型）：

```bash
python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario-config config/carla_scenarios/slow_lead.yaml \
  --seed 1 --inspect-scene \
  --prefix slow_lead_check --episodes-dir outputs/carla/scene_checks
```

报告保存为 `outputs/carla/scene_checks/slow_lead_check_scene_selection.json`。其中可用起点只是地图条件通过，仍需运行验证生成空间、传感器及控制。把选定索引填入该地图YAML的 `experiment.spawn_index` 可固定地点；不能照搬其他地图索引。真实运行每轮保存 `carla_scene_selection.json`，选址失败时也保存逐点原因。

场景模式版本为 `carla-scene-1.1`；旧1.0文件仍可读，新增默认参数和实际选址算法版本会被记录：

| YAML字段 | 含义 |
|---|---|
| `experiment.route_length_m: 180` | 完整自车导航长度，不自动缩短 |
| `experiment.interaction_length_m: null` | 自动计算近直非路口的局部交互段：慢车/急刹/占道为前车距离+10m；行人再加偏移；切入不少于前车距离+前车速度×(切入时长+8s)。可显式加长，不能小于模板所需长度 |
| `experiment.navigation_step_yaw_limit_deg: 30` | 每2m导航采样的局部航向变化上限，非整条导航累计转角；交互段仍保留相对起点0.3rad限制 |
| `experiment.spawn_index` | null按seed检查起点；整数只检查该点，不因失败偷换地点 |

导航采用公开合法后继中局部转角较小者优先的有界回溯；保持相对起点1.5m的高度范围、每段0.1—3m连续性、0.5m重复空间点拒绝，单起点最多扩展5000个搜索节点。这不是指定终点的全局A*。远端导航可经过路口不代表已经验证完整无信号路口路权、坡道或回环场景。上述阈值是开发起点。

五类脚本交互复用同一个核心预测、目标管理、MPC、检索及反思流程。**每个场景一份独立YAML**；`config/carla.yaml`只配置服务、SDK和传感器。以下`--scenario`分别选择`config/carla_scenarios/`中的同名文件：

```bash
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --exclusive-world --scenario slow_lead --duration 30 --video
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --exclusive-world --scenario hard_brake --duration 30 --video
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --exclusive-world --scenario cut_in --duration 30 --video
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --exclusive-world --scenario parked_obstacle --duration 30 --video
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --exclusive-world --scenario occluded_pedestrian --duration 30 --video
```

复制场景文件即可创建相似变体，然后修改新文件中的`name`、`map`、`desired_speed_mps`和`experiment`参数。显式文件入口与`--scenario`互斥，相对路径以项目根目录为准：

```bash
cp config/carla_scenarios/cut_in.yaml config/carla_scenarios/cut_in_close.yaml
nano config/carla_scenarios/cut_in_close.yaml
python scripts/run_carla.py --scenario-config config/carla_scenarios/cut_in_close.yaml --dry-run
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --exclusive-world --scenario-config config/carla_scenarios/cut_in_close.yaml --duration 30 --video
```

同地点变体应固定`map`和`experiment.spawn_index`。首次`spawn_index: null`会按seed查找适用地点；从`frames.jsonl`的`actual.events`中找到`created`事件的实际索引，再填回YAML。随后可改变车距、速度、触发时机、切入侧及切入时长。相同seed而参数改变时，自动搜索可能选到不同地点，不能仅用seed认定同地点。每轮`configuration.json`保留原场景YAML、SHA256和生效参数，`carla_map.json`保存地图OpenDRIVE指纹，`created`事件保存起点道路标识及世界位姿。

确认本地Qwen服务已启动，并在`config/models.yaml`分别填写快模型/反思模型地址及模型名后，先进行不使用记忆的完整模型链路测试：

```bash
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --exclusive-world --scenario slow_lead --duration 3 --fast server --reflection server --no-retrieve --no-write-memory --video --episodes-dir outputs/carla/model_smoke
```

六个当前RGB视角拼为一张图，相机拼图在前、同帧BEV在后，送给快模型和原帧反思。提示词在`config/prompts.yaml`修改。只有模型、`--video`或`--live`需要图像时，才创建六路RGB，相机`sensor_tick=0.1`，真实采集和录像10Hz，原图及模型源帧归档1Hz；不创建任何深度相机。纯基础且无录像/显示时为零相机。对象输入采用同层120米范围内真值，不判断遮挡，来源版本`range_limited_ground_truth_no_occlusion-1.0`；BEV/OCC/八向仍沿用各自范围和评分含义。该输入不等同于相机可见性或真实感知输出。`camera_frames/`只含真实RGB、拼图、标定和源帧状态，存储版本`carla-rgb-source-1hz-2.0`；不重复旧图补帧，不修改原实验资产。RGB保持原生后处理，只转换BGRA到RGB。单Frenet目标协议及反思仅审查原帧的职责不变。

CARLA规划轨迹显示与录像（只影响显示，不改变MPC）：

```bash
# 六相机真实RGB拼图录像；不在世界中生成debug线，适用于严格原图实验。
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --host 127.0.0.1 --port 22000 --scenario-config config/carla_scenarios/slow_lead.yaml --exclusive-world --allow-world-reset --duration 30 --fast off --reflection off --no-retrieve --no-write-memory --no-world-plan --video --episodes-dir outputs/carla/base_video

# 同时在CARLA世界绘制青色MPC规划线，便于旁观端观察。
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --host 127.0.0.1 --port 22000 --scenario-config config/carla_scenarios/slow_lead.yaml --exclusive-world --allow-world-reset --duration 30 --fast off --reflection off --no-retrieve --no-write-memory --world-plan --video --episodes-dir outputs/carla/world_plan_video
```

CARLA运行时钟（`carla-range-clock-3.0`）：在`config/carla.yaml`设置`runtime_mode: sync`或`async`，也可用`--runtime-mode`覆盖。CARLA重规划在YAML中可选1、2、10Hz（默认10Hz），MPC内部30段×0.1秒，物理50Hz；Highway通过通用时钟配置选择同样频率。

- `sync`：当前状态冻结，完成本次模型与规划后，按1/2/10Hz分别执行前10/5/1段控制，每段5个物理步；不会将首控制保持整个周期。
- `async`：第一条有效基础规划后启动独立世界时钟；后台快模型最多一个在途任务及一个最新待处理帧。等待模型时MPC继续依据当前有效目标或基础输入规划；返回目标绑定原参考线，不能随车漂移。延迟原帧的质量、反思和入库验收在后台重新规划，不能冒用当前帧分数。
- 异步由专用线程以固定物理步驱动，CARLA底层仍保持`synchronous_mode=True`以对齐状态与RGB；这里的异步指世界不等待主程序或模型，并不表示取消固定步长。初始化到首个计划前的暂停、墙钟落后、过期控制段、跳过的源帧均记录。硬件太慢时只能报告落后，不能据此宣称实时。
- 有限控制序列按源时标消费；若新计划长期未返回，3秒序列耗尽则明确结束本轮，不无限保持最后控制。轮末先停止世界推进，再排空已接纳快模型、原帧验收和反思任务，最后清理传感器。

以下两条分别运行同步与异步纯基础实验，主程序使用`ep-driver-carla`环境；`--carla-python`仅指定独立SDK子进程解释器。去掉`--video`即可不创建本程序RGB相机；外部观察器仍可独立显示。

```bash
conda activate ep-driver-carla
cd /data/zjj/code/EP-Driver/EP-Driver-CARLA
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --host 127.0.0.1 --port 22000 --scenario-config config/carla_scenarios/slow_lead.yaml --exclusive-world --allow-world-reset --duration 30 --runtime-mode sync --fast off --reflection off --no-retrieve --no-write-memory --world-plan --episodes-dir outputs/carla/base_sync_safe_tracking
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --host 127.0.0.1 --port 22000 --scenario-config config/carla_scenarios/slow_lead.yaml --exclusive-world --allow-world-reset --duration 30 --runtime-mode async --fast off --reflection off --no-retrieve --no-write-memory --world-plan --episodes-dir outputs/carla/base_async_safe_tracking
```

异步模型实验将对应指令改为`--fast server --reflection server`；检索、写入开关仍独立。`timing.jsonl`保留前台阶段计时，`snapshot_s`包含等待新观测和状态构建，另列`observation_wait_s`及`snapshot_build_s`，避免把刻意等待下一1Hz源帧算成构建开销。`bridge_timing.jsonl`分别记录物理tick、传感器等待、状态读取、图像处理和节拍等待；后台原帧验收耗时另列，不计作控制MPC。规划频率是配置与调度目标，实际以源时间间隔、墙钟间隔和deadline统计为准。本地只验证了确定性SDK/模型替身及真实Highway回归；CARLA渲染、服务器吞吐和真实模型效果仍需实测。

`config/carla.yaml`的`world_plan_debug`默认false，命令行`--world-plan/--no-world-plan`优先。开启后在CARLA世界绘制当前有效MPC折线：线宽0.08米、抬高0.15米、青色、寿命为1.1个重规划周期（1Hz时1.1秒）。它可能被RGB或外部旁观相机拍到，严格原图实验请关闭。外部观察脚本连接同一22000端口，只观察、不调用`world.tick()`。

`display_plan_overlay`默认true：本机`--live`和`--video`在六相机RGB拼图的副本上投影青色规划线，并裁剪镜头后方、画布外的线段；该投影不改写原始RGB、模型相机拼图、BEV或DINO图。服务器无桌面时不要加`--live`；通过原有端口转发观察世界，录像不依赖桌面或浏览器窗口。世界线关闭时，录像仍可显示投影轨迹。摄像头视野外的规划段自然不可见，不用虚构图像显示它。

每轮在`--episodes-dir/<episode_id>/`生成`video.mp4`及`video_recording.json`。首个编码器不可用时显式尝试`video.avi`，以日志、状态文件和`summary.json`中的实际`path`为准。仅编码真实取得的10Hz源帧，编码帧率为10fps，不重复旧图补帧；异步前台落后时可能跳过源帧，此时视频时长不代表完整仿真时间，应结合源帧时间戳核查。终态没有对应RGB时不复用上一张图。轮末释放编码器并实际逐帧解码核对帧数、尺寸和帧率，`status=complete`且`playback_verified=true`才表示录制核查通过；解码检查增加收尾耗时。`status=no_frames`表示初始化或规划等阶段没有产生可录像画面，应查看同轮`run.log`和`summary.json`；编码/解码失败明确记录为`failed`。运行中MP4可能尚未封装完成，正常退出后再播放；强制杀进程不能保证尾部封装成功。

BEV文字上限在`config/parameters.json`的`bev.labels`修改：

```json
"labels": {
  "max_dynamic_labels": 8,
  "max_vehicle_labels": 5,
  "max_vulnerable_labels": 3
}
```

三个上限均为0至16的整数。默认在BEV内按车体净距选择最近5辆机动车、最近3个行人/自行车，总动态标签最多8个；自车单独标注。静态环境物和历史残影不写ID，全部实体与轨迹仍绘制，完整范围对象仍提供给模型文本；MPC按常规前30米、后10米的八向代表筛选，并保留可停止距离扩展及直接冲突安全例外。标签避让实体和其他文字，无安全空位或ID过长时省略；不截断ID或生成别名。Frenet参考线每20米标一次弧长，BEV米制刻度保留。DINO继续读取无字底图，标签调整不要求重建原记忆库的DINO特征。

批量20轮，seed21至40；只读指定经验库，新增合格反思经验另写一个库：

```bash
python scripts/run_carla.py --carla-python /home/zjj/miniconda3/envs/carla37/bin/python --exclusive-world --scenario cut_in --episodes 20 --seed 21 --seed-mode increment --duration 30 --fast server --reflection server --retrieve --write-memory --memory-mode formal --memory-read-dir /data/zjj/memory/frozen_carla --memory-write-dir /data/zjj/memory/new_carla_reflections --episodes-dir outputs/carla/cut_in_21_40 --video
```

检索和写入默认关闭、相互独立；`--memory-dir`为同库读写简写，独立路径优先。启用记忆功能仍需真实DINO权重，当前轮新增经验下一轮快照才可见。RGB真实采集与录像10Hz、原图归档1Hz；BEV、检索与诊断源帧1Hz；纯基础默认也构建表示和参照评分。旧库不迁移或改写，新记录在情境与核查资产记录观测来源，旧深度权限记录不能冒称为本次范围真值设置；跨权限使用须在实验中报告。

| CARLA参数 | 含义 |
|---|---|
| `--carla-config` | SDK、服务、传感器公共配置，默认`config/carla.yaml` |
| `--scenario` | 选择内置独立场景YAML，默认slow_lead；与`--scenario-config`互斥 |
| `--scenario-config` | 独立场景YAML路径，可自定义文件名；相对路径以项目根为准 |
| `--carla-python` | 安装CARLA 0.9.15 SDK的解释器绝对路径；默认未配置，真实运行必须提供或在YAML设置 |
| `--host / --port / --map` | 服务地址、RPC端口及地图覆盖；地图默认取场景YAML，不强制Town05 |
| `--exclusive-world` | 确认该服务为本实验专用，每次真实运行必需 |
| `--world-plan / --no-world-plan` | 开关CARLA世界青色规划线；默认关闭，开启可能进入相机原图 |
| `--video` | 录制六相机RGB显示拼图；保存实际路径、帧数与可播放核查状态 |
| `--allow-world-reset` | 专用空世界中允许切换到请求地图，默认禁止；不是强制清空现有世界 |
| `--list-scenarios / --dry-run` | 分别列出配置场景、校验运行清单，均不连接服务 |
| `--live / --video` | 显示或录制同一六相机拼图；需GUI时使用`--live`，无显示服务器用`--video`；录像10FPS，每个实际RGB采样帧一帧 |
| `--fast / --reflection` | CARLA入口默认均为off，显式选择server或api才调用；与highway入口沿用YAML默认的行为不同 |
| 其他批量及记忆参数 | `--episodes --seed --seed-mode --duration --retrieve --write-memory --memory-mode --memory-dir --memory-read-dir --memory-write-dir --dino-model --episodes-dir --prefix --models-config --prompts-config --config`与上文同名定义一致 |

| 独立场景YAML字段 | 含义与开发默认值 |
|---|---|
| `schema_version` | 旧五模板接受carla-scene-1.0/1.1；多演员长路线使用carla-scene-1.2，不使用跨场景继承 |
| `name / template` | 实例名称与模板分开；原五模板保留，新增scripted_traffic多演员模板 |
| `map` | 当前场景地图，填写实际安装的名称或资源路径；CLI --map可显式覆盖并记录 |
| `desired_speed_mps` | 核心规则/MPC期望速度10m/s，与初速度独立，不代表地图限速 |
| `experiment.spawn_index` | null表示按seed搜索满足模板的地点；整数固定当前地图生成点，失败不换位置 |
| `experiment.route_length_m / lead_distance_m` | 导航路线180米、交互对象摆放距离35米；地图waypoint离散摆放，实际位姿另存 |
| `experiment.initial_speed_mps / lead_speed_mps` | 自车初速度10m/s、前车6m/s；静态模板前车为0 |
| `experiment.trigger_distance_m / trigger_min_time_s` | 距离不超过28米且时间至少0.5秒触发；仅用于急刹、切入和行人模板 |
| `experiment.hard_brake_amount` | 急刹模板制动输入，(0,1]；默认1，不是m/s² |
| `experiment.cut_in_side / lane_change_seconds` | 从left或right合法同向邻道切入；持续4秒，均检查可跨越边界 |
| `experiment.pedestrian_speed_mps / pedestrian_longitudinal_offset_m` | 行人1.4m/s，沿路线在遮挡车摆放参考点前6米 |
| `experiment.occluder_lateral_fraction` | 遮挡车向右偏移当前车道宽度的0.35倍，允许[-0.5,0.5]；不改变道路规则 |
| `experiment.ego_blueprint / lead_blueprint / occluder_blueprint` | 自车、交互车、遮挡车的实际车辆蓝图名 |
| `experiment.pedestrian_blueprint` | null按蓝图ID确定性选择首个行人，或填写实际存在的walker.pedestrian.*蓝图 |

各文件显式列出完整实验参数；注释注明仅某模板使用的字段，修改不相关字段不会为其他模板新增交互。以下公共传感器设置仍在`config/carla.yaml`中：

| 公共平台参数 | 含义与开发默认值 |
|---|---|
| `carla.camera_hz / camera_save_hz` | 10Hz采集、1Hz归档；录像10FPS，不补帧 |
| `carla.rgb_postprocess_effects` | 默认true，保留CARLA原生RGB处理；本版本没有深度传感器 |
| `carla.camera_width / camera_height / camera_fov` | 单相机640×360、FOV100度；按消费者创建六路10Hz RGB或零相机 |
| `perception.radius_m` | 当前同层范围真值的半径120米；范围外未知，不做遮挡判断。MPC另按局部八向、可停止距离扩展及直接冲突例外筛选 |

每轮沿用第7节的核心日志，另有`carla_bridge.log`、`carla_map.json`、`carla_environment_audit.json`、`camera_frames/`及相机资产清单；`--video`保存真实相机拼图录像。`carla_environment_audit.json`记录地图环境框的读取范围、数量及镜像网格半尺寸规范化明细。CARLA 0.9.15的组件负缩放可能产生负半尺寸，程序只对有限非零半尺寸取绝对值，保持原世界中心、旋转及八角点集合，不删除障碍物、不修改地图；零半尺寸或非有限值仍拒绝，并报告ID、名称、类别及原始框数值。录像可在RGB显示副本投影MPC轨迹；世界调试线由独立开关控制，计划另存BEV诊断图与`plans.jsonl`。当前验证仅包括接口、配置、几何与测试替身，不能据此宣称已在CARLA实现无碰撞或实时运行。


## 10. 第二批：逐地图长路线与多演员场景

每个场景仍是一份独立的 `config/carla_scenarios/*.yaml`。保留旧五场景，新增16份主体场景及8份 `_opt` 地图变体，共覆盖12个Town、20个公开地图资源名称。Town08/09/14等挑战赛保留地图不在本地已安装公开地图清单中，不虚构可用配置。

**验证边界：道路和布置使用本机CARLA 0.9.15 SDK读取真实OpenDRIVE进行离线检查；没有代替服务器执行物理闭环。** 检查不证明道具蓝图可生成、动态避碰成功或MPC实时。场景配置验收与本轮规划器改动应分别验证；初始化失败、执行误差、碰撞与正常等待须在实验日志中分开，不能全部解释为场景难度。

| YAML文件名（不含.yaml） | 地图 | 路线/m | 时限/s | 交互 |
|---|---|---:|---:|---|
| `town01_urban_follow` | Town01 | 500 | 120 | 城市日常跟驰：慢车与远处前车，连续街区及路口。 |
| `town02_school_crossing` | Town02 | 400 | 150 | 校园式低速路段：两名行人分时从真实人行道横穿；不是地图学校POI标注。 |
| `town03_junction_flow` | Town03 | 600 | 150 | 城市多路口：本车道前车、邻道车辆和公开支路横向来车，保留实际信号约束。 |
| `town04_multilane_exit` | Town04 | 900 | 180 | 多车道主路经公开分流连接进入单车道出口，保持已知导航分支。 |
| `town04_multilane_merge` | Town04 | 900 | 180 | 单车道入口经公开连接汇入多车道主路；前方慢车形成速度调整需求。 |
| `town05_multi_vehicle` | Town05 | 650 | 180 | 多车交互：本车道跟驰、左右邻车不同速度；不预先要求自车换道。 |
| `town05_roadworks` | Town05 | 600 | 180 | 临时施工占道：锥桶渐变收窄、停放工程车辆和邻道通行车辆。 |
| `town06_cut_in` | Town06 | 900 | 180 | 相邻车辆在合法虚线区触发切入，远处慢车增加交互；切入过程连续。 |
| `town06_ramp_entry` | Town06 | 1000 | 180 | 匝道汇入：单车道入口进入多车道公路，前车沿同一合法连接汇入。 |
| `town06_ramp_exit` | Town06 | 1000 | 180 | 公路驶出：已知导航沿公开出口连接驶离多车道主路。 |
| `town07_rural_follow` | Town07 | 600 | 180 | 乡村弯道路段：慢速货车跟驰，保留对向道路和真实弯曲拓扑。 |
| `town10hd_stop_go` | Town10HD | 500 | 180 | 城市走停交通：前车触发停车，停稳等待4秒后恢复前进。 |
| `town11_long_follow` | Town11 | 800 | 180 | 未装饰大地图的连续道路跟驰；用于长路线和坡度适配，不模拟建筑语义。 |
| `town12_mixed_flow` | Town12 | 800 | 180 | 大地图连续街区：两辆不同速度前车，路线包含真实路口。 |
| `town13_roadworks` | Town13 | 800 | 200 | 大地图临时施工：工程车与锥桶占据本车道，邻道保留通行交通。 |
| `town15_campus_crossing` | Town15 | 450 | 180 | 校园地图低速交互：真实人行道起步横穿，前方车辆低速通行。 |

Town01、02、03、04、05、06、07、10HD各有一份文件名以`_opt`结尾的分层地图版本，YAML中显式指定`TownXX_Opt`。它们保持对应主体场景的道路锚点和演员配置；没有把原地图改名后自动替换运行。

### 10.1 无模型运行

在代码更新至本次版本后，使用现有环境与已启动的专用CARLA服务。以下指令不调用VLM、不检索也不写记忆；默认仍构建BEV、OCC、八向表示和真实DINO特征，并运行参照规划与质量诊断。不会关闭CARLA进程。

```bash
conda activate ep-driver-carla
cd /data/zjj/code/EP-Driver/EP-Driver-CARLA
python scripts/run_carla.py --list-scenarios
python scripts/run_carla.py --scenario town01_urban_follow --dry-run

python scripts/run_carla.py \
  --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
  --host 127.0.0.1 --port 22000 \
  --scenario town01_urban_follow \
  --exclusive-world --allow-world-reset \
  --episodes 1 --seed 1 \
  --fast off --reflection off --no-retrieve --no-write-memory \
  --world-plan --video \
  --episodes-dir outputs/carla/long_routes/town01_urban_follow
```

未传`--duration`时使用该场景的`duration_s`，而非统一30秒；传入则显式覆盖。`--world-plan`使旁观窗口可看世界中的规划线，但线也可能被相机拍到；正式无叠加采集使用`--no-world-plan`。RGB采集及录像10Hz，原图和模型源帧1Hz归档；以实际帧号核查，不重复旧图填帧。

可先分场景运行，再串行批量运行选定场景。切勿并行控制同一个CARLA世界。下面批次总共16轮，每轮各用seed1；`BATCH_DIR`保留完整输出和终端日志。单场景返回非零时仍进入下一场景，并单独记录返回码。

```bash
BATCH_DIR="outputs/carla/long_routes_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BATCH_DIR"
set -o pipefail
for scene in town01_urban_follow town02_school_crossing town03_junction_flow \
  town04_multilane_merge town04_multilane_exit town05_roadworks town05_multi_vehicle \
  town06_ramp_entry town06_ramp_exit town06_cut_in town07_rural_follow \
  town10hd_stop_go town11_long_follow town12_mixed_flow town13_roadworks town15_campus_crossing
do
  python scripts/run_carla.py \
    --carla-python /home/zjj/miniconda3/envs/carla37/bin/python \
    --host 127.0.0.1 --port 22000 --scenario "$scene" \
    --exclusive-world --allow-world-reset --episodes 1 --seed 1 \
    --fast off --reflection off --no-retrieve --no-write-memory \
    --world-plan --video --episodes-dir "$BATCH_DIR/$scene" \
    2>&1 | tee "$BATCH_DIR/$scene.log"
  printf '%s,%s\n' "$scene" "${PIPESTATUS[0]}" >> "$BATCH_DIR/exit_codes.csv"
done
```

先检查`summary.json`的终止原因、真实碰撞、进度、帧数和`video_recording.json`。`timeout`只表示用完时限，不等于到达或成功。触发/完成事件保存在帧日志，选址信息在`carla_scene_selection.json`；切入指令结束也不等于真实横向到达。

### 10.2 修改配置构造相似场景

新模板的参数全部在单份YAML中：

| 字段 | 含义 |
|---|---|
| `description` | 实验说明，不自动进入VLM/规划器输入 |
| `duration_s` | 场景默认最长仿真时间，秒；命令行可覆盖 |
| `experiment.route_anchors` | 公开地图`road_id/lane_id/s_m`锚点；第一项固定起点，其余项要求经过，不是周车隐藏路线 |
| `experiment.spawn_index` | 与非空route_anchors互斥；不固定锚点时才使用地图生成点 |
| `experiment.route_length_m` | 请求导航长度，实际2m采样通常略长；不能自动缩短凑通过 |
| `route_requirements.kind` | any、junction、merge、diverge；汇入/分流须有路线选定分支和非路口单/多车道过渡证据 |
| `route_requirements.min_junctions` | 最少不同路口数量，不保证灯控/无保护左转等语义 |
| `route_requirements.min_same_direction_lanes` | 路线上至少一段同向车道数；不代表每处都可变道 |
| `route_requirements.search_limit` | 单起点有界搜索上限；按合法连接，不穿越无连接道路 |
| `experiment.weather` | CARLA天气预设，默认ClearNoon；退出恢复原天气 |
| `actors[].role/type/blueprint` | 唯一演员名称、vehicle/pedestrian/prop、实际资源蓝图 |
| `actors[].placement.s_m` | 沿自车已知导航路线的摆放距离，米 |
| `actors[].placement.lane` | current/left/right，行人使用sidewalk_left/sidewalk_right |
| `actors[].placement.offset_m` | 相对选中车道中心横移，左正，米；非论文模型目标d |
| `actors[].placement.anchor` | 交叉车辆可使用独立公开支路锚点，替代s_m/lane；须验证同层路口内与自车路径横向相交 |
| `actors[].behavior` | follow、parked、cut_in、stop_resume；行人为cross；道具为parked |
| `actors[].speed_mps` | 场景演员目标速度；自车始终使用主程序MPC，不被脚本接管 |
| `actors[].trigger` | time_s与ego_distance_m同时满足才触发；距离null表示只按时间 |
| `actors[].duration_s` | 切入几何过渡时间，秒；不会瞬移；行人按真实剩余距离结束横穿 |
| `actors[].stop_duration_s` | 前车停稳后保持的秒数，再恢复行驶 |

复制一个场景、修改name及演员距离/速度/触发条件，保留地图和锚点，即可构造同路段变体。道路改变后务必重新检查可生成条件；`--inspect-scene`只读检查**当前已加载**地图，不能替你换图。道具缺失、道路不适用或生成碰撞会明确报错，不偷偷丢弃演员。切入不仅要求有邻道，还检查跨越许可；行人要求两端真实人行道。学校路段是实验语义，不声称学校建筑、斑马线或儿童行为已被真实建模。

固定锚点且不修改演员参数时，seed变化不会自动生成不同车距和行为；这些配置主要用于可控同地点对比。交通灯保留服务器周期并记录初始相位，当前seed不保证信号相位完全相同。跨支路车辆使用160m公开路线，之后停下；NPC是简单跟踪、跟驰与信号控制，不是已验证专家策略，也没有实现完整无信号路口协商。

### 10.3 长路线的地图与平台边界

导航通过公开`next(2m)`连通，不再要求整条路线近直或处于起点高度。地图提取沿最近导航段插值高度，以1.5m容差筛选道路，版本`carla-route-elevation-1.0`；拒绝导航自身跨层平面重叠和不可区分的层高投影，不把立交上下层直接并为同一可行驶面。**当前物体的高度准入仍按自车高度±1.5m，坡道远处对象可能被排除；本次没有放宽这项感知权限。**

Town11/12/13/15使用hero自车支持CARLA大地图瓦片加载。公开道路文件的连通性验证不包含GPU瓦片加载、静态环境对象缓存完整性或世界原点重定位；现有0.9.15环境框接口在大地图重定位后的坐标需要服务器进一步核查。这四张图配置可用于开发测试，暂不作为已验证正式闭环能力。建议先运行Town01/02/05，再测试坡道和大地图。Town11本身无完整装饰，不能用其图像代表真实城市视觉复杂度。

本批场景保持原预测、APF、MPC、目标协议、模型开关和记忆接口。新场景不引入真实API调用，也不改变既有记忆库。地图和演员配置越复杂，越需要区分场景初始化失败、主程序内部异常、碰撞及正常等待，不能仅以“跑满时限”统计成功。

### 10.4 初始化、RGB失败与首帧制动排查

`--allow-world-reset`在每轮开始时重新加载世界，即使地图名称相同；加载前设置同步固定物理步长，并在加载时保留设置，依据[CARLA 0.9.15同步及复现实验说明](https://carla.readthedocs.io/en/0.9.15/adv_synchrony_timestep/#physics-determinism)。未授权重载时不会主动重载同图。主程序仍拒绝接管已有车辆、行人、脚本道具或处于同步运行的世界。观看脚本须支持世界重载后重新寻找自车，不能自行调用`world.tick()`。

场景YAML的`experiment.initial_speed_mps`是请求初速。自车先静止生成，经过一次注册物理帧取得CARLA地图限速，再使用`min(请求初速, 地图限速)`作为初始速度；不改写YAML，不改变周车策略，不放宽规划器速度上限。非有限、非正地图限速会明确报错，不补默认值。注册发生在相机创建与本轮`t0`之前，初始化步骤和实际采用值保存到`carla_initialization.json`。该行为属于合法场景初始化，不是运行中将真实车速截断；后续真实超速仍记录为不满足安全条件。

如果初态已经不符合安全认证条件，未来控制无法改写初态，程序跳过注定不能获得正常安全资格的QP迭代，直接对本帧的制动候选重新积分和验收。无法通过验收的制动仍标为`emergency_unverified`；这不是取消限速或将紧急候选改称成功。`plan.diagnostics.speed_limit_evidence`区分实际初速、地图规则限速、配置上限和最终硬上限；曲率参考速度不冒充地图硬限速。

日志同时给出完整场景对象数及实际规划对象数。对象筛选详情保存在`plans.jsonl`及决策源帧的`captures/*/planning_evidence.json`；`selected_count`为实际规划集合大小，不能用完整场景数量估算MPC对象规模。

RGB仍请求10Hz、六视角同帧同时间，原图1Hz落盘；不为补图额外推进物理帧，也不拿旧图拼当前状态。同步失败时，`camera_sync_failure.json`记录实际世界帧、已推进物理帧、最近相机回调、队列丢帧、传感器存活及时间设置，`summary.json.runtime_failure`单列这些证据。成功执行步数为0不代表异常发生前世界完全没有推进。桥出错后不再向它重复发送pause；仍尝试逐项释放资源，并将已经收到的真实展示帧写入视频。一帧录像只表示保留了初始画面，不能算完成一次控制。

服务器0.9.15实测`fixed_delta_seconds=0.02`、`sensor_tick=0.1`时，首两张RGB相隔6帧，此后相隔5帧；浮点前驱值也呈现同样序列。创建后的首张图不能直接作为稳定10Hz采样原点。初始化现在以零初速生成演员，注册后保存原车辆控制并制动保持；六路RGB共同当前帧连续两次相隔5帧后，再按下一共同帧对齐开始实验。该帧前才恢复原控制、设置NPC配置初速与自车合法初速，防止预热期间前车提前驶远；停车车辆原手刹保留。不改变`sensor_tick=0.1`，不增加运行期补图tick，不复用旧图。

预热只允许初始化阶段的有界推进，不触发场景交互、采样驾驶历史、归档图片或计入运行步数。`carla_initialization.json`记录预热帧、相机实际间隔、初速、前后布局和正式源帧的交通灯相位；灯周期仍随世界推进，不承诺seed决定信号初相位。演员绝对及相对自车平面漂移超过0.25m、航向超过3°或发生碰撞时拒绝初始化，不瞬移修正；垂向悬架/落地变化单列。初始化失败详情写入`carla_initialization_failure.json`，已收到的部分相机证据保留。真实服务器完整闭环仍需验收，本地模拟队列测试不能替代真实GPU采样测试。

以后检查相机节拍可直接用短命令（SDK环境运行，实验已退出且使用本任务独立世界）：

```bash
/home/zjj/miniconda3/envs/carla37/bin/python scripts/check_carla_camera_clock.py --port 22000 --exclusive-world
```

脚本对比0.1、其float32前驱与每帧采样三台诊断相机，只收集帧号和时间戳；不会运行车辆、模型或保存图像。它会临时同步推进空闲世界，结束后销毁自建相机、恢复原世界设置并输出JSON路径；同步世界或已有车辆/行人/道具时拒绝运行。
