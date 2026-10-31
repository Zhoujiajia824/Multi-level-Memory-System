# EP-Driver 运行指令与参数

开发分支为 `EP-Driver`。所有运行指令在项目根目录执行；配置和命令中的相对路径均以项目根目录为准。

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

模板中的两个模型均为 `"off"`。`config/models.yaml`、测试、研究文档、验证脚本、记忆库及运行输出不上传；每台机器的模型地址和个人配置自行维护。下述运行指令使用已发布的三个脚本：`run_highway.py`、`check_environment.py`、`memory_admin.py`。

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

```bash
python scripts/run_highway.py --scenario highway --seed 1 --duration 30 --fast off --reflection off --live --video
python scripts/run_highway.py --scenario merge --seed 1 --duration 30 --fast off --reflection off --live --video
python scripts/run_highway.py --scenario roundabout --seed 1 --duration 30 --fast off --reflection off --live --video
python scripts/run_highway.py --scenario intersection --seed 1 --duration 30 --fast off --reflection off --live --video
```

以上指令显式关闭两个模型；记忆检索和写入默认关闭。未指定`--fast / --reflection`时沿用`config/models.yaml`中的个人设置，可能调用模型。纯 MPC 仍使用地图、规则、EKF 和基础风险场。碰撞、到达目标或规划失败会提前结束；30 秒是每轮上限。

本版按仿真时间同步步进。MPC使用实时SQP与OSQP，后续求解目标预算100毫秒，每个规划器实例首次初始化预算500毫秒，控制执行0.1仿真秒。首次初始化、后续求解及超期均单独记录；这不是整个Python程序的硬实时保证，预测、表示、评分和同步模型调用另有耗时。

`config/parameters.json` 中的 `mpc.warm_start_recovery_speed_mps` 默认 `0.3` m/s，表示低于此速度、且无规则等待时允许比较导航初值；设为 `0` 关闭这项低速比较。初值仍须经过同一个MPC优化和完整验收，共用原求解预算，不直接作为控制执行。

`--live` 显示highway-env原生窗口，`--video` 保存相同原生视角的50FPS录像；两者可独立使用。视频按仿真时间播放，不包含等待模型的停顿，尚未叠加MPC轨迹线。自绘BEV另存于 `captures/`，不替代原生录像。

空交通检查（匝道原有静态障碍物仍保留）：

```bash
python scripts/run_highway.py --scenario highway --vehicles 0 --duration 30 --fast off --reflection off
python scripts/run_highway.py --scenario merge --vehicles 0 --duration 30 --fast off --reflection off
python scripts/run_highway.py --scenario roundabout --vehicles 0 --duration 30 --fast off --reflection off
python scripts/run_highway.py --scenario intersection --vehicles 0 --spawn-probability 0 --duration 30 --fast off --reflection off
```

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

检索或写记忆需要真实 DINOv2 Base 权重。默认使用仓库内的 `models/dinov2-base`，首次克隆后先执行 `git lfs pull`。使用其他本地目录时，在下列命令追加 `--dino-model "模型目录"`。关闭检索和写入时不加载 DINO 权重。

当前表示版本为 `bev-occ-grid-1.4`；NAVSIM白底、浅灰道路及对象配色，不绘制车道边界线和分割线，仅保留深灰虚线中心参考路径；历史轨迹采用鲜亮的同色系饱和度渐变。目标参考线与沿线距离标记只进入模型图和回放，不进入DINO特征图或OCC。模型文本含当前状态、八向区域、目标参考摘要与图像解释，命中时附历史情境和参考答案；不发送完整地图边界、历史状态列表或检索四矩阵。目标参考摘要另携带输入帧、仿真时间、参考身份、当前横偏、候选车道及相关导航规则，供正确解释Frenet目标。未入格的可见对象仍提供。

新经验须声明 `local-goal-frenet-1.0` 并保存原帧目标参考。旧多点笛卡尔经验不会自动转为新目标，保留原库但排除新检索；特征版本和九宫格含义不因目标协议改变。需要继续使用旧库时，应在独立旧版本运行，或另行设计显式迁移；当前实现不静默迁移。

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
| `--video` | 原生物理子步50FPS保存 `video.mp4`，可与 `--live` 同用，无规划叠加 |
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

每轮目录保存 `summary.json`（终止原因及统计）、`run.log`、`timing.jsonl`、`model_inputs.jsonl`、`model_outputs.jsonl`、`retrieval.jsonl`、`predictions.jsonl`、`plans.jsonl`、`quality.jsonl`、`execution_audit.jsonl`、`frames.jsonl` 及 `captures/`。`timing.jsonl`区分表示、模型等待、预测、主MPC、参照和评分，明确管线计时范围；summary区分首次与后续MPC耗时和超期次数。入库经验的资产同时保留实际预测、规划与参照结果；反思通过后另存原帧重新规划验收结果。未启用的功能不一定产生对应日志。批量统计保存在输出根目录的 `*_batch.json`。

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
