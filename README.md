# EP-Driver-Fail2Drive

本分支从 `EP-Driver` 的 `6b4853d` 建立，接入固定版本的官方 Fail2Drive（`bceb18afb7d62e06e990739e013e750d92ad7a88`）。Fail2Drive 入口支持 `--execution-mode A_base`（纯基础规划）和 `--execution-mode B_model`（基础规划＋服务器模型，默认）。两组使用相同路线、seed、官方规则和表示／参照评分设置，均不写记忆；A不读取模型配置、记忆库或调用任何模型。B快模型不检索，默认在论文规划轨迹验收失败时触发原帧反思，可用 `--reflection off` 关闭；反思可显式读取独立的新长尾记忆库。后面的 highway 说明保留作原项目参考。

官方 evaluator 独占世界推进、演员生命周期、终止条件与评分；使用官方同步 20Hz，不设置项目单轮 60 秒上限。Agent 阻塞等待冻结源帧的规划结果，规划可选 1/2/10Hz，正常模型按仿真时间 1Hz 调用。官方墙钟 watchdog 保留；快模型默认最多两次请求、反思最多一次，每次60秒，同一源帧总HTTP上限3次。启动时核验总预算给规划、表示和归档留出至少30秒，数值 worker 总等待上限240秒，不能无限等候。

这是 **MAP／范围内当前真值输入** 适配。使用当前对象状态、公开地图和官方导航，不读取场景未来事件或周车隐藏路线，不能称为纯视觉感知提交。模型输入是六路同帧 RGB、BEV 和筛选后的候选真值。BEV、OCC、八向张量、DINO、独立参照与内部质量评分保留；它们的评分不代替官方成绩。

模型请求使用最小 JSON Schema 约束四部分结构与行动枚举，真实候选ID、原帧目标和道路许可仍独立严格验收；单字、占位、重复字符理由明确拒绝，不将该检查冒称语义正确证明。每个源帧最多两次请求，退化输出重试携带错误反馈并放松输出格式，不原样重复固定温度0的失败请求，也不修改目标、删除虚构ID或编造理由。空白循环、正文截断、上下文拒绝和网络错误分别归档，记录实际服务usage及图像／提示词哈希。`max_tokens`不再硬编码，可在模型配置调整；默认4096、配置上下文16384与当前8000服务一致。上下文窗口未作tokenizer精确预检，不能用字节估计声称已测准，真正超限明确失败，不裁图文输入。

规划连续性采用上一已选择的规划加速度，执行器反馈请求单独记录。无安全解时重新生成并审核非正加速度的制动候选；如必须放松首段纵向jerk，会明确记录且仍标紧急状态。执行端禁止紧急制动重新产生油门。有限路线末端仅用已验证公开车道补足坐标支持，原终点、模型目标区间和官方到达条件不改。离线回归不代替真实CARLA验收。

## Fail2Drive 环境与身份

服务器专用安装位置：

| 用途 | 路径／环境 |
|---|---|
| 官方未修改源码 | `/data/zjj/code/EP-Driver/fail2drive-devkit` |
| 定制 CARLA | `/data/zjj/carla_fail2drive` |
| 评测、Agent、定制 SDK | Conda `ep-fail2drive`，Python 3.10 |
| 数值规划与 CPU DINO | Conda `ep-driver-fail2drive`，Python 3.12 |
| 服务器模型 | `http://127.0.0.1:8000/v1`，`Qwen2.5-VL-3B-Instruct` |

启动入口核验官方源码、模拟器安装清单、启动脚本／服务器二进制／wheel 哈希，并将实际导入的 SDK 扩展与该 wheel 的内容哈希绑定。普通 CARLA 0.9.15 的版本号相同不代表定制资产相同。`--dry-run` 仅查本地文件、导入 SDK 和打印计划，不连接 CARLA。

`config/fail2drive.example.json` 是独立运行配置，模型模板为 `config/fail2drive.models.example.yaml`；个人地址和认证可放在忽略的 `config/models.yaml` 并用 `--models-path` 指定。DINO 必须是实际权重而非 LFS 指针。数值 worker 固定隐藏 CUDA，默认两个 CPU 线程；服务器模型仍由已有服务执行。

对象车道范围使用 `config/parameters.json` 的 `planning_objects`（默认前40米、后20米），候选范围继续遵循车道集合与当前 BEV 集合的并集。Fail2Drive 将已确认的物理异常实体也纳入 BEV 候选和障碍占据通道；围栏、墙、杆不会因 `static` 前缀被统一当作障碍物。移动异常实体明确保存 `motion_model=free`，不套用静止障碍预测或机动车道先验。实际车体碰撞检查保留。

论文规划参数统一来自 `parameters_path`，适配器不覆盖速度代价归一化、权重或增加执行净空软代价；旧私有运行配置中的 `speed_normalization_mps` 只作为未采用字段归档。有效输入的控制与独立参照均运行论文规划器，外部审计只评价、不替换轨迹；必需导航或交通规则输入无效时记录接入错误并结束该样本，不生成额外制动控制。模型目标连接采用连续插值参考与五次偏移，精确保留原始目标世界坐标，仍检查实际曲率、车体、道路和不可跨线。

势场坐标的起端覆盖只从已确认的当前车道后向几何补充，歧义前驱不擅自接入；原任务路线、终点和模型冻结目标不变。停止线按实际受控车道对应的导航段及公共高程关联，不能仅凭该车道在后续路线出现就对当前路段生效。已识别的、会丢失必需导航车道的高程歧义会明确拒绝，不用合并不同楼层来补齐地图；多层道路尚未完整支持。

官方终点平面只用于记录真实 Actor 原点是否通过，不附加 Fail2Drive 专用停车墙、终点零速包络或外层制动策略。有限路线之后的已确认公共车道仍作为论文规划的真实几何输入；同车道延续段上的已知停止线继续提供真实 `stop_s` 约束，由论文规划器求解。执行器只跟踪计划，停车保持仅适用于规划明确给出的零速目标，正目标低速行驶不会被改成停车。

不可跨线的硬约束与附加净空分别处理：真实车体不得接触或跨越实线，原净空余量仍进入原平方软代价，不将“仍未压线但不足5厘米”误记为已经违法、从而永久否决所有轨迹。日志分别记录真实线距与软余量亏量。低速暖启动的数值初值比较不因存在远处停止线而禁用，所有候选仍接受同一红灯、车体和动力学验收。QP返回的非有限值、不可行零敏感度约束及明显越出信赖域的步长明确拒绝，不把达到迭代上限视为可接受任意增量。

低速暖启动初值比较在首次QP之前进行，避免QP失败或耗尽预算时跳过恢复；只比较同一参考和原目标函数下的既有初值，不强制起步。B组反思沿用原四部分答案、单引导点和关注对象：输入同帧正常图文、原答案及当前验收失败原因，再通过隔离恢复原求解前状态的同一论文规划器重求解。MPC内部约束保留，返回后不再按安全、紧急或密集复核标志做额外反思准入；只检查同源帧、有限数值和可执行传输契约。若MPC自身返回紧急制动，也照实采用和标记，不冒称正常安全解。反思输出协议错误或求解没有可执行结果时保留原计划。合理等待、低质量分数或QP迭代上限本身不触发反思。没有同帧归档图像的中间规划帧只记录延迟，下一1Hz图像源重新评估，不混用旧图新状态、不额外推进世界。反思端点未单独配置或原私有配置为off时，B组默认复用现有快模型服务并使用独立reflection提示词；实际设置来源、HTTP返回、候选轨迹及最终采纳状态分别归档。查看器按答案角色和精确源帧展示，不把反思答案当作原快答案。此功能不增加速度输出、动作原语或记忆写入。

仅反思检索使用 `--reflection-memory-dir <新库目录>` 或运行配置的 `reflection_memory_dir`；默认null，不读取原有大库，也不从旧 `memory_read_dir` 回落。显式路径不存在、库为空或发布清单损坏会报错；没有合格召回则不补齐记忆，反思继续使用当前图文。库沿用现有记录／离线冻结格式，BEV、OCC和八向各取Top-1并合并同ID，最多3条；只有历史文本进入反思，不附历史图像，不复制历史ID或坐标。范围、表示版本、观测签名和DINO真实权重指纹必须兼容；异常静态实体目前主要由BEV/OCC检索，八向仍沿用原车辆／自行车／行人协议。库在轮初固定只读快照；启动元数据指纹、快照ID、召回分数和实际提供的记忆IDs独立归档，变更库不能恢复旧指纹运行。

## Fail2Drive 单条与完整评测

下面命令在本分支项目根目录执行。专用 CARLA 必须另行在可用 GPU 启动，RPC 为39000、streaming为39001、secondary为39002，Traffic Manager为39003；本入口不会启动模拟器，也不管理旧22000服务。

先做不连接世界的安装与配置检查：

```bash
python scripts/run_fail2drive.py \
  --fail2drive-root /data/zjj/code/EP-Driver/fail2drive-devkit \
  --simulator-root /data/zjj/carla_fail2drive \
  --evaluator-python /home/zjj/miniconda3/envs/ep-fail2drive/bin/python \
  --worker-python /home/zjj/miniconda3/envs/ep-driver-fail2drive/bin/python \
  --routes fail2drive_split/Base_Animals_0075.xml \
  --seed 1 --planner-hz 10 --video --record \
  --output-dir output/fail2drive/B_model_smoke --dry-run
```

单条真实评测使用同一命令去掉 `--dry-run`，不添加 `--duration`。两组分别指定 `--execution-mode A_base/B_model` 与不同 `--output-dir`，其余路线、种子和参数一致。`--video` 将官方 `viz_path_agent.py` 示例相机的真实1920×1080、20Hz画面原尺寸直接单次编码为H.264，并核验解码帧数，不再缩小拼接六路相机。六路模型PNG仍1Hz归档；`viz/`仍按官方示例布局存2Hz展示JPEG，展示相机不进入模型。编码需要现有ffmpeg，失败明确留证，不以插帧补画面。项目视频编码与已有 `ep_agent_viewer` 不属于官方独立网页查看器。

完整运行把 `--routes` 改为 `fail2drive_split`、`--seed 1` 改为 `--seeds 1 2 3`、输出目录改为正式目录：200条原始 XML × 3种子，共600个独立 evaluator 任务。可用 `--route-ids 75 1075` 同时选择一对原始路线进行联调；这是子集测试，不能写成完整基准成绩。

每个 XML 单独以官方 `--repetitions=1` 执行。重复种子与 Traffic Manager 种子不同：后者严格为 `route_id % 1000 + 10000 * repetition_seed`，使 Base/Generalization 配对。成绩保存在 `<方法根>/<种子>/res/####_res.json`；内部规划与视频在同种子的 `agent_runs/####/agent/<路线运行>/planning/`，官方录像在 `recorder/`。可用既有 viewer 只读打开方法根目录；相机与规划以严格源帧关联，未提供标定的相机不伪装为可做3D投影。

`--resume` 沿用官方检查点，要求已启动任务的代码、参数、路线及种子指纹一致；批量中未启动的任务正常新建。更换方法或配置请使用新的方法根目录，不能混写旧成绩。运行失败仍保留官方错误和内部执行证据，不伪造成功。

每条结束后直接打印官方状态、完成率与分数；`entry_status=Finished` 仅表示收尾完成。单条失败或批量中存在失败路线，入口在完成所请求的批次后返回状态码 2；原始官方结果和评分不修改。

完整成绩由原始官方脚本汇总：

```bash
/home/zjj/miniconda3/envs/ep-fail2drive/bin/python \
  /data/zjj/code/EP-Driver/fail2drive-devkit/tools/f2d_result_parser.py \
  output/fail2drive/B_model_formal --method EP-Driver-B --strict
```

当前回归包含真实 CPU DINO、实际数值子进程、本地 HTTP 模型协议替身、执行映射、源帧绑定和公开规则适配；这些不等于部署模型与真实 CARLA 闭环验证。公开停牌和红黄灯已接入；Yield 路权明确标为尚未覆盖，不能声称完整道路规则已验证。

## 测试结果查看器

结果查看器复用 NAVSIM 的 EP Agent Viewer 界面，分支自带部署副本，无需另建环境。它读取正在增长或已结束的 `output/`，不连接 CARLA、不调用模型，不改写官方评分或实验产物。

服务器在本分支根目录运行：

```bash
/home/zjj/miniconda3/envs/ep-driver-fail2drive/bin/python scripts/view_fail2drive.py \
  --root output --host 127.0.0.1 --port 39081
```

本机另开终端建立转发后打开 `http://127.0.0.1:39081/`：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:39081:127.0.0.1:39081 zjj@172.28.133.34
```

可选择路线／seed／运行，拖动、单步或倍速回看录像；新轮默认是官方示例相机原尺寸20Hz视频，旧六相机录像仍兼容并按真实来源标记。“官方示例视角”单独播放已有2Hz展示JPEG，三维场景明确为自定义诊断。模型面板按场景理解、横纵向决策、关注对象和局部引导点排版，失败原因置前，原始JSON和各次尝试折叠保留。官方RC、DS与内部quality分开；语义上下文次数、安全正常行驶的语义启用次数和紧急状态不混称成功。仅在完整物理帧证据一致时同步录像与时间轴；缺少证据时独立播放，不猜测对齐。

可以用 `--root output/某次运行` 只查看一轮。单独授权 `planning/` 时不会读取其外部的官方成绩或展示图。39081是结果回放，39080继续用于原有实时转发。查看器只增加前端读取，不改变评测推进、模型或规划算法。

## 服务器画面转发

`scripts/watch_fail2drive.py` 使用 Python 标准库，在已有 `ep-fail2drive` 环境运行即可。它只读评测 Agent 保存的官方展示相机 JPEG 与同名源帧清单，不连接 CARLA、不新增传感器，也不推进世界。清单在 JPEG 原子落盘后发布，包含实际帧号、仿真时间和图像哈希，转发端不从文件名猜测物理帧。

服务器辅助目录为 `/data/zjj/code/fiil2drive_test/`，私有运行配置为 `fail2drive.server.json`、模型配置为 `models.yaml`、观看入口为 `watch_fail2drive.sh`。实验产物放在本分支 `/data/zjj/code/EP-Driver/EP-Driver-Fail2Drive/output/`；旧私有配置可用 `--output-dir` 明确覆盖。这里的 `fiil2drive_test` 沿用用户指定拼写。脚本本体随 Git 分支维护，辅助入口调用它，避免另存一份实现。

服务器端观看服务：

```bash
conda activate ep-fail2drive
bash /data/zjj/code/fiil2drive_test/watch_fail2drive.sh
```

本机新开终端建立 SSH 转发，然后在浏览器打开 `http://127.0.0.1:39080`：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:39080:127.0.0.1:39080 zjj@172.28.133.34
```

服务只监听服务器回环地址，无需开放公网端口。页面自动发现最新路线，展示实际帧号、仿真时间与距最近保存画面的墙钟时长。实时转发沿用官方示例的每10个物理帧归档一次，即仿真2Hz；同步等待模型或规划时画面可能静止，页面不会把旧帧循环发送冒充实时帧，也不会据此判定仿真故障。无评测运行时显示等待画面。原尺寸官方示例视角20Hz录像由Agent独立保存，结果查看器播放这份视频。

---

# 原 EP-Driver 运行指令与参数

> 当前实现边界：正常行驶候选除软跟踪代价外，仍需满足窄跟踪带硬资格（允许误差为 `0.25m + 初始横向偏差 × exp(-路径进度/15m)`）；制动候选不使用此项硬资格。尚未实现“车道中心仅作为软偏好”的后续调整。CARLA物理仿真效果仍需服务器验证。

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
### 当前基础规划：五次参考与安全优先MPC

参数版本 `development-1.2.0-lane-frenet-apf`。常规尽量保持车道中心；基础规划允许合法、安全的左侧超车、普通右变道和超车后归右。导航必需变道优先于速度收益，交通条件相当时可以归右，不要求先有左变道历史；前后间隙、实线、车体走廊检查及换道完成后的停留时间仍生效。默认只禁止为了绕过当前已限制行驶的慢前车而主动右侧超车，不能把这一规则当作禁止所有右变道。有效模型单目标决定五次参考的经过位置；不改变原帧Frenet语义，不吸附目标，不增加到达时间或停车要求。新目标立即替换旧目标，纵向经过后释放，实际状态再平滑归中。

当前车道属于既定导航时，局部参考保留平台已经构造并核查的完整平滑连接，不再用原生车道中心线重拼而截断环岛入口或出口。普通车道保持直接跟踪这条中心参考，不再把瞬时转角误差外推成30米五次恢复曲线；初始横向偏差由MPC既有的渐进恢复走廊和跟踪代价处理，转角速率与加速度连续性约束不变。主动变道和模型局部引导点仍生成五次参考。非导航邻道仍核验后继拓扑，不直接跳回原导航车道，也不放宽转向角或扩大道路来掩盖转弯失败。

沿车道制动也使用曲率前馈和质心侧偏补偿，避免已进入弯道时因为航向误差变小而把转角回零。有限制动候选先按安全验收、预测碰撞、实际越界越线排序，再比较越停止线状态、速度积分得到的真实制动行程及末速；制动能力相同时优先减小导航横向/方向偏差，不能因偏离导航后Frenet投影前进较少而获选。`braking_risk_metrics`与`braking_risk_order_fields`公开排序证据；没有安全解仍单列紧急状态，不把最低风险候选记为安全解。

跟车使用前后车外廓之间的净间距，期望值为 `mpc.follow_min_gap_m + mpc.follow_time_headway_s * 自车速度`，默认2米加1.2秒时距，例如10m/s时为14米。速度目标结合时距和相对停车距离包络；`mpc.safety_penalties.following_headway`默认20，仅惩罚欠时距，不增加QP硬约束或安全资格门槛，近距和切入后仍可连续减速恢复。真实车体碰撞验收独立保留。`plan.diagnostics.following`记录实际/期望初始净间距和预测最小时距裕量。

跟驰包络只作用于参考走廊内的同向主运动对象（沿参考向前速度大于横向速度绝对值），以及静态/低速阻挡物；预测切入后成为同向前车的阶段仍可加入。移动横穿与对向对象继续参与全部逐时实体避碰和路权审计，不套同向跟车的时距与终端停车包络。`following.candidate_classification`记录逐预测时刻分类，`entity_collision_target_ids`用于核查对象并未从碰撞检查中删除。

道路优先级审计在当前感知候选（车道范围对象与BEV内动态对象的并集）中冻结具有公开车道的车辆/自行车状态，独立于八向代表及模型关注筛选判断让行。规则通路由当前运动方向和公开后继几何确定，不复用EKF短时后继假设；EKF后继仍用于动态避碰预测。`road_rule_observations`绑定原帧、时间和坐标；`rule_observation_scope`分别记录路权对象与MPC优化对象，不读取周车隐藏路线。

对象分为三个集合：①车道范围集L：当前车道及当前位置可跨越同向邻道沿唯一连续中心线链前40米、后20米内的全部车辆、自行车、行人及明确道路障碍；②感知/预测集P：L与本帧BEV视口内全部当前车辆、自行车、行人的去重并集，BEV补充不要求位于可跨越车道；③优化集O：L与模型最多3个关注对象的并集。模型接收P的当前真值和因果历史，P中的动态对象全部进行EKF，静态障碍只保留静态预测与几何。O中全部对象参与势场与实体碰撞约束，范围内被关注者增强原势场，不重复创建对象；L外的BEV对象未被关注时保留预测，不自动进入优化。八向每区域最近代表主要用于检索与角色展示，不再把L裁成最多8个。

轨迹质量评分与执行前审计检查P中的全部预测对象，独立于O的势场/优化约束选择；未关注的BEV对象也不会从独立碰撞审计中消失。正常计划执行前审计报警时重新求制动；无安全解仍按原约定执行单列的最小风险紧急制动，不标成正常安全解。`predictions.jsonl`保存完整P，计划和审计分别记录`prediction_actor_ids`、`optimization_actor_ids`及实际检查对象，便于核对。

`config/parameters.json`的`planning_objects`使用`lane-all-attention-planning-4.0`，默认`front_m: 40`、`rear_m: 20`、`adaptive_front: false`；该范围只定义L，沿允许车道链弧长计算。P的补充范围严格对应实际BEV矩形：自车坐标前70米、后30米、左右各25米，车身部分进入画面也纳入。当前渲染画布固定500×1000像素，修改配置文件中的`bev`数值不会自动改变既有画布。`conflict_horizon_s/conflict_margin_m`仅保留L内运动冲突诊断；它们和旧制动距离参数都不扩大L。感知范围、预测集合和优化集合分别归档，不能把配置频率或局部验收称为全场景安全证明。

MPC时域3秒、控制网格0.1秒、物理子步0.02秒。重规划频率可选1、2、10Hz，默认10Hz；同步模式计算完成后执行前10、5、1段控制。本分支在 `config/parameters.json` 改 `clock.planner_hz`，不提供CARLA运行入口。不要把1Hz误理解为第一项控制恒持1秒。

普通候选必须满足真实尺寸避碰、道路/实线/停止截面、限速、控制边界、控制变化率与跟踪带，并经0.02秒子步复核。未找到普通安全候选则重新计算并验收制动；日志分别保存 `normal`、`safe_brake`、`emergency_unverified`。最后一类按用户确定的实验策略仍执行，但仍可能碰撞，不是安全成功，也不作为合格经验。异常输入仍报错。求解名义预算冷启动2秒、后续1秒、最多10次SQP，不宣称实时达标；同步优先验证规划效果。本地CPU检查可出现数百毫秒单次求解，1、2、10Hz是仿真重规划节拍，整体墙钟耗时必须另测。

四个模型/记忆开关都关时默认仍在1Hz构建BEV、OCC、检索八向和真实DINO，并计算独立无语义参照MPC、质量分和轨迹BEV。`diagnostics.enabled / dino` 为显式诊断开关，默认均true；诊断不打开模型、不读写记忆。真实DINO权重须已下载，不能用随机特征替代。

```bash
python scripts/run_highway.py --scenario highway --seed 1 --duration 30 --fast off --reflection off --no-retrieve --no-write-memory --live --video
python scripts/run_highway.py --scenario merge --seed 1 --duration 30 --fast off --reflection off --no-retrieve --no-write-memory --live --video
python scripts/run_highway.py --scenario roundabout --seed 1 --duration 30 --fast off --reflection off --no-retrieve --no-write-memory --live --video
python scripts/run_highway.py --scenario intersection --seed 1 --duration 30 --fast off --reflection off --no-retrieve --no-write-memory --live --video
```

原生画面青色线为当前实际计划；轨迹BEV与原始检索图分开。`plans.jsonl`保存参考身份、活动目标、筛选与例外理由、约束残差、密集几何检查和执行分类；`summary.json`分别统计制动、紧急、实际碰撞、到达与异常。`timing.jsonl`分别记录EKF、MPC、表示、额外参照、质量、物理步进、显示录像和整体周期；配置10Hz不等于墙钟达到10Hz。新的有效目标可复用移位控制作为数值初值，必须重新积分、计价、验算；不能复用旧目标代价或旧状态轨迹。

本地确定性几何/协议和Highway检查不能代替真实模型或全部场景验收。未验证场景应查看紧急率、碰撞、违规、导航进度和计时，不以程序完成一轮认定规划成功。


近停车制动使用 `stop_control.py` 的离散释放包络，避免将加速度直接截为 `-v/dt` 造成jerk跳变。若当前速度与上一实际制动已导致jerk和非倒退无法共同满足，记录 `braking_release_conflicts` 并归类紧急；不把该冲突伪装为正常可行。

## 3. 配置两个模型

系统提示词在 `config/prompts.yaml` 中修改：`fast.system` 用于快模型，`reflection.system` 用于反思模型，两者独立。使用 YAML 多行文本 `system: |`；程序不对大括号作模板替换。默认回答包含 `scene`、`decision`、`attention`、`local_goal` 四部分，输出协议为 `driving-decision-2.0`；单点Frenet坐标协议仍为 `local-goal-frenet-1.0`。修改后重新启动生效，实际提示词原文及SHA256保存在每轮 `configuration.json` 的 `models` 中。

四部分回答格式（以下是空交通直路示意，数值仍须通过本帧道路校验）：

```json
{
  "scene": {
    "description": "自车沿直路行驶，当前输入没有其他交通对象。",
    "reason": "本车道可通行，已知导航要求继续前进。"
  },
  "decision": {
    "lateral": {"action": "保持车道", "reason": "当前没有换道需求。"},
    "longitudinal": {"action": "维持速度", "reason": "当前速度符合给定速度要求，前方没有已知通行约束。"}
  },
  "attention": [],
  "local_goal": {
    "delta_s_m": 40.0,
    "d_m": 0.0,
    "reason": "沿当前直路中心参考向前引导，保持现有行驶方向。"
  }
}
```

`delta_s_m` 从输入原帧自车投影开始沿冻结参考线向前，`d_m` 相对同一参考线左正右负，均为米；`d_m`不是相对自车的横移。新有效目标在下一次MPC立即替换旧目标；自车中心纵向经过原目标截面后全部释放，即使横向未完成。目标没有到达时间或目标速度，不是停车终点。解释目标所需的参考、合法候选车道及导航规则摘要由程序随原帧提供，不能用BEV直角坐标替代Frenet弧长。

`attention` 每项只含 `id`、`reason`，最多3项；数组顺序由程序转为内部 `rank=1,2,3`。场景、行动和目标理由均不可为空。横纵向行动采用提示词列出的枚举值；这些行动文字用于解释和核查，不直接设定速度、加速度或制动，规划仍使用关注对象与一个几何引导点。完整四部分保存在模型日志的 `answer.model_output`，并记录 `answer.output_protocol_version`。

为读取既有经验，内部 `explanation` 及在线经验写入保留旧存储契约：场景、决策及选点理由由程序映射，适用条件明确标注为程序兼容说明；模型不再被要求生成旧四段说明。已冻结旧提示词仍可返回完整旧三部分回答，日志会标记 `attention-goal-explanation-1.0`，不接受两种协议混合。既有记忆库不自动改写。

本轮修改不自动覆盖实验目录中的配置副本。若命令使用 `--prompts-config "$EXP_ROOT/config/prompts.yaml"`，加载的仍是该实验副本；开始新的实验批次时明确选用新版，B/C等对比组保持同一提示词与协议版本。

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

新八向状态写入 `grid.npz`，保留原始双精度、对象身份和空/未知掩码；不再重复写 `grid.json`。检索对象距离使用数值数组，保持现有部分匹配公式和各通道Top1。旧JSON只读兼容，损坏NPZ不会静默回退。存储格式、3×3×16张量接口和离线库转换要求见 [八向数组存储说明](docs/eight_grid_storage.md)；正式离线库不会自动迁移。

检索或写记忆需要真实 DINOv2 Base 权重。默认使用仓库内的 `models/dinov2-base`，首次克隆后先执行 `git lfs pull`。使用其他本地目录时，在下列命令追加 `--dino-model "模型目录"`。纯基础诊断默认也加载DINO；只有显式关闭DINO诊断且检索、写入均关闭时才不加载。

DINO身份计算先按官方纯重命名规则统一注意力层的`query/key/value/output.dense`与`q_proj/k_proj/v_proj/o_proj`，再校验完整配置及全部张量的数据类型、形状和字节。相同Base权重在已验证的Transformers 5.10.2与5.22.2中得到同一指纹，无需改写当前离线库或重提特征；参数名规范化有冲突则报错。其他未知架构/权重差异仍不能混检。曾保存旧运行时指纹`181f…`的其他库不会自动改标签，须另做真实资产核验；当前正式离线库使用`7fdb…`，保持原文件与审核证据。

当前表示版本为 `bev-occ-grid-2.1`；NAVSIM白底、浅灰道路及对象配色，不绘制车道边界线和分割线，仅保留深灰虚线中心参考路径；历史轨迹采用鲜亮的同色系饱和度渐变。目标参考线与沿线距离标记只进入模型图和回放，不进入DINO特征图或OCC。模型文本包含自车状态、统一候选对象列表、合法目标参考和必要图像说明，命中时附历史情境和参考答案；不发送完整地图边界、历史状态列表或检索四矩阵。目标参考摘要另携带输入帧、仿真时间、参考身份、当前横偏、候选车道及相关导航规则，供正确解释Frenet目标。同一对象真值仅发送一次，提示词聚焦驾驶任务、决策依据和输出协议。

在线新经验须声明 `local-goal-frenet-1.0` 并保存原帧目标参考。离线回顾型经验通过独立只读适配器读取完整文本，不伪造在线模型答案或PDM验收。道路、运动或观测证据不足仅限制对应检索通道，未知保持未知；原帧、时间、资产哈希、版本、DINO身份及人工审核绑定仍须通过检查。未审核的`pending`记录不进入快照，冻结离线库不能作为在线写入库。被排除原因可在轮初 `memory_snapshot.jsonl` 和 `configuration.json` 查看。

BEV、OCC、八向分别在自身有效候选池中取最相似的一条，八向分数至少0.2的门槛仅约束八向召回；同一经验ID合并，最多返回3条，不补齐，不加权融合。某通道证据不足不会连带排除其他有效通道；有效0分与无证据分开记录。确认空场景可以是有效证据，遮挡未知不能当成空场景。以下参数在 `config/parameters.json` 修改，并随每轮配置保存；参数签名不兼容的通道不会静默混算。

| 检索参数组 | 开发默认值及含义 |
|---|---|
| `retrieval.minimum_grid_score` | `0.2`，仅限制八向通道，不阻止有效BEV或OCC参与各自排名 |
| `retrieval.require_all_channels` | 固定 `false`，各通道独立准入；旧配置显式设为`true`会报错 |
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

本版取消加权融合、融合分门槛和最终Top-K重排；旧 `weights / minimum_score / final_k` 配置会明确报错，请使用以上字段。`retrieval.jsonl`保存各通道有效性、排除原因、第一名及`matched_channels`，没有`fused`分数。检索策略版本为`bev-occ-eight-channel-top1-3.1`；允许只读加载3.0/2.0的兼容资产并按当前策略重算，不改写旧特征、教师回答或审核证据。各通道资格由`representation_channel_eligibility()`实际计算，不使用空函数或缺失值补零。仅自比较有效不代表跨场景一定命中。

以上为开发起点，不是已优化的检索参数。禁跨邻道及40米/20米范围外的对象不进入模型数值真值或MPC对象集合；原始快照、BEV/OCC图像上下文和物理世界保留，不能据此把其数值重新加入规划。检索八向使用自身现有范围与准入规则。

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
| `local_goal.longitudinal_scale_m / lateral_scale_m` | `30 / 15`米，同一冻结平滑车道参考上的弧长/横偏尺度 |
| `local_goal.potential_shift` | `6`，吸引能量减去的常数，不改变梯度或最优解 |
| `risk.classes.*.longitudinal_m / lateral_m / peak` | 机动车`20/6/8`、自行车`10/3/12`、行人`10/10/16`、道路障碍`12/4/8`；前两项为椭圆半轴（米），末项为无量纲中心峰值 |
| `risk.attention_increment` / `mpc.weights.risk` | `1 / 2`，关注倍率为`1+1/排名`，外围风险权重独立乘2 |

排斥场用自车中心与对象中心在同一冻结平滑车道Frenet参考上的差构成椭圆半径`rho`；`rho<1`时使用`S=1-10rho³+15rho⁴-6rho⁵`，中心到边界连续衰减，边界外为零。椭圆沿车道弯曲，不再按实体边界净距或矩形平台构建势场。势源取优化集O，即车道范围L内全部对象与模型最多3个关注对象的并集；L外关注对象允许向合法自车域内产生排斥影响。L内关注者按原排名增强，同ID只计算一个势源。场仅在该合法连续车道走廊内有定义；域外标无效，不能作为零成本捷径。

吸引项在目标冻结的平滑Frenet参考上比较弧长/横偏，在原协议首次上穿截面位置或未到达的末端只评价一次，并减常数6；目标替换、释放、失效协议不变。规划风险按预测时刻对齐，逐时刻求和再平均；二维叠加图显示空间势值，不等于将固定目标吸引在每个节点重复累加。完整车体碰撞、道路及不可跨越线约束仍保留，旧倒距离软势与线类软势已移除。峰值有限且中心梯度为零，平滑场本身不保证避碰。

## 9. 采集正式场景的离线记忆素材

`scripts/capture_highway.py` 读取与实验相同的参数和场景配置，由当前基础规划器实际驾驶。采集种子应与正式评测分开；不要将正式评测轮次用于补充其检索库。

在项目根目录的 Bash 终端执行，例如高速场景采集5轮、seed 1001–1005、每轮最多30秒：

```bash
python scripts/capture_highway.py \
  --scenario highway \
  --config config/parameters.json \
  --scenarios-config config/scenarios.yaml \
  --episodes 5 --seed 1001 --duration 30 \
  --output-dir "E:/datasets/highway_env/ep_driver_memory/captures/highway"
```

如果实验使用了冻结配置，将这两个配置路径替换为对应文件。其他场景将 `highway` 换为 `merge`、`roundabout` 或 `intersection`，输出目录也相应修改。追加 `--live` 可以观看；默认不打开窗口。每次运行创建独立批次，不覆盖已有采集。

每轮沿用离线工作台的原始输入格式：

```text
<批次>_<场景>_seed<种子>/
  run.json             # 配置、种子、来源和完成状态
  frames.jsonl         # 当时可观测的场景、实际执行状态与审计
  future_states.jsonl  # 独立保存完整对象真值，仅供离线审计
  native/*.png         # 原生RGB画面，无规划轨迹叠加
```

按真实控制步保存图像和状态（当前为10Hz），包括初帧和实际终帧；1/2/10Hz重规划均保留这些控制步。发生碰撞或到达终点会提前结束，不能把实际执行轨迹自动当作正确示范。中断或失败的轮次标记 `complete=false`，离线工具会跳过。

采集保留主规划、预测和执行安全审计；不调用视觉模型、检索或写记忆，不计算DINO、BEV/OCC/八向特征、额外参照规划或轨迹评分。无需模型配置和密钥，也不生成视频。

采集完成后，在现有离线工作台的整理/`prepare` 步骤直接选择 `captures` 目录；后续特征、教师总结和人工审核沿用原流程，无需转换格式。


### 在线对象范围与模型图像（2026-11-22）

车道范围L继续采用`lane-local-object-scope-1.0`和`lane-topology-role-assignment-1.1`：当前物理车道及当前位置可跨越的同向邻道，沿公开导航及唯一前后继链计算前40米、后20米。对象中心决定前后弧长范围，真实外廓决定是否占用车道物理宽度带；不使用圆形距离。道路可跨越证据只授权自车行驶，不限制BEV内来车进入感知集合。

`lane-plus-bev-dynamic-perception-1.0`定义感知/预测集P＝L＋实际BEV视口内全部当前车辆、自行车、行人，按ID去重。BEV范围固定前70米、后30米、左右各25米，随自车姿态旋转；以车身与矩形窗口相交判断，包含边缘部分可见对象。模型文字和关注白名单使用P，动态对象全部进行EKF；只提供当前真值和因果历史，不读取隐藏路线或未来，不判断遮挡，不能将其解释为真实传感器感知。

`lane-all-attention-planning-4.0`定义势场与碰撞优化集O＝L＋模型最多3个关注对象。L内全部对象参与，不只八向最近代表；L外BEV对象被关注后进入O，L内被关注则按原排名倍率增强势场。L外未关注对象仍预测，质量评分及执行前审计检查完整P。八向槽外关注对象不伪造车道角色，也不授予逆行或跨实线权限。

CARLA围栏、护栏、墙、杆、灯牌不当作交通障碍物。`Static/Other`及任意家具不因语义标签自动准入；施工锥、临时路障、掉落物须精确类别证据且外廓进入L。地图停放车保持车辆类别；交通灯、停止线和公开路权作为规则保留。

检索八向仍按既有前70米、后30米和拓扑规则构建，在线40米/20米关系用于角色展示与规划范围说明。BEV/OCC/八向编码及三通道检索算法、距离指标和排名规则不变。模型BEV稀疏ID标签扩展到P的动态对象，仍受标签数量和防重叠限制；未画出ID文字不代表该对象未入模，实际列表以结构化输入为准。标签只进入模型图，无字DINO图不变；本次不要求整体迁移或重算既有离线记忆库，新经验正文可记录扩展候选。

highway-env每个低频模型源帧保存无规划叠加的原生俯视图，输入顺序为原生俯视图、BEV；前者世界x向右/y向下，自车不固定朝上，后者自车朝上。CARLA保持同帧六RGB拼图、BEV。所有图像绑定同一冻结帧，采图不推进仿真。`model_inputs.jsonl`记录实际入模ID、车道范围ID、BEV动态ID及范围外补充ID；完整原始快照独立留存。

本地离线检查：`python scripts/regression_model_perception_scope.py`、`python scripts/check_lane_object_scope.py`、`python scripts/check_model_object_scope.py`、`python scripts/check_highway_model_camera.py`。CARLA另有`python scripts/check_carla_object_categories.py`；离线通过不代替真实模型或服务器闭环实验。
# 官方原场景专家示范采集

按官方原始场景收集经验时，使用 `scripts/capture_fail2drive_official.py`。
它复用锁定版本的官方 evaluator、原始路线 XML、定制 CARLA 资产、地图、自车车型、
天气、周车脚本、事件条件和随机种子，只把自车 Agent 替换为独立的官方 AutoPilot
专家适配器；不修改官方源码、评分或60秒清障等待，不使用下节的 Town05 代理场景。
evaluator 独占世界推进；采集 Agent 不创建或删除周车、不修改灯态、不启动 CARLA。
专家的场景记账及随机数隔离，禁用原专家删除异常周车和卡住后强制油门的行为。
这是程序专家示范，不标作真实人类驾驶；同一行为程序中的周车也会因自车行为不同而
改变触发时刻及位置，不能承诺逐帧轨迹相同。

`config/fail2drive_official_demonstrations.json` 默认选择每类一条原始 Generalization
路线，共17条；`--list` 可直接列出。需要与某条实际测评路线完全对应时，使用
`--route-ids <原始路线ID...>`，并采用相同 seed；一类的一条示范不代表该类所有路线。
`--family BadParking` 选单类，`--all-families` 选17条。入口默认只预检并打印计划，
缺少原始源码、速度数据、定制模拟器或SDK身份时直接失败，只有显式 `--run` 才启动
官方 evaluator。必须先由用户启动专用官方 CARLA；不新建Python环境。
每次使用本分支 `output/official_demonstrations/` 下的新目录，保留原路线、源码、
实际车辆、配置、seed、命令和文件哈希，拒绝覆盖旧采集。

采集保留1Hz真实六相机拼图、各相机原始同帧外参、当前公开状态，以及20Hz实际自车
状态/踏板和1920×1080官方示例视角视频。公开模型输入不包含专家读取的隐藏场景记账。
`scripts/build_fail2drive_official_memory.py finalize --capture <实际轮次目录>
--checkpoint <官方结果JSON> --launch <本轮launch.json>` 在官方结算后核验并导出。
只有完整官方路线成功、没有官方违规、Agent/资源及视频检查通过的采集才能进入
`check` / `prepare`；保留失败证据，不能改成成功。官方整轮评分不是逐物理帧独立安全证明。

`prepare --captures <已finalize采集目录> --output-dir <新build>
--workbench-root <已有offline_memory_builder目录>` 复用原工作台，按真实制动、转向和
重新起步选择最多3个具有后续3秒证据的锚点；仍需真实DINO、部署教师标注、人工审核
和冻结发布。正式论文 Agent/MPC 不调用专家；新库仅由显式反思检索入口读取。
在测评原路线采集再辅助同场景测评，应披露为已知场景示范辅助，不能当成未见路线
零样本泛化成绩；也不保证检索到一条记忆就一定通行。

# 旧独立长尾示范采集（代理开发数据）

`config/fail2drive_demonstrations.json` 为17类机制各定义一项独立设计；配置中的
`pending_execution` 不是成功经验。使用标准CARLA资产和独立开发地图，不读取正式
Town13路线或未来事件答案。动物、物体图案等代理的视觉差异保留在每项 `proxy` 中，
不能据此宣称复现官方异常外观或完全未见机制泛化。

`scripts/capture_fail2drive_demo.py list` 列出设计；`check --scenario Wall` 只核查配置，
不连接CARLA。实际 `capture --scenario Wall --worker-python <现有数值Python>` 由现有
CARLA SDK环境执行，只接受空闲专用世界；需要换图时显式加 `--allow-map-load`。
默认录制1920×1080、20Hz示例视角视频，1Hz保存严格同帧六相机与状态、20Hz保存
实测控制和独立物理审计。所有产物只进入本分支 `output/demonstrations/`，不覆盖旧轮次。
全封闭类保留60秒清障等待。教学参考与跟踪器仅用于离线采集，正式Agent/MPC不调用它。

地图条件、蓝图、合法绕行或实测净空不满足时拒绝采集；碰撞、离路、规则违规、
交互未完成或恢复行驶不足时不得入库。设计核查和离线运动学测试均不是CARLA成功证明。

采集完成后，在现有数值环境运行 `scripts/build_fail2drive_demo_memory.py check
--captures <采集目录>`；通过后使用 `prepare --captures <目录> --output-dir <新build>
--workbench-root <已有offline_memory_builder目录>`，每轮选择1～3个具有真实后续3秒
证据的事件锚点。继续复用该工作台的真实DINO提取、部署教师标注、验证、人工审核和
冻结发布流程，不自动伪造审核或成功记录。新冻结库通过读取器核验后，显式配置
`--reflection-memory-dir <新库>`；普通模型调用仍不检索，不回退到旧库。
