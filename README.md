# EP-Driver-NAVSIM：特权输入基线与官方评测

本分支从 `EP-Driver@f91af2d` 创建，共享算法与模型提示同步至 `EP-Driver@6b4853d`，包含 `development-1.3.0-bev-perception-lane-all-planning` 对象范围与统一候选对象输入更新，复用五次参考、Frenet APF、MPC、BEV/OCC、八向张量、DINO、参照规划与质量诊断。下面的 Highway 指令保留用于原算法回归；NAVSIM 使用 `scripts/run_navsim.py`。

NAVSIM 按 `--benchmark` 选择固定的官方协议，运行未经修改的官方缓存和评分代码：

| benchmark | 官方版本与场景集 | 评分入口与结果 |
|---|---|---|
| `v1_navtest` | v1.1分支，`3e8291bfa89ff247231e0227778840cd0a036896`，navtest | `run_pdm_score.py`，PDMS，单阶段 |
| `v2_navtest` | v2.2，`359c7f72304bfa8273e754224a213d3751bd2340`，navtest | `run_pdm_score_one_stage.py`，EPDMS，单阶段 |
| `v2_two_stage`（兼容默认） | 同一v2.2，warmup/navhard | `run_pdm_score.py`，两阶段响应交通EPDMS |

v1/v2 navtest均使用官方12,146个锚点、非反应式周车；两版本SDK目录、运行环境、评分缓存及输出分开，同版本A/B共享缓存。`prepare_navsim_runtime.py --profile server --benchmark v1_navtest --groups A_base B_model ...` 生成对应配置，v2用 `v2_navtest`；默认服务器v1 SDK为 `navsim-devkit-v1.1` / `ep-navsim-v1`，v2沿用 `navsim-devkit` / `ep-navsim`。入口检查SDK源码指纹、实际导入路径、完整样本及缓存覆盖。v1结果读取官方 `average` 行，v2单阶段读取 `average_all_frames`；未定义的EC保留为空，禁止补分。两阶段聚合继续由原官方入口完成，不能用单阶段均值代替。结果记录显式版本、交通策略、模型有效参与和回退数量，查看器按协议显示指标与阶段。

服务器模型同时接收八路当前相机和一张BEV，共九图；配置生成器默认输出额度为4096 tokens，本轮navtest正式配置使用16384 tokens、128000上下文。服务的上下文还需容纳图像和完整文本输入，不能把4096当作整个上下文大小。本次九图实测输入已超过16K；服务容量应在正式评分前以真实请求验证，`model_service`元数据必须与实际启动参数一致，不能靠删图或静默裁剪输入绕过容量不足。NAVSIM结构化服务使用greedy解码，说明字段最多192字符；vLLM专用服务启用`--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'`，限制JSON字段间空白，避免greedy空白退化耗尽输出额度（字符串正文不裁剪）；目标横向范围由同帧合法候选车道几何给出保守外包，不替模型选目标、不夹紧返回值。没有合法语义参考或纵向目标区间时，保存原因并跳过HTTP，继续原基础MPC；按锚点记录实际调用尝试次数，这类回退不能作为成功模型样本复用。

**输入条件是 privileged baseline，不属于标准 sensor-only 输入条件。** 规划只访问当前及历史四帧（约1.5秒）的对象标注、自车状态、当前地图/roadblock 导航、当前红灯标记与八路当前图片；不访问未来自车轨迹、未来对象标注或评分缓存。模型与预测候选为当前/可跨越邻道沿线前40米、后20米内全部对象，与当前BEV视口内车辆、自行车、行人的并集；优化使用道路范围全部对象与最多3个关注对象，独立审计保留完整候选预测，八向代表主要用于检索。一般家具类 `generic_object` 排除，锥桶、施工标志和隔离障碍作为候选，仍需通过道路范围筛选。NAVSIM 提供相邻拓扑，未提供完整车道线跨越权限；本基线明确记录“同向相邻拓扑视为可跨越”的适配假设，不能把它写成真实标线真值。红灯布尔值的反值记为 `not_red`，不虚构绿/黄灯细分。

每个官方评测锚点独立重置规划器和预测状态，不跨锚点延用计划。为满足官方输出，NAVSIM 单独使用4秒、0.1秒控制网格，输出未来0.5至4秒共8个**当前后轴坐标系**位姿；先将核心质心轨迹按 Pacifica 轴距和后轴偏移换回后轴，不能直接裁剪/补齐原3秒轨迹。内部质量分和预测执行审计与官方 EPDMS 分开保存。若启用模型，模型失败时基础规划回退及错误会留证。

NAVSIM 的 `navsim_execution.postsolve_safety_policy=audit_only` 保留 MPC 的避障、道路、动力学约束及势场代价，但取消求解后因独立安全诊断不通过而再次替换制动轨迹。完整4秒诊断保留碰撞、离路、越线、限速和控制约束结果；诊断未通过但可执行的候选标为 `diagnostic_unverified`，不能称作安全解。控制种子和优化步必须先满足加速度、转角、非倒车、jerk、转角速率和质心横向加速度的可达限制，再重新积分轨迹；观测初态已不可行时保留受速率限制的恢复证据，不伪造瞬时合规。初态超速时，QP硬限速采用观测速度、加速度及jerk限幅推导的最早合法恢复包络，真实限速诊断和软罚不变。未收敛且约束残差过大的QP步不作为有效优化步。只有没有数值可执行候选时才使用最小风险制动，并如实记录 `emergency_unverified`。交通实体额外保留0.75米执行跟踪裕量，道路优化保留0.2米跟踪预留；真实道路几何及独立离路诊断不变，预留不足单列记录。这些预留不是已证明的误差上界。此策略仅适用于 NAVSIM，不改变 Highway/CARLA 的执行策略。

首个QP步因数值残差不合格而被拒绝时，NAVSIM最多尝试一次已有种子的重新初始化：按同一MPC目标函数比较数值可执行的导航种子和制动导航种子，共享原有时间预算与迭代上限。选择不读取安全诊断结果，未收敛的种子不会冒称QP优化成功。

路口重叠区域分别保留原始几何占道推断和可向前延续的公共导航锚点；参考、对象范围、模型目标坐标及APF使用同一初始化恢复走廊。恢复只使用当前状态、已观察历史及有向地图，不读取未来轨迹。硬输入核查覆盖有限坐标、地图、有向拓扑及端点身份；参考线的整车越界、转角可行性、前后覆盖不足和远期域分片另记为`reference_quality`，不能代替规划后的安全否决。优先使用可行恢复参考，否则明确标记`reference_only_unverified`，由原MPC优化，不冒称安全轨迹。普通导航接缝只在拓扑、中心和附件自洽时构造窄连接，并裁剪到官方道路内；不把整个路口授权为势场域。不明接缝保留缺口及原因，不授权跨越它的模型目标；邻道未来连接失败不删除当前位置的合法邻道对象。NuPlan矢量车道边界记为虚拟边界和未知标线，不能当作地图提供的不可跨越实线。

首次准备两个独立环境：现有 EP-Driver Python3.12 数值环境，以及官方 SDK Python3.9 环境。按 [NAVSIM 官方安装说明](https://github.com/autonomousvision/navsim/blob/v2.2/docs/install.md)取得上述固定版本 NAVSIM 与 nuPlan v1.2；SDK 环境使用 `config/navsim/sdk-requirements.txt` 的推理/评分依赖，并将两份 devkit 以 `pip install --no-deps -e <路径>` 安装。Linux 为官方支持环境；本机 Windows 短测仅增加 Linux pickle 路径兼容和无下载的 `fcntl` 导入保护，评分源码/公式不变，不能据此声称完成 Linux 全量验证。缺少地图而需要 POSIX 文件锁时会明确失败。

复制 `config/navsim.example.json` 为被忽略的 `config/navsim.local.json`，填写 SDK、两个 Python、数据与真实 DINO 权重目录；不得使用 LFS 指针代替权重。原始输入来自 `navsim_logs/test` 和 `sensor_blobs/test`，仅两阶段协议的合成后续输入来自对应 `*_two_stage` 目录；navtest不需要合成场景数据。入口将地图矢量文件校验后复制到本地 `outputs/navsim/maps_runtime`，官方临时文件不写入原数据目录。个人配置和运行产物不提交。

在本分支根目录、数值环境中逐条运行。`test/` 中的回归脚本仅保留在本地开发工作区，不随分支发布；服务器可跳过该行，正式运行入口保留在 `scripts/`：

```bash
python test/check_navsim_integration.py
python scripts/run_navsim.py check
python scripts/run_navsim.py smoke --output-dir test/artifacts/navsim/smoke
python scripts/run_navsim.py cache
python scripts/run_navsim.py evaluate
```

默认 `warmup_two_stage` 含16个原始锚点和204个合成锚点。`smoke` 只对其中原始/合成各一个执行官方模拟和评分，明确不产生完整两阶段成绩。正式 `navhard_two_stage` 含450个原始锚点和5462个响应合成锚点，运行 `check/cache/evaluate` 时均加 `--split navhard_two_stage --output-dir outputs/navsim/navhard/A_base`。不能只评450个原始锚点后声称完成两阶段官方测评。

`check/evaluate` 在长时间评分前检查所选集合的全部当前位姿、地图和导航连接，结果保存为 `navigation_preflight.json`。每个样本的成功或失败即时落盘，未检查完的报告不会标作通过；复用结果必须匹配实际适配源码、官方SDK、地图、几何依赖、完整样本集合及当前位姿/公开路线。导航检查缓存位于 `<metric_cache_path>_ep_navigation_preflight`（与官方缓存并列，避免干扰首次创建），属于本项目检查元数据，不替代官方评分缓存。`navigation_preflight_workers` 默认1；生成的服务器配置为8个独立进程，Windows仍串行。它只并行地图检查，不调用模型、规划器或评分器。

适配器在每个独立锚点调用固定版本 PDM 的当前位姿/地图拓扑纠正函数，既可补接公开导航首段前驱，也可修复中间连接。重叠路口先按原有航向与投影门槛筛选，再比较覆盖和投影距离。若纠正结果丢失有效的当前关联，仅在公开路线保持顺序、通过有界官方BFS补齐缺链并逐段验证有向连通后保留公开路线；纠正尝试和拒绝原因均归档，不移动实际自车位姿。参考路径的前向长度不包含自车后方弧长。证据写入输入包的 `route_association`，不读取未来轨迹或评分缓存。车道交集中的零面积线/点从势场面域中排除，不缓冲扩大车道。任一样本预检失败会停止，不以删样本或替代分数完成评测。

`smoke/evaluate` 默认开启 `bev_video: true`（命令行 `--bev-video`）；逐样本保存 `samples/<摘要ID>/official_replay/official_bev.mp4`、首帧图、压缩执行记录及独立导出状态，完成后打开输出目录下 `visualizations/index.html` 选择播放。动画使用本次官方控制器实际模拟的自车轨迹和本次官方交通策略的对象轨迹（v1取官方评分占用序列，v2取所选策略的模拟返回），叠加提交规划、记录人类参考和官方红灯区域；不是把数据集未来帧冒充算法执行。原始事件诊断与人类违规过滤后的官方整段指标分开显示，不虚构逐帧 EPDMS 或完整两阶段成绩。0至4秒的41个实际状态按10帧/秒编码，视频编码时长4.1秒；不同锚点不拼成连续驾驶。

可视化在评分侧旁路采集，不把未来状态送回规划器，不修改官方SDK、响应策略、评分公式或聚合。人类参照评分前先复制算法对应的执行数据，避免被第二次评分覆盖。视频失败单列状态，不把官方分数改成失败或成功；`--no-bev-video` 可关闭导出。导出时间独立记录，不计作规划耗时。修改代码后使用新的输出目录，正在运行的旧评测不会自动获得动画。

对已保存的执行记录可独立重绘，不运行规划器或官方仿真；只有旧CSV/静态规划图而无执行记录的实验不能恢复响应交通动画：

```bash
python scripts/run_navsim.py render --output-dir outputs/navsim/warmup/A_base_bev
python scripts/run_navsim.py gallery --output-dir outputs/navsim/warmup/A_base_bev
```

`render` 可加 `--token <原始token>` 只重绘一个样本；`gallery` 可在评测运行期间刷新播放列表，既有视频不重算。默认绘制由官方 SDK 的地图、标注框、轨迹与相机绘图函数完成，导出状态保存 SDK 绘图源码哈希及 `renderer_version=navsim-sdk-plots-1.0`。旧版自定义视频不会被标作新版官方样式。源锚点八相机只表示输入时刻；动态自车与周车始终取官方执行记录，不以记录的人类未来画面替代算法执行。

默认视频导出需要支持 `libx264` 的 FFmpeg，置于 PATH 或用 `EP_NAVSIM_FFMPEG` 指定可执行文件。服务器查看既有视频不需要 SDK；从其他电脑迁来的归档若需重绘，应在 `ep_driver_navsim.replay_render` 入口显式覆盖 SDK、Python、地图和数据路径，不能直接使用原电脑的绝对路径。

当前数值规划实际使用40段0.1秒控制，主规划和独立参照均覆盖4秒。最终提交0.5、1.0、…、4.0秒的8个后轴位姿，拒绝短轨迹外推或补齐。`planning_evidence.json` 中的 `horizon_contract` 可核查主时域；4.9秒预测只用于4秒轨迹及后续0.9秒TTC诊断，不表示额外提交时长。

`evaluation_workers` 控制官方评分的独立进程数（1–8，默认1；服务器配置建议4），保存在冻结配置中。Windows 保持串行；Linux 并发使用官方进程池及 `fork` 继承旁路可视化观察器，不能改为共享可变评分器的线程池。缓存生成仍串行，评分公式与聚合不变。少量并发验证只检查接入，不产生完整基准成绩。

默认 `resume=true`。完成样本以最后原子写入的 `sample_complete.json` 为准，复用前核对数值源码、参数、模型版本和预处理、提示词、记忆快照、实际相机内容及必要资产哈希；只有 `trajectory.json` 不足以认定完成。模型组需提供明确 `model_revision`，未通过回答校验的基础规划降级样本不复用。不固定版本的API可显式设 `resume=false`。续跑仍会重新执行官方评分和可视化，不会把旧得分套给新输入。

新输入包无损保存为 `input_packet.json.gz`，读取兼容旧JSON；命中记忆保留文本、来源、分数与资产引用，表示数组留在记忆库，完整候选诊断另存 `retrieval_diagnostics.json.gz`。检索和模型输入算法不因归档压缩改变。

服务器迁移使用独立 Python3.12 数值环境与既有 Python3.9 SDK 环境。`scripts/prepare_navsim_runtime.py --profile server --model-revision <权重及配置指纹>` 可生成被忽略的A/B/C个人配置，不安装依赖或启动测评。`--profile local --base-url http://127.0.0.1:18100/v1` 可在本地通过模型端口转发先做集成检查。生成的服务器配置声明9图、每图602112像素处理上限和16384上下文，须与实际模型服务保持一致；所有模型组使用相同模型与图像规则。

生成的服务器模型配置开启 NAVSIM 专用 `structured_model_output`：请求 JSON Schema 将对象 ID 限于本帧合法集合，保留原有四段输出协议和目标点道路/几何验收。服务不支持结构输出时明确报错，不悄悄退回无约束请求。回答语法正确也可能因目标不可达而被拒绝，失败原因与基础规划回退会独立归档。

`scripts/export_navsim_test_assets.py` 使用固定官方筛选器生成原始test日志与当前八相机迁移清单，原日志完整保留，不重写或裁掉未来标注。默认仅清单，显式 `--tar` 才打包；迁移后 `--verify-manifest ... --verify-root ...` 逐文件校验。地图与已存在合成数据不重复打包；清单不代表服务器数据已完整。

本分支 `tools/ep_agent_viewer` 是现有独立只读查看器的可部署副本，NAVSIM每样本单独展示，官方视频为默认主画面，三维与曲线明确作为非官方诊断。源图不借给未来帧，无标定不投影。修改工作区原版查看器后用 `python scripts/sync_navsim_viewer.py` 同步，并可加 `--check` 核对副本。服务器启动示例：

```bash
python tools/ep_agent_viewer/run_viewer.py --root /data/zjj/code/EP-Driver/experiments/navsim --host 127.0.0.1 --port 8765
```

本地另开终端执行 `ssh -N -L 8765:127.0.0.1:8765 zjj@172.28.133.34`，浏览器访问 `http://127.0.0.1:8765/`。查看器不启动模型、规划器或模拟器，不修改测评证据。

默认 `fast: off, retrieve: false`；B组设置 `fast: server` 或 `api` 并指定个人模型配置，C组再启用 `retrieve: true` 和只读记忆库。沿用现有 `MemoryStore` 及其离线库只读适配，库必须通过现有资产/特征协议与新增来源审计，不自动迁移旧库。本地A/B/C已分别用原始、合成各一个锚点完成集成检查，B/C实际请求九图服务器模型，C仅使用三条真实记忆的隔离小快照；当前几何验收拒绝的模型目标仍记录为基础规划回退，不算有效模型回答。这些短测不代表完整正式集或完整记忆库的实验结果。全部组关闭在线写入和反思，启动时冻结记忆快照并在数值进程复核，拒绝评测同log/同token记忆及来源不明的真实数据记忆；不同组应使用不同输出目录。输出保存冻结配置、输入包、图像/表示、轨迹及诊断、官方原始CSV和完成状态。修改数值源码或参数后需新输出目录，避免混合实验。

---

# EP-Driver 原有运行指令与参数

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

参数版本 `development-1.3.0-bev-perception-lane-all-planning`。常规尽量保持车道中心；基础规划允许合法、安全的左侧超车、普通右变道和超车后归右。导航必需变道优先于速度收益，交通条件相当时可以归右，不要求先有左变道历史；前后间隙、实线、车体走廊检查及换道完成后的停留时间仍生效。默认只禁止为了绕过当前已限制行驶的慢前车而主动右侧超车，不能把这一规则当作禁止所有右变道。有效模型单目标决定五次参考的经过位置；不改变原帧Frenet语义，不吸附目标，不增加到达时间或停车要求。新目标立即替换旧目标，纵向经过后释放，实际状态再平滑归中。

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

以上为开发起点，不是已优化的检索参数。模型当前数值真值与关注白名单使用感知集P；MPC优化集O由车道范围L与最多3个模型关注对象组成，L外未关注的BEV对象仍保留预测并参与独立审计。对象输入不改变自车合法可行驶域，检索八向使用自身现有范围与准入规则。

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

本地离线检查：`python test/regression_model_perception_scope.py`、`python test/check_lane_object_scope.py`、`python test/check_model_object_scope.py`、`python test/check_highway_model_camera.py`。回归脚本仅保留在本地 `test/`，不随分支发布；CARLA专属对象分类检查在对应开发工作区执行。离线通过不代替真实模型或服务器闭环实验。
