# EP Agent Viewer

独立、只读的驾驶智能体展示前端。读取正在增长或已经结束的实验目录，不修改 EP-Driver / EP-Driver-CARLA，不连接 CARLA，不推进仿真，不调用模型，不写入实验产物。前端资源全部本地提供，无 CDN、npm 构建或外部服务要求。

## 启动

使用现有实验 Python 环境（Python 3.10+，建议沿用项目的 Python 3.12 环境）。

```powershell
python "D:/code/AutoDriver_Agent_memory_system/Multi-level Memory System for Intelligent Driving/ep_agent_viewer/run_viewer.py" `
  --root "D:/实验数据/第一篇论文实验数据/highway-env/正式实验_20261119/A/roundabout" `
  --engine-root "D:/code/AutoDriver_Agent_memory_system/Multi-level Memory System for Intelligent Driving/.worktrees/EP-Driver" `
  --open
```

`--root` 可以是单轮目录，也可以是含多轮结果的父目录。浏览器默认地址为 `http://127.0.0.1:8765/`。增加 `--follow` 默认跟随最新规划；可在页面暂停跟随并回看。`--port` 可修改端口。停止展示服务使用 Ctrl+C，不影响实验。

CARLA 使用相同启动方式，把 `--root` 指向 CARLA 运行结果，把 `--engine-root` 指向与该运行匹配的 EP-Driver-CARLA 源码目录。`--engine-root` 只读加载道路几何和势场函数，不启动主程序。

已有环境缺少依赖时，只需安装本目录 requirements.txt。HTTP 服务及 JSONL 读取使用 Python 标准库；几何和势场使用 NumPy / Shapely / SciPy。不需要 CARLA SDK、Torch 或模型权重。

## Fail2Drive 连续路线结果

Fail2Drive 分支提供相同界面的部署副本 `tools/ep_agent_viewer/`，入口为 `python scripts/view_fail2drive.py --root output --host 127.0.0.1 --port 39081`。沿用现有数值环境即可，不需要 CARLA SDK 或模型服务。`--root` 可以是包含多轮的 `output/`、一轮方法根目录或单个 `planning/`。逐条官方路线独立选择，不跨路线串接播放。

默认播放已归档的六路真实 CARLA 相机录像；“官方示例视角”读取沿用官方示例相机配置的 JPEG 和同名帧清单，不是重新生成画面。2Hz 展示相机与1Hz模型输入相机分开标明来源和时间。三维视图明确为自定义诊断（非官方）。查看器本身是项目工具，不冒称官方独立查看器。

官方 RC、DS、违规系数及违规原文与内部 quality 分开展示。只在路线ID、seed、归档配置及运行时序均匹配时关联官方检查点；`Finished` 不能单独表示完成路线。结果缺失、过期或不能确认归属时显示不可用。只授权单个 `planning/` 时不越界读取父目录的成绩、执行或展示相机文件。

视频只有在保存帧数、首末物理帧、解码结果及每帧执行时间连续一致时才与规划时间轴同步，正确保留非零仿真起点。证据缺失时作为独立录像播放。模型面板展示每个源帧的真实七张输入图、完整回答、每次尝试和错误；`finish_reason=length` 明确标记输出达到长度上限。历史回答仅按规划的 `answer_source` 关联。

服务器监听回环地址39081；本机使用 `ssh -N -L 127.0.0.1:39081:127.0.0.1:39081 user@server` 后打开 `http://127.0.0.1:39081/`。这个结果回放入口与39080实时画面转发、NAVSIM及离线记忆工作台独立。

## NAVSIM 独立锚点测评产物

可以直接将 `--root` 指向 NAVSIM 的整轮输出、包含多轮的父目录，或单个 `samples/<digest>`。无需 `--engine-root`：

```powershell
conda activate mulmem_highway
python "D:/code/AutoDriver_Agent_memory_system/Multi-level Memory System for Intelligent Driving/ep_agent_viewer/run_viewer.py" `
  --root "D:/code/AutoDriver_Agent_memory_system/Multi-level Memory System for Intelligent Driving/.worktrees/EP-Driver-NAVSIM/outputs/navsim/warmup/A_base_adapter_fixed" `
  --host 127.0.0.1 --port 8765 --open
```

每个 sample 作为独立 episode，播放到自己的结尾就停止，绝不按 token 顺序串成连续驾驶。读取 `input_packet.json` / `input_packet.json.gz`、`planning_evidence.json`、`trajectory.json`、`official_replay/trace.json.gz`（亦兼容未压缩 trace）。缺少或损坏 trace 时仍可检查单个源锚点。

NAVSIM 默认主画面为 `official_replay/official_bev.mp4` 的归档执行视频；三维视图标为“非官方诊断”，三轨迹图默认折叠为辅助诊断。只有 `render_status.json` 同时保存 `renderer_version="navsim-sdk-plots-1.0"` 与 `official_visualization=true` 时，才标记为 NAVSIM SDK 官方绘图。缺少新标记的已有视频明确标为“旧版自定义BEV视频（非官方绘图）”，不能因为评分数据来自官方或文件名含 `official` 就视为官方可视化。没有完整视频时可显示其已归档首帧，不自动切换到自定义三维主展示；只读查看器不会自行重绘旧视频。

源规划 states 是源帧局部坐标中的车身中心；提交轨迹是未来 0.5–4 秒的 8 个后轴相对姿态；官方 trace 是含端点的 41 个 0.1 秒 global-map 采样。查看器按源帧后轴全局位姿作 SE(2) 变换，绘制后轴轨迹，并用归档后轴到车身中心的距离放置自车模型。actor 的第三列是航向，位置参考点是包围框中心，不能当作高度或后轴。三条轨迹保留各自参考点、来源和采样说明。

相机和源 BEV 只在 t=0 展示，不生成或借用未来相机图；当前产物没有相机内外参，不做精确投影。未来状态来自已保存的官方评分执行，仅供离线回放，不作为新的规划、模型调用或记忆输入。NAVSIM 模式不会加载规划核心或重建势场。

“执行证据”分开列出内部 quality、trace 的官方单锚点最终 PDM 指标、官方 CSV 本 token 行以及 `benchmark_result.json` 的整轮两阶段 EPDMS 聚合值。CSV 后处理和整轮聚合不冒充 trace 的单锚点分数；各分数不随回放时间变成瞬时指标。只授权单个 sample 目录时，不越界读取父目录整轮分数。

模型面板展示已有 `model_prompt.txt`、原始 `model_raw`、解析 `model_answer` 和归档错误；检索面板保留真实经验文本、ID、分数及诊断摘要，省略内部地图、历史和特征数组。旧产物没有实际模型输入图片清单时，不推断所有导出相机图都曾发送给模型。`official_bev.mp4` 是评分 BEV 回放，不是未来真实相机录像；41 帧覆盖 4 秒物理时间，10fps 编码时长为 4.1 秒。

服务器上运行同一入口并将 `--root` 指向服务器产物路径，推荐监听回环地址，再从本机转发：

```bash
# 服务器：使用已部署的 viewer 目录和可用 Python 环境
python /path/to/ep_agent_viewer/run_viewer.py --root /path/to/outputs/navsim/warmup/A_base_adapter_fixed --host 127.0.0.1 --port 8765
# 本机另开终端，连接自己的服务器账号
ssh -N -L 8765:127.0.0.1:8765 user@server
```

本机浏览器打开 `http://127.0.0.1:8765/`。这些是启动示例，查看器不会自行连接或启动远端服务。

## 离线记忆库浏览

支持单独打开离线发布库，也可以与实验结果一起打开。使用本地 PowerShell：

```powershell
conda activate mulmem_highway
python "D:/code/AutoDriver_Agent_memory_system/Multi-level Memory System for Intelligent Driving/ep_agent_viewer/run_viewer.py" `
  --memory-root "E:/driving_memory_workspace/libraries/driving_experience" `
  --port 8772 --open
```

页面为 `http://127.0.0.1:8772/memory`。同时查看实验时，在原启动命令末尾增加 `--memory-root "记忆库完整目录"`，通过页头“离线记忆库”切换。

`--memory-root` 应指向含 `manifest.json` 的库根目录，不能选其下的 `records` 子目录。仅加载记忆库时，首页会自动进入记忆库页面；实验回放还需通过 `--root` 指定实验结果目录。

仅浏览 `manifest.json` 登记的已发布经验，支持搜索、来源筛选、分页、经验原文、BEV和八向图、NPZ张量摘要与归档证据切换。不会扫描 `pending` / `.history`，不迁移格式、不重新提取特征、不调用模型、不向原库写文件。图像及细节按需核验归档哈希；损坏或缺失内容会注明原因。

选择经验后，点击“三维场景”可旋转、缩放查看归档道路、自车模型、交通对象和醒目的八向角色框。物理标线与可开关的中心拓扑分开绘制，未知边界不画成实线；地图自身的粗几何保留来源说明。支持俯视、复位、图层开关及截图。证据按钮按实际保存时刻切换。已登记的 `audit/supervision.json` 提供归档后续实驶轨迹，并可显示沿自车宽度生成、按道路域裁剪的几何走廊；这属于离线回顾，不是 MPC 输出、在线预测或安全通行证明。后续证据仅显示尚未驶过的已录制段，点数不足时停用。

前视面板按同一证据已校验的图像映射、相机内外参和相机姿态叠加真实三维框及轨迹走廊。OpenScene 可使用已归档三维中心与尺寸；缺少中心 z 的 nuScenes 对象不补造真实三维框。轨迹路面使用局部平面近似，图像曝光时间偏移和几何来源明确显示。highway-env 未保存前视图时不会用俯视图冒充前视。八向角色只读取已存匹配 ID，不重新分配。未保存的势场源、规划、模型关注和目标不补造。

离线后续观察与当前检索帧明确分开；记录声明、资产可读与主程序实际检索验收分别显示。`eligible`、`active` 或能浏览图片，不代表此版本主程序已验收兼容；主程序未验证时保留“未验证”。缺少原始相机或三维标定时不会把BEV当相机画面或伪造相机投影。

## 展示与播放

- 三维驾驶视图采用本地 Three.js WebGL、CARLA 0.9.15 原版 Tesla Model 3 的 GLB 自车模型（黑色车漆、原版车身/轮毂/内饰贴图和玻璃材质）、光照和阴影；默认近距离跟车，支持鼠标旋转、缩放、平移和俯视切换；地图、八向沿车道区域、势场对象、关注对象、轨迹、行车走廊与引导点可分别显示。
- 时间轴支持播放、暂停、单步、倍速和模型调用时刻定位。实时跟随读取文件追加，末尾尚未写完的 JSONL 行留待下次读取。
- 相机图像可展示同帧三维框、轨迹和走廊投影。无标定或缺少高度时显示缺失原因；旧数据的示意高度明确标为 2.5D。
- 实验播放器的规划轨迹按原始点序从绿色渐变到黄色：首点绿、末点黄。三维与前视一致，图像裁剪保留原始规划进度，不把剩余片段重新铺满渐变。行车走廊为半透明黄色填面，不描外沿或三角网格边。离线记忆库采用相同的车辆、轨迹渐变与无描边黄色走廊样式，但仍标注“未来实驶”，不将其当作规划输出。
- 模型面板显示归档的真实输入、输出和明确关联到当前计划的历史回答；基础规划不生成虚构对话。记忆面板展示已归档检索记录、分数和可取得的经验内容。
- 视频使用独立播放器，支持拖动、倍速和全屏。仅在归档提供可信时间映射时与规划时间轴同步；没有逐帧映射的旧 MP4 作为独立视频播放，不假装与当前状态精确对齐。
- 浏览器不支持原始 `mp4v` 编码时，点击“生成浏览器兼容副本”。展示程序通过 FFmpeg 在自身 `.cache/video` 中生成 H.264 副本，不覆盖原视频；仅允许处理已结束的录像。需要本机 FFmpeg，默认从 PATH 查找，也可用 `--ffmpeg "路径/ffmpeg"` 指定。`--cache-dir` 可另设副本目录。运行中的实验仍只读其已有产物。
- 势场默认显示原始 `Uatt + ΣUrep`，可切换独立的 MPC 加权项；采用密集网格、viridis 色系、二维等势线和可拖动三维势面。道路域外留空，显示窗口独立于对象40/20米筛选范围，必要时覆盖已选源的完整作用范围。仅在归档诊断和指定源码的理论版本、参数及作用域相符时，调用实际势场公式重建；缺少必要信息会明确停用。不会以示意高斯代替真实规划势场。
- 可以导出当前三维视图 PNG；导出由浏览器保存，不写实验目录。

## 数据与对齐边界

优先使用 `plans.jsonl`、`frames.jsonl`、`captures/*/scene.json`、`planning_evidence.json`、`model_inputs.jsonl`、`model_outputs.jsonl`、`retrieval.jsonl`、`map.json`、`summary.json` 以及 `camera_frames/*/manifest.json` / `current_observed_state.json`。缺失的产物独立标注，不使整轮无法播放。检索卡片只显示产物实际保存的经验 ID、分数与正文；仅保存 ID 的旧日志不会凭空还原经验正文，完整输入 prompt 仍按原文展示。

关联依据是 **源 frame_id 和源时间同时一致**，不是“取最近一张图”。执行结果的时间不冒充规划源时间。模型旧回答仅通过计划中明确的 `answer_source` 绑定。相机同时检查图像 frame 与 timestamp；当前帧没有归档图片时允许跳转到确有图片的帧，不借图叠加。

现有 CARLA 六路 RGB 实际采集为 10Hz，但图片只按 1Hz 归档；因此实时图像更新受归档频率限制。已完成的 MP4 可完整播放。要实现不依赖归档的每帧视频流，需要主程序额外输出；本工具遵守不改主程序的要求，不暗中新增采集或 tick。

三维图中的行车走廊是沿实际轨迹左右各半个自车宽度的连续带，按原道路域裁剪；车体扫掠另存为证据，不作为展示走廊。合并路面保留孔洞，仅绘制实际标线，不再把每条车道画成闭合边框。走廊不能作为无碰撞或可执行性的独立证明。道路区域和 APF 显示分别依据各自的真实作用域，不能把视觉上下文对象都标成规划对象。

势场按需重建并缓存。播放时只保留一个势场计算请求，独立标明势场源帧和与主画面的时间差，不将较慢的重建结果冒充当前帧。常值零场不会补造峰谷。二维狭长窗口可旋转90°展示，坐标轴明确标记；三维横纵米制比例保持一致，势能高度为相对单位。

自车模型使用本地 CARLA 0.9.15 的原版 Tesla Model 3 车身及灯具 LOD0 几何和贴图，转换为浏览器 GLB；Unreal 材质适配为 PBR，并设置黑色车漆，按归档车辆外廓缩放。模型不表示该次实验车辆的品牌，材质表现不等同于 Unreal 原渲染。Third-party 资源、来源及许可证见 `web/assets/CREDITS.txt` 与 `web/vendor/*LICENSE*`；所有运行依赖均本地提供。AutoAgent0 和 CW-Net 仅作为视觉参考，未复制其演示视频、页面图片或内部渲染器。

## 真实数据集接入约定

具体数据集适配器不与主程序耦合。将数据整理到上述单轮产物格式即可接入：每帧 `frame_id` / `time`，自车与对象的统一世界坐标、heading（弧度）、length / width / height，规划状态 `[x,y,yaw,speed]`；相机保存图像路径、宽高、FOV 或内参、同帧世界到相机变换。

当前相机接口遵循 CARLA 的 `world_to_camera_ue`（UE 相机前方为 x，右方为 y，上方为 z），内部驾驶坐标为 x / y 向左 / z 向上。其他数据集必须先转换坐标与相机轴，不能直接套用矩阵。没有三维真值或标定时只播放图像，不从图片猜测“准确三维框”。
