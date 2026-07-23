# 感知输入升级：六视角环视拼接 + Oracle 感知对象

> 本文档说明 v0.1 之后新增的感知升级特性：(1) 六视角 surround-view mosaic 输入模式；
> (2) 基于 nuScenes GT 标注投影的 oracle 感知对象 `perception_objects`。
> 两种特性均**保留原有单前视角（single_front）模式**，通过配置切换，不破坏既有流程。

---

## ⚠️ 重要：Oracle 来源声明

`perception_objects` 中的**检测框、类别、位置、速度、加速度全部来自 nuScenes
ground-truth 3D 标注（`sample_annotation`）的投影与因果差分**，**不是外部检测模型
或运动学模型的预测结果**。

- 每个对象的 `is_oracle` 字段恒为 `True`，显式标注其 oracle（真值）性质。
- 这些对象用于**研究/评测**场景下为决策 VLM 提供准确的感知先验，便于隔离"感知质量"
  与"决策质量"。**不可等同于真实部署中的在线检测结果。**
- 该 oracle 性质已在五处显式标注：代码注释（`oracle_perception.py`）、配置
  （`data_nuscenes.yaml`）、输出字段（`is_oracle`/`perception_objects`）、决策提示词
  （"Oracle 感知对象（nuScenes GT 标注投影，非检测模型预测）"）、本文件。

---

## 一、配置（`config/data_nuscenes.yaml`）

```yaml
perception:
  mode: "single_front"        # single_front | surround_mosaic
  cameras:                    # 2x3 行优先顺序
    - "CAM_FRONT_LEFT"
    - "CAM_FRONT"
    - "CAM_FRONT_RIGHT"
    - "CAM_BACK_LEFT"
    - "CAM_BACK"
    - "CAM_BACK_RIGHT"
  mosaic:
    cell_width: 480
    cell_height: 270
    label_subimages: true     # 每子图标相机名，便于 VLM 识别视角
  oracle_objects: false       # 是否注入 GT 投影 perception_objects
  oracle:
    max_distance_m: 50.0
    box_visibility: "ANY"     # ALL | ANY | NONE
```

- `mode: single_front`：原有行为，单张 CAM_FRONT 图，向后完全兼容。
- `mode: surround_mosaic`：把 6 相机拼成一张 2×3 mosaic 图替代前视图，进入 VLM 场景
  理解、决策、DINOv2 特征、短期/中期记忆、决策记录全流程。
- `oracle_objects: true`：在 `enrich_keyframes_with_state` 阶段为每帧生成
  `perception_objects` 并注入 keyframe / 决策 prompt / jsonl 输出。

---

## 二、六视角 surround-view mosaic

布局（2 行 × 3 列，行优先顺序与 `perception.cameras` 一致）：

```
上排：CAM_FRONT_LEFT | CAM_FRONT | CAM_FRONT_RIGHT
下排：CAM_BACK_LEFT  | CAM_BACK  | CAM_BACK_RIGHT
```

- 这是**六个独立相机视角的网格拼接**，不是连续全景图。每个子图左上角标注对应相机名
  （白字 + 黑描边），便于 VLM 明确识别每格的视角来源。
- 拼接实现：`src/vla_memory/perception/surround_mosaic.py` 的 `build_surround_mosaic()`。
- 缺图时该格留深灰底并标注 `(missing)`，不整体失败。
- **设计要点**：mosaic 落盘为单张 JPEG（`outputs/mosaic/<sample_token>.jpg`），令
  `kf["image_path"]` 指向它。由于全链路图像以"文件路径"流转，下游 DINOv2 特征提取、
  场景理解、短期/中期记忆、决策 VLM 输入、决策记录**全部自动适配**，无需改动图像通路
  签名。
- DINOv2 特征策略：**单张 mosaic 提取 1 个 768d 向量**（CLS token），FAISS 维度不变
  （768），`MemoryRecord` / `ShortTermMemoryItem` schema 不改。

---

## 三、Oracle 感知对象 `perception_objects`

实现：`src/vla_memory/data/oracle_perception.py`，入口 `NuScenesAdapter.get_perception_objects()`。

### 生成流程
1. 遍历当前 sample 的所有 `sample_annotation`，对每个标注：
   - 类别/属性/尺寸/全局位置（来自 annotation + instance/category/attribute 表）；
   - ego-centric 位置 / 到 ego 距离 / 相对朝向（复用 `EgoState.quat_to_yaw` + 全局→ego
     2D 旋转范式，与项目轨迹一致）；
   - **因果运动学**：速度/加速度沿 annotation 的 `prev` 链差分（见下）。
2. 按 `max_distance_m` 过滤。
3. 对 6 相机逐个投影（`nusc.get_sample_data(sd_token, BoxVisibility.ANY)` +
   `view_points`），给每个对象挂 `boxes_2d`（各相机 2D 框）与 `visible_cameras`。
4. 仅保留**至少在一个相机可见**的对象（喂给相机 mosaic VLM），按距离升序排序。

### 每对象字段（`schemas/perception.py` 的 `PerceptionObject`）
`annotation_token, instance_token, category(映射到 VALID_OBJECT_TYPES),
category_name_raw, semantic_label, attributes, size[w,l,h], position_global[x,y,z],
position_ego[x,y], distance_to_ego, heading_global, heading_ego, visible_cameras,
boxes_2d{cam:[x1,y1,x2,y2]}, velocity(ego-centric[vx,vy]), speed, acceleration(ego-centric),
acceleration_mag, velocity_available, acceleration_available, kinematics_source,
num_lidar_pts, visibility_level, is_oracle=True`。

> velocity/acceleration 为 **ego-centric 坐标系**（vx 前向、vy 左向），是目标自身速度/
> 加速度旋转到 ego 轴向（非相对 ego 运动）。

---

## 四、运动学因果性（红线）

速度/加速度**严格满足在线因果性**：

- **仅沿 `annotation["prev"]` 链回溯**（当前帧 + 历史帧），**绝不读取 `next`（未来帧）**。
- 2 帧差分 → 速度；3 帧差分 → 加速度。
- 时间戳取自父 `sample`（annotation 自身无 timestamp 字段）。
- nuScenes 标注仅存在于 2Hz 关键帧，故差分窗口约 0.5s（速度）/ 1.0s（加速度），
  `kinematics_source` 中标注 `_2hz`。
- **缺历史时置空并标记不可用，禁止填假速度/假加速度/默认 0**：
  - 首次出现（无 `prev`）：`velocity=None, velocity_available=False,
    kinematics_source="unavailable_no_history"`；
  - 仅有 1 个 `prev`（无更深历史）：仅有速度，`acceleration=None,
    acceleration_available=False, kinematics_source="..._velocity_only"`。

`kinematics_source` 取值（审计标记）：
- `annotation_keyframe_diff_2hz`：速度+加速度均可用；
- `annotation_keyframe_diff_2hz_velocity_only`：仅速度可用；
- `unavailable_no_history`：首次出现，均不可用；
- `unavailable_invalid_dt`：相邻帧时间差非正，均不可用。

---

## 五、输出字段

每帧决策记录（`outputs/decisions_<mode>_<run_id>.jsonl`）新增：
- `perception_mode`：`single_front` / `surround_mosaic`；
- `perception_objects`：oracle 对象列表（仅 `oracle_objects: true` 时非空）；
- `vlm_image_paths`：surround_mosaic 模式下含 mosaic 路径。

---

## 六、运行

> 注意：`07_run_full_demo.py` **没有 `--run-id` 参数**。要避免覆盖既有产物，用
> `--output` 指定独立的输出 jsonl 路径（切勿省略，否则会写入默认的
> `decisions_<mode>_default.jsonl`，覆盖既有基线）。

```bash
conda activate mulmem
cd vla_memory_demo

# 单前视角（回归，向后兼容）
#   config: perception.mode=single_front, oracle_objects=false
python scripts/07_run_full_demo.py --mode memory_off --max-scenes 1 --max-frames 1 \
  --output outputs/decisions_memory_off_sf_test.jsonl

# 六视角环视拼接 + oracle 感知
#   config: perception.mode=surround_mosaic, oracle_objects=true
python scripts/07_run_full_demo.py --mode memory_on --max-scenes 1 --max-frames 1 \
  --output outputs/decisions_memory_on_mosaic_test.jsonl
```

> 切换感知模式/oracle 开关：改 `config/data_nuscenes.yaml` 的 `perception.mode` 与
> `perception.oracle_objects`，无需改代码。

---

## 七、相关文件

| 角色 | 文件 |
|---|---|
| 配置 | `config/data_nuscenes.yaml`（`perception` 块） |
| Mosaic 拼接 | `src/vla_memory/perception/surround_mosaic.py` |
| Oracle 投影 + 因果运动学 | `src/vla_memory/data/oracle_perception.py` |
| PerceptionObject 模型 | `src/vla_memory/schemas/perception.py` |
| Adapter 多相机 + oracle 入口 | `src/vla_memory/data/nuscenes_adapter.py` |
| enrich 集成（拼图+注入） | `src/vla_memory/pipeline/full_demo_pipeline.py` |
| 在线循环（prompt 注入+record） | `src/vla_memory/pipeline/online_loop.py` |
| 决策 prompt 渲染 | `src/vla_memory/decision/prompt_builder.py` |
| 场景理解 prompt 注入 | `src/vla_memory/perception/scene_understanding.py` |
| 提示词模板 | `config/prompts.yaml`、`config/api_models.yaml` |
