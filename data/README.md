# 数据目录说明

本目录用于存放智能驾驶 VLA 分层记忆系统 demo 所需的数据文件。

## 目录结构

```
data/
├── nuscenes/
│   ├── raw/           # nuScenes 原始数据集（需手动下载放置）
│   └── processed/     # 预处理后的数据
├── knowledge/
│   ├── long_term_rules.yaml      # 长期驾驶规则知识库
│   ├── driving_strategies.yaml   # 驾驶策略知识库
│   └── knowledge_graph/          # 知识图谱目录（预留）
└── README.md
```

## nuScenes 数据集

### 获取方式

请从 nuScenes 官网下载数据集：
- 官网地址: https://www.nuscenes.org/nuscenes
- 第一版使用 **v1.0-mini** 版本（约 4.5GB）

### 放置方式

下载并解压后，将文件放置到 `data/nuscenes/raw/` 目录下，确保目录结构为：

```
data/nuscenes/raw/
├── v1.0-mini/
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── log.json
│   ├── map.json
│   ├── sample.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── scene.json
│   ├── sensor.json
│   └── visibility.json
├── samples/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── CAM_FRONT/          # ← single_front 模式默认使用此摄像头
│   ├── CAM_FRONT_LEFT/     # ← surround_mosaic 模式会用到全部 6 个相机目录
│   ├── CAM_FRONT_RIGHT/
│   ├── LIDAR_TOP/
│   ├── RADAR_BACK/
│   ├── RADAR_BACK_LEFT/
│   ├── RADAR_BACK_RIGHT/
│   ├── RADAR_FRONT/
│   ├── RADAR_FRONT_LEFT/
│   └── RADAR_FRONT_RIGHT/
├── maps/
├── sweeps/
└── v1.0-mini/
```

### 重要说明

- **数据集不会自动下载**，必须手动下载并放置。
- 数据相关脚本如果检测不到数据集，会明确报错并告诉用户应该把数据放到哪里。
- **不允许使用假数据伪装成 nuScenes 跑通**。
- 环境检查阶段（`00_check_environment.py`）仅 warning，不报错。
- 数据相关脚本运行时如果没有数据集，会 hard fail 并输出中文错误信息。
- **Oracle 感知**：开启 `perception.oracle_objects` 时，会读取 nuScenes 的
  `sample_annotation / instance / category / attribute / visibility / sample_data /
  calibrated_sensor` 等元数据表做 GT 标注投影，生成 `perception_objects`。这些是
  **nuScenes 真值标注**，不是检测模型预测，详见
  [docs/perception_upgrade.md](../docs/perception_upgrade.md)。
- **surround_mosaic 模式**：会读取全部 6 个相机目录（CAM_FRONT/FRONT_LEFT/FRONT_RIGHT/
  BACK/BACK_LEFT/BACK_RIGHT）的图像并拼成 2×3 环视图。

## 知识文件

长期驾驶规则和策略知识文件已经预置：
- `knowledge/long_term_rules.yaml`: 包含 15+ 条驾驶规则，按场景和天气分类
- `knowledge/driving_strategies.yaml`: 包含 15+ 条驾驶策略，按场景分类
