# 后续迭代路线图

> 本文档描述 VLA Memory Demo 从 v0.1 到 v1.0 的迭代方向。
> 每个版本有明确的优先级和依赖关系。

---

## 版本迭代计划

### v0.1（当前版本）— 基础 Demo

- ✅ nuScenes v1.0-mini + CAM_FRONT 单摄像头
- ✅ DINOv2-base 图像特征提取（768 维）
- ✅ Qwen-VL 场景理解（通过 DashScope API）
- ✅ 短期记忆（deque 滑动窗口）
- ✅ 中期记忆（FAISS IndexFlatIP + 6 维联合评分）
- ✅ 长期记忆（YAML 规则匹配）
- ✅ memory_on / memory_off 对比评测
- ✅ ADE / FDE / 轨迹有效率 / 行为准确率指标
- ✅ CSV / JSONL / Markdown 三种评测报告

---

### v0.2 — API 健壮性与批处理增强

**优先级：高**

- [ ] VLM API 批处理：支持批量发送场景理解请求，减少 API 调用开销
- [ ] API 响应缓存：对相同图像 + prompt 的 VLM 结果做本地缓存，避免重复调用
- [ ] 失败重试增强：指数退避重试策略，支持断点续传
- [ ] 并发请求：支持 asyncio 并发调用 VLM API（受限于 API 速率限制）
- [ ] 输入校验增强：图像预处理（分辨率适配、格式校验、损坏检测）
- [ ] 配置热更新：运行时修改部分参数无需重启

**依赖：** 无外部依赖，纯代码优化。

---

### v0.3 — 接入本地 VLM

**优先级：高**

- [ ] 支持 HuggingFace Transformers 本地 VLM 推理（如 Qwen2-VL、LLaVA、InternVL）
- [ ] 支持多模态模型加载（GPU / CPU 自动检测）
- [ ] VLM 推理性能优化（量化、KV cache、TensorRT 加速）
- [ ] 配置切换：通过 YAML 一键切换 API 模式 / 本地模式
- [ ] 成本分析：API 模式 vs 本地模式的时间和费用对比

**依赖：** GPU 推荐但非必需（CPU 可运行但慢）。

---

### v0.4 — 更复杂关键帧策略

**优先级：中**

- [ ] 基于运动检测的自适应关键帧采样（非固定 1Hz）
- [ ] 基于场景变化的关键帧采样（场景切换、天气变化时增加采样密度）
- [ ] 支持用户自定义关键帧策略（继承 BaseKeyframeSampler）
- [ ] 多摄像头关键帧融合（CAM_FRONT + CAM_FRONT_LEFT + CAM_FRONT_RIGHT）
- [ ] 关键帧质量评估（模糊检测、曝光检测）

**依赖：** v0.1 基础架构。

---

### v0.5 — CAN Bus Expansion

**优先级：中**

- [ ] 接入 nuScenes CAN bus 数据（转向角、油门、刹车）
- [ ] 车辆动力学状态增强（使用真实 CAN 数据替代差分估计）
- [ ] 轨迹控制量输出（从 waypoint 转向角转换为 steering/throttle/brake）
- [ ] DynamicsAdapter 从 stub 升级为真实动力学模型

**依赖：** nuScenes CAN bus expansion 数据包。

---

### v0.6 — Map Expansion

**优先级：中**

- [ ] 接入 nuScenes 高精地图数据（lane、road_block、road_segment）
- [ ] 基于地图的导航语义替代 RouteInfer 伪标签
- [ ] 实现真正的 collision_proxy（基于地图边界和障碍物位置）
- [ ] 实现真正的 offroad_proxy（基于地图车道边界）
- [ ] 地图可视化工具（将 ego 轨迹叠加到地图上展示）

**依赖：** nuScenes map 数据。

---

### v0.7 — CARLA 仿真环境

**优先级：中**

- [ ] 实现 CARLA 适配器（继承 BaseDrivingDataset）
- [ ] 支持多传感器数据（多摄像头、LiDAR 点云）
- [ ] 动态场景生成（行人横穿、前车急刹、施工区域等）
- [ ] 在线评测闭环：决策输出直接控制仿真车辆
- [ ] CARLA 场景批量生成和评测自动化

**依赖：** CARLA 仿真器 + Python API。

---

### v0.8 — 动力学模型与 MPC

**优先级：低**

- [ ] Bicycle Model 轨迹跟踪控制
- [ ] MPC（模型预测控制）轨迹优化
- [ ] PID 控制器基线对比
- [ ] 学习型控制器（神经网络端到端控制）
- [ ] 轨迹平滑度和舒适性指标（加速度变化率、曲率变化率）

**依赖：** v0.5 动力学数据。

---

### v0.9 — 知识图谱长期记忆

**优先级：低**

- [ ] 使用知识图谱（Neo4j / NetworkX）替代 YAML 规则
- [ ] 场景 → 行为 → 结果 的关系图谱
- [ ] 基于图谱的推理和泛化（相似场景类比）
- [ ] 图谱可视化和交互查询工具
- [ ] 从历史驾驶数据自动提取规则

**依赖：** v0.1 长期记忆框架。

---

### v1.0 — 严格 Planning Benchmark

**优先级：最终目标**

- [ ] 对接 nuScenes 官方 planning benchmark 评测协议
- [ ] 支持 nuScenes trainval 完整数据集（1000+ 场景）
- [ ] GPU FAISS 加速（faiss-gpu）
- [ ] 多摄像头融合 + BEV 特征
- [ ] LiDAR 点云辅助感知
- [ ] 与 nuScenes planning baseline 方法对比（UrbanDriver、PlanTF 等）
- [ ] 论文级实验报告

**依赖：** v0.2-v0.6 的积累。

---

## 优先级矩阵

```
高优先级 ────────────────────────────────────────┐
  v0.2 API 健壮性与批处理                          │
  v0.3 本地 VLM 接入                               │
                                                   │
中优先级 ────────────────────────────────────────┤
  v0.4 关键帧策略增强                              │
  v0.5 CAN Bus Expansion                           │
  v0.6 Map Expansion                               │
  v0.7 CARLA 仿真                                  │
                                                   │
低优先级 ────────────────────────────────────────┤
  v0.8 动力学模型 / MPC                             │
  v0.9 知识图谱长期记忆                             │
                                                   │
最终目标 ────────────────────────────────────────┘
  v1.0 严格 Planning Benchmark
```

---

## 外部依赖与资源

| 资源 | 用途 | 版本要求 |
|------|------|----------|
| nuScenes 数据集 | 自动驾驶场景数据 | v1.0-mini（当前）、v1.0-trainval（v1.0） |
| DINOv2 | 图像特征提取 | facebook/dinov2-base（当前）、可扩展到 large/giant |
| FAISS | 向量相似性检索 | faiss-cpu（当前）、faiss-gpu（v1.0） |
| Qwen-VL | 视觉语言模型 | 通过 DashScope API（当前）、本地部署（v0.3） |
| CARLA | 仿真环境 | 0.9.13+（v0.7） |
| Neo4j | 知识图谱 | 5.x（v0.9） |
| HuggingFace Transformers | 本地模型推理 | 4.36+（v0.3） |

---

## 贡献指南

欢迎贡献！建议流程：

1. 从 `main` 分支创建 feature 分支。
2. 实现功能并添加测试。
3. 确保所有测试通过（`pytest`）。
4. 提交 PR 并描述变更内容。

**代码规范：**
- 所有代码使用中文注释和 docstring。
- 使用 `black` + `isort` 格式化代码。
- 使用 `ruff` 做静态检查。
- 测试覆盖率不低于关键模块的 80%。
