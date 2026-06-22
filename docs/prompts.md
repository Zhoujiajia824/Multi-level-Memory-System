# 在哪里改提示词（Prompts Editing Guide）

> 所有 VLM 提示词集中在 **`config/prompts.yaml`** 单一文件，无需改 Python 代码。

## 结构总览

```yaml
scene_understanding:
  system: |
    <场景理解 VLM 的 system 指令>
  user: |
    <场景理解 user prompt，占位符 {image_meta} {ego_brief}（当前阶段未启用占位符）>
  required_fields:
    - scene_description
    - ego_status_text
    - lanes
    - vehicles
    - pedestrians
    - traffic_lights
    - intersections
    - traffic_density
    - risk_factors
    - scene_id
    - weather_id

decision:
  system: |
    <决策 VLM 的 system 指令>
  user: |
    <完整决策 prompt，占位符：
     {scene_block} {ego_state_block} {nav_block} {history_block} {memory_block}
     {waypoint_min_num} {waypoint_max_num} {horizon_seconds} {dt}
     {behavior_enum_str}>

memory_integration:
  short_term_block: |
    ## 短期记忆（最近关键帧上下文）
    {short_term_summary}
  mid_term_item: |
    ### 经验 {index} (相似度: {score})
    - 场景: {scene_id}, 天气: {weather_id}
    - 决策原因: {decision_reason}
    - 行为: {behavior}
  long_term_block: |
    ## 相关驾驶规则（长期记忆）
    {long_term_rules_text}
```

## 占位符说明

模板使用 Python `str.format_map` 语法（`{var}`），由
[`src/vla_memory/common/prompt_loader.py`](../src/vla_memory/common/prompt_loader.py) 渲染。
未定义的占位符默认**保留原文 `{var}`** 并发 warning（`strict=False` 模式），
便于调试，不会中断流程。

### 决策 prompt（`decision.user`）的占位符全集

| 占位符 | 来源 | 说明 |
|---|---|---|
| `{scene_block}` | `prompt_builder._render_scene_block` | 当前场景理解段（含 P4 结构化字段：lanes/vehicles/pedestrians/traffic_lights/intersections） |
| `{ego_state_block}` | `prompt_builder._render_ego_block` | 自车状态段（含 P3 CAN bus 字段：yaw_rate/steering/throttle/brake） |
| `{nav_block}` | `prompt_builder._render_nav_block` | 导航指令段（无导航时为空字符串） |
| `{history_block}` | `prompt_builder._render_history_block` | 历史轨迹段（最近 10 个点的 ego-centric 坐标） |
| `{memory_block}` | `prompt_builder._render_memory_block` | 三层记忆段（memory_off 模式时为空字符串） |
| `{waypoint_min_num}` | `config/decision.yaml -> trajectory.waypoint_min_num` | 轨迹点数下限 |
| `{waypoint_max_num}` | `config/decision.yaml -> trajectory.waypoint_max_num` | 轨迹点数上限 |
| `{horizon_seconds}` | `config/decision.yaml -> trajectory.horizon_seconds` | 预测时间窗（秒） |
| `{dt}` | `config/decision.yaml -> trajectory.dt` | 轨迹采样间隔（秒） |
| `{behavior_enum_str}` | `config/decision.yaml -> behaviors.valid_behaviors` | 行为枚举字符串（管道符分隔） |

### 中期记忆单条（`memory_integration.mid_term_item`）的占位符

| 占位符 | 说明 |
|---|---|
| `{index}` | 记忆排序序号（1, 2, 3...） |
| `{score}` | 检索相似度得分（保留 3 位小数） |
| `{scene_id}` | 历史经验的场景类型 |
| `{weather_id}` | 历史经验的天气类型 |
| `{decision_reason}` | 历史经验的决策原因摘要 |
| `{behavior}` | 历史经验采取的行为 |

## 安全编辑指南

1. **直接编辑 `config/prompts.yaml`**，无需改 Python 代码。
2. 占位符用 Python `str.format_map` 语法（`{var}`）。注意 YAML 字符串中要写两个花括号 `{{` `}}` 才能输出字面的 `{` `}`（例如 JSON 示例段）。
3. **不要重命名占位符**，除非同时 grep 以下文件确认 ctx dict key 一致：
   - [`src/vla_memory/decision/prompt_builder.py`](../src/vla_memory/decision/prompt_builder.py)
   - [`src/vla_memory/perception/scene_understanding.py`](../src/vla_memory/perception/scene_understanding.py)
4. **不要在 prompt 中硬编码路点数字**，请使用 `{waypoint_min_num}` / `{waypoint_max_num}` —— 数字会被 `config/decision.yaml` 中的设置自动注入，改 yaml 即可全链生效。
5. 改完后跑下面的命令验证模板仍能渲染：
   ```bash
   conda activate mulmem
   python -m pytest tests/test_prompt_loader.py tests/test_prompt_builder.py -q
   ```
6. VLM 原始响应在 INFO 级别打印，标记 `[SCENE_UNDERSTANDING]` / `[DECISION]`，用于调试提示词回归：
   ```bash
   ls outputs/logs/scene_understanding_*.log outputs/logs/decision_client_*.log
   ```

## 常见任务

### 修改路点数量

编辑 [`config/decision.yaml`](../config/decision.yaml)：
```yaml
trajectory:
  waypoint_min_num: 22    # 改这里
  waypoint_max_num: 28    # 和这里
```
Prompt 模板、parser、schema 三处会自动同步。

### 修改决策模型接收的图片数量

编辑 [`config/decision.yaml`](../config/decision.yaml)：
```yaml
vlm_inputs:
  image_context_size: 5         # 默认 3：当前帧 + 2 张历史
  include_current_frame: true
  max_images_per_call: 4        # Qwen-VL-Max 建议 ≤ 4
```

### 为场景理解新增检测类（如施工标志 / 限速牌）

1. 在 [`config/prompts.yaml`](../config/prompts.yaml) → `scene_understanding.user` 中加新字段说明 + few-shot 示例。
2. （可选）在 [`schemas/scene.py`](../src/vla_memory/schemas/scene.py) 中加对应子模型 + 字段，让校验更严格。
3. 在 [`scene_understanding.py::_parse_and_validate`](../src/vla_memory/perception/scene_understanding.py) 的 `defaults` 中加默认值，缺失时降级为空数组而非中断。
4. （可选）在 [`prompt_builder.py::_render_scene_block`](../src/vla_memory/decision/prompt_builder.py) 中加渲染逻辑，把新字段展示到决策 prompt。

### 切换 VLM 提供商（不改 prompt 模板）

编辑 [`config/api_models.yaml`](../config/api_models.yaml) 切换 `provider` / `base_url` / `model_name`，prompts.yaml 不动。

## 为什么 YAML 而非独立 `.txt`

- **单文件单一来源**：所有 prompts 集中一处，便于审查和版本化。
- **diff 可读**：每段提示词都有 YAML key 标记，PR 中改动一目了然。
- **支持注释**：可在 yaml 中写中文注释解释设计意图。
- **配合代码同步演进**：prompts.yaml 与代码一起版本化，避免 prompt 改了代码忘改、或反之的不一致。
