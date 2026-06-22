# API 配置说明文档

> 本文档详细描述 `config/api_models.yaml` 的配置方法、支持的服务商列表、
> Prompt 模板设计以及错误处理和重试机制。

---

## 1. config/api_models.yaml 配置详解

### 1.1 配置文件结构

`api_models.yaml` 是 API 模型配置文件，定义了系统使用的视觉语言模型（VLM）
服务商、模型名称、接口地址和认证信息。

```yaml
# config/api_models.yaml 配置结构

# 默认使用的服务商和模型
default:
  scene_provider: "qwen"              # 场景理解默认服务商
  decision_provider: "qwen"           # 决策默认服务商
  scene_model: "qwen-vl-max"          # 场景理解默认模型
  decision_model: "qwen-vl-max"       # 决策默认模型

# 服务商配置列表
providers:
  qwen:
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: "DASHSCOPE_API_KEY"    # 环境变量名称
    models:
      - name: "qwen-vl-max"
        max_tokens: 4096
        supports_vision: true
      - name: "qwen-vl-plus"
        max_tokens: 4096
        supports_vision: true

  siliconflow:
    base_url: "https://api.siliconflow.cn/v1"
    api_key_env: "SILICONFLOW_API_KEY"
    models:
      - name: "Qwen/Qwen2.5-VL-72B-Instruct"
        max_tokens: 4096
        supports_vision: true

  zhipu:
    base_url: "https://open.bigmodel.cn/api/paas/v4"
    api_key_env: "ZHIPU_API_KEY"
    models:
      - name: "glm-4v"
        max_tokens: 4096
        supports_vision: true

  deepseek:
    base_url: "https://api.deepseek.com/v1"
    api_key_env: "DEEPSEEK_API_KEY"
    models:
      - name: "deepseek-vl"
        max_tokens: 4096
        supports_vision: true
```

### 1.2 配置项说明

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `default.scene_provider` | string | 场景理解使用的默认服务商 |
| `default.decision_provider` | string | 决策使用的默认服务商 |
| `default.scene_model` | string | 场景理解使用的默认模型名称 |
| `default.decision_model` | string | 决策使用的默认模型名称 |
| `providers.{name}.base_url` | string | 服务商的 API 基础 URL |
| `providers.{name}.api_key_env` | string | 存储 API Key 的环境变量名称 |
| `providers.{name}.models` | list | 该服务商支持的模型列表 |
| `providers.{name}.models[].name` | string | 模型名称 |
| `providers.{name}.models[].max_tokens` | int | 最大输出 token 数 |
| `providers.{name}.models[].supports_vision` | bool | 是否支持视觉输入 |

---

## 2. 支持的服务商列表

### 2.1 阿里云通义千问（Qwen）

| 项目 | 说明 |
|------|------|
| 服务商 | 阿里云百炼平台 |
| base_url | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| api_key_env | `DASHSCOPE_API_KEY` |
| 支持模型 | `qwen-vl-max`、`qwen-vl-plus` |
| 特点 | 支持视觉输入，中文理解能力强，API 兼容 OpenAI 格式 |
| 获取 API Key | [阿里云百炼控制台](https://dashscope.console.aliyun.com/) |

**配置示例**：
```yaml
qwen:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key_env: "DASHSCOPE_API_KEY"
  models:
    - name: "qwen-vl-max"
      max_tokens: 4096
      supports_vision: true
```

### 2.2 硅基流动（SiliconFlow）

| 项目 | 说明 |
|------|------|
| 服务商 | 硅基流动 |
| base_url | `https://api.siliconflow.cn/v1` |
| api_key_env | `SILICONFLOW_API_KEY` |
| 支持模型 | `Qwen/Qwen2.5-VL-72B-Instruct` |
| 特点 | 提供多种开源模型的托管服务，价格相对较低 |
| 获取 API Key | [硅基流动官网](https://siliconflow.cn/) |

**配置示例**：
```yaml
siliconflow:
  base_url: "https://api.siliconflow.cn/v1"
  api_key_env: "SILICONFLOW_API_KEY"
  models:
    - name: "Qwen/Qwen2.5-VL-72B-Instruct"
      max_tokens: 4096
      supports_vision: true
```

### 2.3 智谱（ZhipuAI / GLM）

| 项目 | 说明 |
|------|------|
| 服务商 | 智谱 AI |
| base_url | `https://open.bigmodel.cn/api/paas/v4` |
| api_key_env | `ZHIPU_API_KEY` |
| 支持模型 | `glm-4v` |
| 特点 | GLM-4V 具有优秀的视觉理解能力，支持多轮对话 |
| 获取 API Key | [智谱开放平台](https://open.bigmodel.cn/) |

**配置示例**：
```yaml
zhipu:
  base_url: "https://open.bigmodel.cn/api/paas/v4"
  api_key_env: "ZHIPU_API_KEY"
  models:
    - name: "glm-4v"
      max_tokens: 4096
      supports_vision: true
```

### 2.4 DeepSeek-VL

| 项目 | 说明 |
|------|------|
| 服务商 | DeepSeek |
| base_url | `https://api.deepseek.com/v1` |
| api_key_env | `DEEPSEEK_API_KEY` |
| 支持模型 | `deepseek-vl` |
| 特点 | DeepSeek 的视觉语言模型，推理能力强 |
| 获取 API Key | [DeepSeek 开放平台](https://platform.deepseek.com/) |

**配置示例**：
```yaml
deepseek:
  base_url: "https://api.deepseek.com/v1"
  api_key_env: "DEEPSEEK_API_KEY"
  models:
    - name: "deepseek-vl"
      max_tokens: 4096
      supports_vision: true
```

### 2.5 服务商选择建议

| 需求 | 推荐服务商 | 推荐模型 |
|------|-----------|----------|
| 最佳中文理解 | Qwen | qwen-vl-max |
| 性价比最高 | 硅基流动 | Qwen2.5-VL-72B |
| 视觉推理能力强 | 智谱 | glm-4v |
| 复杂场景推理 | DeepSeek | deepseek-vl |

---

## 3. 场景理解 VLM 和决策 VLM 的区别

### 3.1 功能对比

系统使用两个不同用途的 VLM 调用（可以使用同一个模型，也可以使用不同模型）：

| 维度 | 场景理解 VLM | 决策 VLM |
|------|-------------|----------|
| 功能 | 理解当前场景的语义信息 | 基于场景信息做出驾驶决策 |
| 输入 | 当前帧图像 + 场景理解 Prompt | 当前帧图像 + 决策 Prompt（含记忆上下文） |
| 输出 | 场景描述、分类、目标列表 | 驾驶行为、轨迹、风险评估 |
| 输出格式 | 结构化文本 | JSON 格式 |
| 时效性 | 需要快速响应 | 可以稍慢，但需要更精确 |
| 配置键 | `default.scene_provider` / `default.scene_model` | `default.decision_provider` / `default.decision_model` |

### 3.2 场景理解 VLM

**职责**：对当前帧图像进行语义理解，生成结构化的场景描述。

**输入**：
- 当前帧图像（Base64 编码或 URL）
- 场景理解 Prompt

**输出**：
```json
{
    "scene_description": "城市直行道路，前方有车辆缓行，右侧有行人",
    "scene_id": "straight_road",
    "weather_id": "sunny",
    "surrounding_objects": [...],
    "traffic_elements": [...]
}
```

**模型要求**：
- 必须支持视觉输入（`supports_vision: true`）
- 需要较好的场景描述能力
- 不需要很强的推理能力，重点在感知

### 3.3 决策 VLM

**职责**：基于场景理解结果和记忆上下文，做出驾驶决策并生成轨迹。

**输入**：
- 当前帧图像
- 决策 Prompt（包含导航指令、自车状态、记忆上下文、长期规则）

**输出**：
```json
{
    "behavior": "FOLLOW",
    "behavior_reason": "前方车辆减速，跟随减速",
    "target_speed": 4.0,
    "risk_level": "medium",
    "trajectory": [...],
    "safety_notes": [...]
}
```

**模型要求**：
- 必须支持视觉输入
- 需要较强的推理能力（综合多源信息做决策）
- 需要较好的 JSON 输出格式遵从性
- 建议使用参数量较大的模型

---

## 4. Prompt 模板说明

### 4.1 场景理解 Prompt 模板

```
你是一个自动驾驶场景分析系统。请分析当前图像，输出以下信息：

1. 场景描述：用一段话描述当前道路场景
2. 场景类型：从以下选项中选择
   - straight_road（直行道路）
   - intersection（路口）
   - curve（弯道）
   - highway（高速）
   - parking（停车场）
3. 天气条件：从以下选项中选择
   - sunny（晴天）/ rainy（雨天）/ cloudy（多云）/ night（夜间）
4. 周围目标：列出图像中的车辆、行人、自行车等目标
5. 交通元素：列出可见的交通标志、信号灯等

请以 JSON 格式输出。
```

### 4.2 决策 Prompt 模板

```
你是一个自动驾驶决策系统。请根据以下信息做出驾驶决策：

【导航指令】
{nav_instruction}

【自车状态】
- 当前速度：{ego_speed} m/s
- 航向角：{ego_yaw} rad

【短期记忆】
{short_term_summary}

【中期记忆（相似场景参考）】
{mid_term_context}

【长期规则】
{long_term_rules}

请输出以下格式的 JSON：
{
    "behavior": "KEEP_LANE|FOLLOW|STOP|TURN_LEFT|TURN_RIGHT",
    "behavior_reason": "行为选择的理由",
    "target_speed": 目标速度（m/s）,
    "risk_level": "low|medium|high",
    "trajectory": [{"t": 时间偏移, "x": x坐标, "y": y坐标}, ...],
    "safety_notes": ["安全提示1", "安全提示2"]
}

约束条件：
- trajectory 至少包含 20 个航点
- 时间间隔 0.1 秒
- 坐标使用自车坐标系（X 轴正前方，Y 轴正左方）
```

### 4.3 Prompt 变量

| 变量 | 来源 | 说明 |
|------|------|------|
| `{nav_instruction}` | 路线推断模块 | 当前导航指令（直行/左转/右转等） |
| `{ego_speed}` | 自车状态 | 当前速度（m/s） |
| `{ego_yaw}` | 自车状态 | 当前航向角（rad） |
| `{short_term_summary}` | 短期记忆 | 最近帧的场景演化摘要 |
| `{mid_term_context}` | 中期记忆 | 相似历史场景的决策参考 |
| `{long_term_rules}` | 长期记忆 | 适用的驾驶规则列表 |

---

## 5. 错误处理和重试机制

### 5.1 错误类型

| 错误类型 | 错误码 | 说明 | 处理方式 |
|----------|--------|------|----------|
| API 调用失败 | - | 网络错误、服务不可用 | 重试 |
| 认证失败 | 401 | API Key 无效或过期 | 报错并停止 |
| 请求频率超限 | 429 | 超出 API 调用频率限制 | 等待后重试 |
| 模型输出格式错误 | - | VLM 输出不是有效 JSON | 解析修复或 Fallback |
| 模型输出内容错误 | - | 缺少必要字段或字段值无效 | 解析修复或 Fallback |
| 超时 | - | API 响应时间过长 | 重试 |

### 5.2 重试机制

```yaml
# config/api_models.yaml - 重试配置
retry:
  max_retries: 3              # 最大重试次数
  base_delay: 1.0             # 基础等待时间（秒）
  max_delay: 30.0             # 最大等待时间（秒）
  backoff_factor: 2.0         # 指数退避因子
  retryable_errors:           # 可重试的错误码
    - 429                     # 频率限制
    - 500                     # 服务器错误
    - 502                     # 网关错误
    - 503                     # 服务不可用
    - 504                     # 网关超时
```

### 5.3 重试策略

采用**指数退避**重试策略：

```
第1次重试：等待 base_delay 秒（1.0 秒）
第2次重试：等待 base_delay * backoff_factor 秒（2.0 秒）
第3次重试：等待 base_delay * backoff_factor^2 秒（4.0 秒）
...

等待时间 = min(base_delay * backoff_factor^(retry_count-1), max_delay)
```

### 5.4 Fallback 机制

当 VLM 输出无法解析或重试次数用尽时，系统会触发规则 Fallback：

1. **输出格式修复**：尝试从 VLM 输出中提取 JSON 片段
2. **部分字段补全**：保留有效字段，用默认值填充缺失字段
3. **完全 Fallback**：使用 `rule_fallback.py` 生成安全兜底决策

Fallback 决策的特征：
- `behavior_reason` 以 `[Fallback]` 前缀标识
- 生成一条直线行驶的保守轨迹
- `risk_level` 设为 `medium`

### 5.5 API Key 管理

**安全要求**：
- API Key 不硬编码在配置文件中
- 通过环境变量读取（`api_key_env` 指定环境变量名）
- 系统启动时检查所有需要的环境变量是否已设置

**环境变量设置示例**：

```bash
# Linux/macOS
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
export SILICONFLOW_API_KEY="sk-xxxxxxxxxxxxxxxx"
export ZHIPU_API_KEY="xxxxxxxxxxxxxxxx.xxxxxxxxxxxxxxxx"
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"

# Windows PowerShell
$env:DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxx"
$env:SILICONFLOW_API_KEY = "sk-xxxxxxxxxxxxxxxx"
$env:ZHIPU_API_KEY = "xxxxxxxxxxxxxxxx.xxxxxxxxxxxxxxxx"
$env:DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxx"
```

### 5.6 日志记录

所有 API 调用都会记录以下信息：

| 日志项 | 说明 |
|--------|------|
| 请求时间 | API 调用的发起时间 |
| 服务商和模型 | 使用的 provider 和 model |
| Prompt 长度 | 输入 Prompt 的 token 数（估算） |
| 响应时间 | API 响应耗时（毫秒） |
| 状态码 | HTTP 响应状态码 |
| 重试次数 | 本次调用的重试次数 |
| 错误信息 | 如有错误，记录详细错误信息 |

日志输出到 `outputs/{experiment}/logs/api_calls.log`。

---

## 6. 配置最佳实践

### 6.1 开发环境

建议使用性价比高的服务商进行开发和调试：

```yaml
default:
  scene_provider: "siliconflow"
  decision_provider: "siliconflow"
```

### 6.2 正式评测

建议使用效果最好的服务商进行正式评测：

```yaml
default:
  scene_provider: "qwen"
  decision_provider: "qwen"
  scene_model: "qwen-vl-max"
  decision_model: "qwen-vl-max"
```

### 6.3 混合配置

场景理解和决策可以使用不同的服务商：

```yaml
default:
  scene_provider: "siliconflow"    # 场景理解用性价比高的
  decision_provider: "qwen"        # 决策用效果最好的
```

这样可以平衡成本和效果：场景理解对模型要求较低，而决策需要更强的推理能力。
