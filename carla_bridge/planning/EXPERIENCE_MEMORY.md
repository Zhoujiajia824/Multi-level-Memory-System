# 规划执行经验库与离线配对证据

规划经验保存在 `--memory-bank` 指定目录下的 `planning_experiences.jsonl`，与原文本正例库分开。它记录实际执行的动作、IDM 参数、进展、停车、碰撞、违规及舒适度结果。候选被选中时只是 `pending`；得到连续观测后才成为 `observed` 或 `failed`。运行中断、观测不足、动作或参数明显变化时截断为 `censored`，不把预测轨迹当成执行成功。

目前没有预训练的介入收益模型，也没有预置的配对收益证据。普通运行可以积累参数执行先验和失败反馈；“安全执行过”不等于“模型介入比纯 PDM 更好”。下面的工具只导入已经完成、独立评价的真实执行配对，不会自动创建配对或给未做的实验补数据。

## 查看已有经验库

在仓库根目录、已有 `mulmem_carla` 环境中运行。将示例银行目录换成实际使用的 `--memory-bank`。
当前闭环版本为 `carla_pdm_execution_v4:<配置哈希>`，完整值来自已结束运行的
`manifest.json.outcome.execution_planner_version`。不同控制器、预瞄、PID及规范化后的
`lane_quality` / `lane_maneuver` 开关和参数会产生不同版本；语义、模型、记忆和恢复开关
本身不改变数值版本。下面先从实际运行读取版本，再检查该版本的库：

```bash
conda activate mulmem_carla
RUN_DIR=outputs/carla_runs/替换为实际运行目录
PLANNER_VERSION=$(python - "$RUN_DIR/manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    manifest = json.load(f)
version = manifest.get("outcome", {}).get("execution_planner_version")
if not isinstance(version, str) or not version.startswith("carla_pdm_execution_v4:"):
    raise SystemExit("该运行未记录当前v4数值版本，请核对运行目录和代码版本")
print(version)
PY
)
python -m carla_bridge.planning.experience_cli inspect \
  --bank outputs/memory_banks/paper_v1/train_general \
  --planner-version "$PLANNER_VERSION"
```

工具为历史兼容仍默认 `carla_pdm_selective_v1`，**不等于当前闭环版本**。
省略 `--planner-version` 可能把当前记录统计为不兼容，不能据此认为当前库为空。
旧v1/v2记录保留，但当前规划不跨版本检索，不迁移或删除原经验。需要查看旧版本时显式指定：

```bash
python -m carla_bridge.planning.experience_cli inspect \
  --bank outputs/memory_banks/paper_v1/train_general \
  --planner-version carla_pdm_selective_v1
```

输出包含所选版本、检测到的版本和 schema、兼容记录的种类和最终状态、坏行及版本不兼容数量。`detected_*` 统计原始 JSONL 行；`compatible_record_states` 按同一 ID 的最新有效记录统计。`raw_json_invalid_lines` 还会标出非法 JSON、重复键所在行。`health` 中运行计数器属于本次工具对象，不是历史训练轮数；`trained_model` 固定为 `false`。

汇总CSV同时保存 `execution_planner_version`、`lane_quality_enabled` 与
`lane_maneuver_enabled`。两项数值功能与语义/模型/记忆开关应分别分组，避免把绕行限速副本
或车道成本的差异归因为模型、记忆收益。实际绕行参数和后续观测仍以逐帧审计和执行经验为准。

查看不会修改经验文件、文本库或创建不存在的记忆库。已有库使用与 `RunSession` 相同的独占锁读取一致快照，可能在**库目录外**留下 `.库名.lock`；锁文件不能手工删除。实验运行正在占用该库时工具直接报忙，待实验结束再运行。

## 导入已经实测的配对证据

先完成纯 PDM 与指定语义动作介入的独立闭环配对执行、外部指标计算和预先约定的效用评价，再导出证据 JSON/JSONL。以下命令中的输入文件必须来自实际实验；工具不提供伪造数据生成命令：

```bash
python -m carla_bridge.planning.experience_cli import-pairs \
  --bank outputs/memory_banks/paper_v1/train_general \
  --input /data/zjj/experiments/paired_train/evaluated_pairs.jsonl \
  --planner-version "$PLANNER_VERSION"
```

输入支持单个 JSON 对象、JSON 对象数组，或扩展名为 `.jsonl` 的逐行对象；整批上限为 32 MiB。必须预先固定评价协议和效用定义，不能根据测试结果挑选最有利的指标、协议或动作再回填。

每个配对对象的必需字段如下。字段描述是格式约束，不是已完成实验的示例结果：

| 位置 | 字段与约束 |
| --- | --- |
| 顶层 | `schema_version=1`、匹配当前代码的 `planner_version`、非空 `experimental_unit_id`、`protocol_id`、`utility_definition` |
| 独立性 | `independently_sampled=true`、`protocol_fixed_before_evaluation=true`、`matching_method="restored_full_state"` |
| 初始状态 | `initial_state` 包含自车 `ego.position`、`ego.velocity`，以及 `actors`、`signals`、`map`；`state_fingerprint` 必须等于该对象规范 JSON 的 SHA256 |
| 检索情境 | `context` 为 schema 1 的当前可见结构情境，包含拓扑、导航、速度、相关信号和 actor 关系；不可包含未来事件的真值标签 |
| 两个执行分支 | `baseline` 和 `intervention` 均包含 `evaluation_kind="executed_closed_loop"`、`independent_evaluation=true`、`complete=true` |
| 分支溯源 | 每个分支有非空 `run_id`、`planner_version`、`initial_state`、`state_fingerprint`、正数 `duration_s`、64 位小写十六进制 `artifact_sha256` |
| 配对一致性 | 两个分支版本、初始状态及指纹、评价时长相同；执行 run ID 和 artifact 哈希分别不同 |
| 实际结果 | 每个分支有 `metrics.collision_count`、`metrics.violation_count`、`metrics.route_progress_m`，均为非负有限值，以及按固定定义计算的 `utility`，范围为 `[0,1]` |
| 实际介入 | `intervention.actual_action` 明确保存实际执行的 `lateral_decision` 和 `longitudinal_decision`；不能拿模型请求但未执行的动作冒充 |
| 可选本地证据 | 分支中的 `artifact_path` 指向与其 `artifact_sha256` 匹配的本地完整运行结果文件 |

规范指纹的计算方式与模块一致：UTF-8 编码的 `json.dumps(initial_state, sort_keys=True, separators=(",", ":"), allow_nan=False)`，再计算 SHA256。数值表示也要一致，例如 `3` 与 `3.0` 的 JSON 不同，不能先算指纹再修改导出数据。

默认只读取输入文件所在目录树中的 `artifact_path`，相对路径也相对此目录解析。证据路径不能越界，符号链接越界、URL、UNC 引用都会被拒绝；不会下载外部文件或输出证据文件内容。确实将结果文件放在其他目录时，由操作者显式授权一个目录树：

```bash
python -m carla_bridge.planning.experience_cli import-pairs \
  --bank outputs/memory_banks/paper_v1/train_general \
  --input /data/zjj/experiments/paired_train/evaluated_pairs.jsonl \
  --planner-version "$PLANNER_VERSION" \
  --artifact-root /data/zjj/experiments/paired_train/artifacts
```

此时相对的 `artifact_path` 相对此 `--artifact-root` 解析。没有提供本地路径时只核验哈希的格式及配对约束，输出 `artifacts_without_local_verification`，不会声称已检查实际文件。

整批 JSON、配对契约与所有可用本地哈希均验证后，才取得与运行程序相同的银行锁。随后原子替换经验文件，保留原有内容；校验失败、银行忙、暂存或替换失败不会导入半批。相同 ID 且内容完全相同的配对重复导入会跳过；同 ID 内容不同则拒绝。既有银行含坏行时也会拒绝导入，先用 `inspect` 查清原因；工具不会自动删除或修补历史证据。

## 经验如何作用于规划

- 普通执行：同类拓扑、信号距离、相关 actor 关系和速度下，至少多个独立 run 的完整成功观测才能提供参数候选先验。连续多个时间片不能冒充多个独立实验。参数仍需在当前状态下重新生成、仿真和评分，不能重放旧 XY 轨迹。执行失败只提供动作特定的有界负反馈，不能证明该动作导致失败。
- 配对执行：同实际动作、同协议和效用尺度的独立配对，用保守的单侧收益下界判断证据是否充分。普通无事故执行不会生成正因果收益。更高进展不能抵消已经增加的碰撞或违规。
- 查询触发：场景检索的 `causal_advantage` 始终保持 `unknown`；只有特定动作具有充分配对收益时，`intervention_priors` 才能提示值得咨询模型。多个动作同时检索时对下界进行多重比较校正。
- 采纳限制：咨询价值不会自动给 STOP/YIELD 放行，也不会增加语义评分预算。采纳阶段另按模型请求的具体动作检查证据，不能把 MAINTAIN 的收益泛化给 STOP。

## 配对实验与论文评估边界

文件哈希只能说明文件完整性。格式检查无法证明外部结果诚实、状态恢复完整，或两个 CARLA 分支真正可比。自车位置、速度相同远远不够：需要控制车辆和行人的状态、信号相位及剩余时间、Traffic Manager 与事件随机状态、天气、同步步长和控制器初始状态，并记录代码、地图与场景版本。若这些条件无法验证，就保留普通执行经验，收益仍应标为未知，不要导入为严格配对收益。

训练场景、配对证据采集场景与最终评测的地图/路线/种子应明确分开。评测采用冻结的库和 `--memory-access read_only`；`memory_off` 禁止所有经验读写。不要把当前 run、相邻片段或同一评估轨迹复制多份作为独立支持。报告路线 100% 完成率、碰撞违规、行程时间和停车时间，并单独报告模型调用次数、采纳率及记忆覆盖率；单个 R05 完成不代表长尾场景已有收益。
