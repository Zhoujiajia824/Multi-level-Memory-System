# 八向状态数组存储与离线库迁移

适用于 EP-Driver 与 EP-Driver-CARLA。存储协议 `lane-eight-array-1.0`。

## 1. 此次改变的内容

新生成的八向数值资产使用 `grid.npz`，逐帧 captures 使用 `<帧ID哈希>_grid.npz`。不再同时写一份完整 `grid.json`。`record.json`、`features.json`、经验文本、审核证据仍各自保留；它们不是需要替换的八向数值矩阵。

运行时对象配对的 P/V/A/H 距离与类型、区域条件使用 NumPy 数组计算，查询状态在一次检索内只整理一次。部分分配仍使用现有 `linear_sum_assignment`。BEV、OCC、八向仍按现有规则各取 Top1、按经验ID合并；没有改成展平余弦、ANN或加权总分检索。

存储版本独立于表示和检索指标版本。本次不改变对象准入、区域匹配、距离归一化、阈值或排名规则。当前模型/MPC默认沿允许车道前40m、后20m；检索八向仍按自身现有70m/30m范围与类型规则构建。不能在转换旧文件时将范围标签改成40m/20m，或补出原来没有的车道归属证据。

## 2. 磁盘契约

使用 `np.savez_compressed`，读取必须 `allow_pickle=False`。行列顺序固定：

```text
左前    前    右前
左侧   自车   右侧
左后    后    右后
```

| NPZ键 | 类型与形状 | 含义 |
|---|---|---|
| `storage_version` | Unicode标量 | `lane-eight-array-1.0` |
| `layout` | uint8标量 | 1为完整布局；0仅保存无法构图的无效诊断 |
| `metadata` | Unicode标量 | 紧凑描述性JSON，保存版本、参数、原因及原有语义证据 |
| `values` | float64 `[3,3,7]` | `P_s,P_d,V_s,V_d,A_s,A_d,heading` |
| `value_valid` | bool `[3,3,7]` | 连续值是否有效，缺失不等于数值0 |
| `kind` | int8 `[3,3]` | -1无对象，0车辆，1自行车，2行人，3道路障碍物 |
| `state` | int8 `[3,3]` | 0 occupied，1 empty，2 unknown，3 not_applicable |
| `object_index` | int8 `[3,3]` | -1无对象，-2自车，非负为 `object_ids` 索引 |
| `object_ids` | Unicode `[N]` | 周边唯一对象ID，按ID排序，N≤8 |
| `slot_flags` | int8 `[3,3,3]` | applicable、clear、state_valid；applicable=-1代表null |
| `slot_evidence` | float64 `[3,3,6]` | coverage、raw_coverage、unobserved_area_m2、region_area_m2、known_area_m2、clearance |
| `evidence_present` | bool `[3,3,6]` | 原字段是否存在 |
| `evidence_valid` | bool `[3,3,6]` | 原字段是否为有效数值，区分缺字段与显式null |
| `ego_velocity` | float64 `[2]` | 自车绝对Frenet速度，防止中心格相对速度0丢失自车运动信息 |
| `roads` | int8 `[2,3]` | 左、右道路的known、exists、crossable；允许为空的字段用-1 |

`layout=0`只含前三项，必须保持 `valid=false` 和原失败原因，不作为有效可检索状态。

P/V/A/H沿用现有 `build_grid` 的坐标与单位：相对Frenet位置、相对Frenet速度、Frenet加速度、相对参考线朝向。中心格P/V为0，自车绝对速度另存。数值不降为float32，不预存归一化副本；归一化使用保存的 `distance_config`。同一压线对象可被多个格引用，检索按唯一对象ID去重，不能把它算成多个物体。

`metadata`仅保存版本、坐标说明、指标和观测配置/签名、原有范围及分配协议、原因、自车ID等描述。不会将完整旧JSON塞进字符串；不重复保存 `objects` 运动列表和四套矩阵，也不复制世界坐标、全体候选车道诊断、完整参考折线。场景回放仍使用原始场景/审核证据。

类别码支持道路障碍物的编码，并不自动修改现行检索准入类型；能否参与匹配仍由检索协议决定。

## 3. 逻辑3×3×16张量

`grid_tensor16` 按需返回 `(tensor, continuous_mask)`：

```text
通道 0..5   P_s, P_d, V_s, V_d, A_s, A_d
通道 6..7   sin(heading), cos(heading)
通道 8..11  车辆、自行车、行人、道路障碍物的独热编码
通道 12..15 occupied、empty、unknown、not_applicable的独热编码
```

`tensor.shape=(3,3,16)`，`continuous_mask.shape=(3,3,8)`。16通道为使用接口，不再另存144维展开副本；正弦/余弦从保存的float64原角度生成，避免反算角度影响阈值。现有精确检索使用数值状态及区域掩码计算距离，不将整个16通道张量直接做余弦相似度。

## 4. 代码接口

```python
from nine_grid.grid_storage import (
    GRID_STORAGE_VERSION, save_grid, load_grid,
    grid_to_arrays, grid_from_arrays, grid_tensor16,
)

save_grid(output_path, original_grid)
grid = load_grid(output_path)
tensor, continuous_mask = grid_tensor16(grid)
```

`save_grid/load_grid`支持文件路径和BytesIO。读取后恢复现有检索需要的字典接口，避免修改模型提示词、经验文本和规划接口；磁盘上的物理状态只保留数组资产。`Representations.grid_path`记录实际资产位置。

## 5. 另一个窗口转换离线库时的操作边界

本轮只修改运行端，不改现有正式离线库和离线构建器。后续转换应在独立暂存库中完成，再按原发布流程生成新的冻结版本。

1. 读取原 `features.json.paths.grid` 声明的旧JSON。调用本模块 `save_grid`；不要另写一套近似编码，也不要调用模型重生成经验。
2. 将新 `paths.grid` 指向NPZ，并声明 `grid_storage_version=lane-eight-array-1.0`。在线记录的对应位置为 `record.features.grid` 和 `record.grid_storage_version`。
3. 使用离线发布器更新内部 `sha256.grid`、记录资产校验清单、清单摘要及冻结/派生链。仅改扩展名或只改某一处hash无法通过运行端验收。
4. 新活动记录只保留一份八向数值资产。旧JSON留在原冻结版本/外部备份，不与新同名NPZ混放在活动在线记录内。不得删除原审核文本、当前帧及未来证据、features中必需资产或来源链；本次没有引入另一个简化离线库协议。
5. 保留原距离配置、观测配置、表示版本、范围与分配证据。历史字段缺失仍缺失，禁止用当前默认值伪装成新语义。缺失必要检索证据的记录仍按现行规则排除。
6. 对转换前后的数组、有效性、八向分数、对象配对、阈值准入和各通道Top1做回归。确认一致后再正式发布；DINO和OCC在纯格式转换时不需要重算。

运行端仍可只读旧JSON库。显式声明NPZ但文件损坏、版本不支持、数据形状错误或校验不符时明确拒绝，不回退JSON、不补造状态。离线NPZ仍走原完整审核和哈希验收链。

## 6. 验证命令

在对应分支目录和项目Python环境执行：

```bash
python scripts/check_grid_storage.py
python scripts/check_grid_memory_io.py
python scripts/check_grid_vector_retrieval.py
```

这些检查不调用模型、不连接CARLA、不修改正式记忆库。性能测试报告单列匹配计算耗时；磁盘压缩体量和实际速度取决于记录内容与库规模，不将本地构造基准当成服务器实验结果。

## 7. 本地验证记录（2026-11-19）

Python 3.12 / NumPy 2.4.6 环境下，1023组新旧匹配比较和19组完整检索对照通过；该次实测分数最大差0、对象配对变化0、Top1及准入变化0，覆盖0.2阈值附近、同分候选和未知状态。不同平台仍应运行上述检查，不作跨平台位级一致保证。

只读抽样24条既有离线记忆，在BytesIO中转换并回读：原八向JSON共204540字节，新NPZ共105895字节，自匹配及跨记录匹配结果一致，原库未改写。这是八向资产的大小变化，不是整个含图片记忆库缩小相同比例。

64条、每条8对象的构造池中，原标量匹配预热中位约75.7ms；新实现配合查询预处理复用约49.3ms。单独一对比较不保证更快，主要收益来自一次查询的重复校验/整理减少；此数据不含整库磁盘读取、DINO或CARLA仿真。
