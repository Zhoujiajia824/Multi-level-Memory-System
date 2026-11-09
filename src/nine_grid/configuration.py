"""版本契约与参数校验：不允许看似可配置、实际未生效的特征格式修改。"""
from __future__ import annotations
import copy
import hashlib
import json
import math


GRID_METRIC_VERSION = 'gospa-inspired-region-partial-1.0'
OCC_METRIC_VERSION = 'observed-soft-iou-2.0'
RETRIEVAL_METRIC_VERSION = 'bev-occ-eight-channel-top1-3.0'
# 只改召回选择策略，表示与单通道距离未变；旧记录仍以当前策略重算，不恢复旧融合逻辑。
READABLE_RETRIEVAL_METRIC_VERSIONS = (RETRIEVAL_METRIC_VERSION, 'bev-occ-eight-joint-2.0')

# 仅限制VLM阅读层的文字；无字DINO图、实体集合和规划输入不受影响。
BEV_LABEL_DEFAULTS = {'max_dynamic_labels':8, 'max_vehicle_labels':5, 'max_vulnerable_labels':3}


def bev_label_config(options=None):
    """显示标签的独立参数，不复用检索或风险权重。"""
    if options is None:
        return dict(BEV_LABEL_DEFAULTS)
    if not isinstance(options,dict) or set(options)-set(BEV_LABEL_DEFAULTS):
        raise ValueError('BEV标签参数只支持动态、机动车和行人/自行车数量上限')
    result = {**BEV_LABEL_DEFAULTS, **options}
    for key,value in result.items():
        if type(value) is not int or not 0 <= value <= 16:
            raise ValueError('BEV标签数量必须为0至16的整数：'+key)
    return result

GRID_DISTANCE_DEFAULT = {
    "scales":{"s_m":20.,"d_m":3.5,"vs_mps":10.,"vd_mps":2.,"as_mps2":3.,"ad_mps2":2.,"heading_rad":math.pi/4},
    "weights":{"position":.35,"velocity":.35,"acceleration":.15,"heading":.15},
    "cutoff_c":.5,"lambda_T":1.,"lambda_E":.5,"min_joint_known_fraction":.75,
    "same_type_only":True,"require_intersecting_slots":True,
    "empty_non_ego_both_is_valid":True,"metric":"gospa_inspired_region_partial_assignment",
}

GRID_OBSERVATION_DEFAULT = {
    'empty_min_coverage':.95, 'empty_unknown_area_tolerance_m2':.25,
    'boundary_local_tolerance_m':.3, 'body_lane_area_tolerance_m2':1e-5,
    'geometry_area_epsilon_m2':.01,
}

OCC_DISTANCE_DEFAULT = {
    'min_common_known_fraction':.5, 'min_foreground_comparable_fraction':.95,
    'empty_min_known_fraction':.95, 'min_union_area_m2':.01,
    'area_tolerance_m2':.01, 'metric':'observed_soft_iou_with_confirmed_empty',
}

NATIVE_RENDER_FIELDS = {'screen_width','screen_height','scaling','centering_position',
                        'show_trajectories','real_time_rendering'}
NATIVE_SCENARIO_FIELDS = {
    'highway': {'initial_lane_id','ego_spacing','vehicles_density'},
    'merge': set(),
    'roundabout': {'incoming_vehicle_destination'},
    'intersection': set(),
}
SCENARIO_FIELDS = {'env_id','desired_speed_mps','initial_speed_mps','vehicles_count','lanes_count',
                   'destination','exit_distance_m','initial_longitudinal_m',
                   'connection_boundary_clip_radius_m','spawn_probability','native'}


def validate_scenario_settings(settings,name):
    """YAML 字段逐项验收，避免拼错或平台不支持的配置静默失效。"""
    if name not in NATIVE_SCENARIO_FIELDS or not isinstance(settings,dict):
        raise ValueError('场景设置必须为已实现平台的配置对象')
    unknown=set(settings)-SCENARIO_FIELDS
    if unknown:raise ValueError(f'{name} 存在未实现的场景字段：{sorted(unknown)}')
    for key in ('desired_speed_mps','exit_distance_m','connection_boundary_clip_radius_m'):
        if key in settings:_positive(settings[key],name+'.'+key)
    for key in ('initial_speed_mps','initial_longitudinal_m'):
        if key in settings:_nonnegative(settings[key],name+'.'+key)
    for key in ('vehicles_count','lanes_count'):
        if key in settings and (type(settings[key]) is not int or settings[key]<(1 if key=='lanes_count' else 0)):
            raise ValueError(f'{name}.{key} 必须为合法整数')
    if 'lanes_count' in settings and name!='highway':
        raise ValueError(f'{name} 的道路几何由原生模板提供，不支持 lanes_count')
    if 'spawn_probability' in settings:
        if name!='intersection':raise ValueError('动态交通生成概率仅用于 intersection')
        _unit_interval(settings['spawn_probability'],name+'.spawn_probability')
    native=settings.get('native',{})
    if not isinstance(native,dict):raise ValueError(f'{name}.native 必须为对象')
    unknown=set(native)-(NATIVE_RENDER_FIELDS|NATIVE_SCENARIO_FIELDS[name])
    if unknown:raise ValueError(f'{name}.native 不支持或禁止覆盖的字段：{sorted(unknown)}')
    for key in ('screen_width','screen_height'):
        if key in native and (type(native[key]) is not int or native[key]<64 or native[key]%2):
            raise ValueError(f'{name}.native.{key} 必须为不小于64的偶数，供原生画面和MP4使用')
    for key in ('scaling','ego_spacing','vehicles_density'):
        if key in native:_positive(native[key],name+'.native.'+key)
    for key in ('show_trajectories','real_time_rendering'):
        if key in native and type(native[key]) is not bool:raise ValueError(f'{name}.native.{key} 必须为布尔值')
    if 'centering_position' in native:
        value=native['centering_position']
        if not isinstance(value,list) or len(value)!=2:raise ValueError('centering_position 必须为二维比例')
        for item in value:_unit_interval(item,'centering_position')
    if native.get('initial_lane_id') is not None:
        lane=native['initial_lane_id']
        if type(lane) is not int or lane<0 or lane>=settings.get('lanes_count',4):
            raise ValueError('initial_lane_id 超出配置的高速车道范围')
    if native.get('incoming_vehicle_destination') is not None:
        destination=native['incoming_vehicle_destination']
        if type(destination) is not int or not 0<=destination<=2:
            raise ValueError('incoming_vehicle_destination 必须为 null 或0/1/2')
    if 'destination' in settings and (not isinstance(settings['destination'],str) or not settings['destination']):
        raise ValueError('自车导航 destination 必须为非空地图节点字符串')
    return copy.deepcopy(settings)


def highway_settings(config,name):
    base=validate_scenario_settings(config.get('scenarios',{}).get(name,{}),name)
    overrides=validate_scenario_settings(config.get('environment',{}),name)
    native={**base.get('native',{}),**overrides.get('native',{})}
    merged={**base,**overrides}
    if native:merged['native']=native
    result=validate_scenario_settings(merged,name)
    identifiers={'highway':'highway-v0','merge':'merge-v1','roundabout':'roundabout-v1','intersection':'intersection-v2'}
    if result.get('env_id',identifiers[name])!=identifiers[name]:raise ValueError(f'{name} 的环境实现固定为 {identifiers[name]}')
    return result

FIXED_CONTRACT = {
    "theory_version":"nine-grid-rag-mpc-1.0",
    "clock":{"history_hz":2,"history_seconds":3},
    "perception":{"partial_visibility_included":True,"fully_occluded_in_current_planning":False},
    "coordinates":{"x_axis":"forward","y_axis":"left","yaw_positive":"counterclockwise","projection_min_abs_one_minus_kappa_d":.1},
    "bev":{"front_m":70,"rear_m":30,"left_m":25,"right_m":25,"width_px":500,"height_px":1000,
           "meters_per_pixel":.1,"ego_pixel":[250,700],"speed_color_max_kmh":120,"history_seconds":3,"arrow_length_body_width_ratio":1},
    "occupancy":{"raw_resolution_m":.5,"retrieval_resolution_m":2,
                 "channels":["crosswalk","ego","vehicle","bicycle","pedestrian","static_obstacle"],
                 "score_channels":["crosswalk","vehicle","bicycle","pedestrian","static_obstacle"]},
    "dino":{"embedding":"CLS","dimension":768,"input_width_px":252,"input_height_px":504,"l2_normalize":True,"labels_in_embedding_image":False},
    "retrieval":{"per_channel_k":1,"selection":"per_channel_top1","merge_same_record":True,"require_all_channels":True,
                 "near_duplicate_removal":False,"cross_platform":True,"cross_road_type":True},
    "grid_distance":{"same_type_only":True,"require_intersecting_slots":True,"empty_non_ego_both_is_valid":True,
                     "metric":"gospa_inspired_region_partial_assignment"},
    "occupancy_distance":{"metric":"observed_soft_iou_with_confirmed_empty"},
    "risk":{"rank_weight":"1/rank"},
    "mpc":{"state":["x","y","psi","v"],"control":["a","delta"],"minimum_speed_mps":0,
           "constraint_policy":"bounded_controls_soft_safety_v1"},
    "models":{"max_total_attempts":6,"sdk_max_retries":0,"retry_backoff_s":[1,2,4,8,16],
              "attention_max_count":3,"answer_protocol_version":"local-goal-frenet-1.0","experience_max_characters":10000,
              "reflection_workers":1,"reflection_waiting_capacity":2},
    "memory":{"snapshot_scope":"episode","drain_before_next_episode":True,"auto_seed":False,"admit_requires_all_checks":True},
    "execution":{"no_feasible_plan":"return_current_bounded_candidate","independent_risk_alarm_with_feasible_plan":"execute_current_candidate",
                 "invalid_fast_answer":"clear_entire_semantics_same_mpc_base","network_failure_without_answer":"same_mpc_base_audit_only",
                 "reflection_control_effect":"none","reflection_failed_admission":"reject_no_second_reflection"},
}


def _merge(default,override):
    result=copy.deepcopy(default)
    for key,value in override.items():
        if isinstance(value,dict) and isinstance(result.get(key),dict):
            result[key]=_merge(result[key],value)
        else:
            result[key]=copy.deepcopy(value)
    return result


def grid_distance_config(config=None):
    if config is not None and not isinstance(config,dict):
        raise ValueError('九宫格距离配置必须为对象')
    if set(config or {})-set(GRID_DISTANCE_DEFAULT):
        raise ValueError('九宫格距离配置包含未实现字段；公式说明不得作为可配置参数')
    result=_merge(GRID_DISTANCE_DEFAULT,config or {})
    _check_fixed(result,FIXED_CONTRACT['grid_distance'],'grid_distance')
    if not isinstance(result['scales'],dict) or not isinstance(result['weights'],dict):
        raise ValueError('九宫格距离尺度和权重必须为对象')
    for key,value in result['scales'].items():
        _positive(value,f'grid_distance.scales.{key}')
    for key,value in result['weights'].items():
        _nonnegative(value,f'grid_distance.weights.{key}')
    if set(result['scales'])!=set(GRID_DISTANCE_DEFAULT['scales']) or set(result['weights'])!=set(GRID_DISTANCE_DEFAULT['weights']):
        raise ValueError('九宫格距离尺度和权重字段必须与版本契约一致')
    if not math.isclose(sum(result['weights'].values()),1.,abs_tol=1e-8):
        raise ValueError('九宫格四项距离权重之和必须为 1')
    _positive(result['cutoff_c'],'grid_distance.cutoff_c')
    _unit_interval(result['min_joint_known_fraction'],'grid_distance.min_joint_known_fraction')
    for key in ('lambda_T','lambda_E'):_nonnegative(result[key],'grid_distance.'+key)
    if sum(result['weights'][key] for key in ('velocity','acceleration','heading'))<=0:
        raise ValueError('自车运动差异必须有速度、加速度或朝向证据')
    return result


def grid_metric_signature(config=None):
    return _metric_signature(GRID_METRIC_VERSION,grid_distance_config(config))


def _metric_signature(version,cfg):
    # 同样的数值 1 和 1.0 应得到相同签名。
    def normalize(value):
        if isinstance(value,dict):return {key:normalize(item) for key,item in value.items()}
        if type(value) in (int,float):return float(value)
        return value
    return hashlib.sha256(json.dumps({'version':version,'parameters':normalize(cfg)},sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


def grid_observation_config(config=None):
    if config is not None and (not isinstance(config,dict) or set(config)-set(GRID_OBSERVATION_DEFAULT)):
        raise ValueError('八向观测配置包含未实现字段')
    result=_merge(GRID_OBSERVATION_DEFAULT,config or {})
    for key,value in result.items():
        (_unit_interval if key=='empty_min_coverage' else _nonnegative)(value,'grid_observation.'+key)
    if result['empty_min_coverage']<=0 or result['boundary_local_tolerance_m']<=0 or result['geometry_area_epsilon_m2']<=0:
        raise ValueError('八向覆盖阈值、边界与几何容差必须为正')
    return result


def grid_observation_signature(config=None):
    return _metric_signature('lane-observation-2.0',grid_observation_config(config))


def occupancy_distance_config(config=None):
    if config is not None and (not isinstance(config,dict) or set(config)-set(OCC_DISTANCE_DEFAULT)):
        raise ValueError('占据距离配置包含未实现字段')
    result=_merge(OCC_DISTANCE_DEFAULT,config or {})
    _check_fixed(result,FIXED_CONTRACT['occupancy_distance'],'occupancy_distance')
    for key in ('min_common_known_fraction','min_foreground_comparable_fraction','empty_min_known_fraction'):
        _unit_interval(result[key],'occupancy_distance.'+key)
        if result[key]<=0:raise ValueError('占据证据阈值必须为正')
    for key in ('min_union_area_m2','area_tolerance_m2'):_positive(result[key],'occupancy_distance.'+key)
    return result


def occupancy_metric_signature(config=None):
    return _metric_signature(OCC_METRIC_VERSION,occupancy_distance_config(config))


def _check_fixed(actual,expected,path=''):
    for key,value in expected.items():
        name=f'{path}.{key}' if path else key
        if key not in actual:
            continue
        current=actual[key]
        if isinstance(value,dict):
            if not isinstance(current,dict):raise ValueError(f'{name} 必须为对象')
            _check_fixed(current,value,name)
        elif isinstance(value,bool):
            if type(current) is not bool or current is not value:
                raise ValueError(f'{name} 是 v1 固定契约，必须为 {value}；修改需同步理论和特征版本')
        elif isinstance(value,(int,float)):
            if type(current) not in (int,float) or not math.isclose(current,value,abs_tol=1e-10):
                raise ValueError(f'{name} 是 v1 固定契约，必须为 {value}；修改需同步理论和特征版本')
        elif current!=value:
            raise ValueError(f'{name} 是 v1 固定契约，必须为 {value}；修改需同步理论和特征版本')


def _number(value,name):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError(f'{name} 必须是有限数值')


def _positive(value,name):
    _number(value,name)
    if value<=0:raise ValueError(f'{name} 必须大于零')


def _nonnegative(value,name):
    _number(value,name)
    if value<0:raise ValueError(f'{name} 不能为负')


def _unit_interval(value,name):
    _number(value,name)
    if not 0<=value<=1:raise ValueError(f'{name} 必须位于 [0,1]')


def validate_config(config):
    """返回独立副本，拒绝固定契约变更、时钟冲突和无意义的数值配置。"""
    if not isinstance(config,dict):raise ValueError('配置必须为对象')
    result=copy.deepcopy(config)
    def finite_values(value,path='配置'):
        if isinstance(value,dict):
            for key,item in value.items():finite_values(item,f'{path}.{key}')
        elif isinstance(value,list):
            for index,item in enumerate(value):finite_values(item,f'{path}[{index}]')
        elif type(value) in (int,float):_number(value,path)
    finite_values(result)
    for group in FIXED_CONTRACT:
        if isinstance(FIXED_CONTRACT[group],dict) and group in result and not isinstance(result[group],dict):
            raise ValueError(f'{group} 必须为对象')
    _check_fixed(result,FIXED_CONTRACT)
    if 'bev' in result:
        result['bev']['labels']=bev_label_config(result['bev'].get('labels'))
    if any(key in result.get('models',{}) for key in ('guide_min_count','guide_max_count')):
        raise ValueError('旧guide_min_count/guide_max_count已取消；只接受单个Frenet local_goal')
    mpc=result.get('mpc',{})
    for key in ('lag','guide_lag','guide_contour'):
        if key in mpc.get('weights',{}):
            raise ValueError(f'mpc.weights.{key} 已取消：局部路径为唯一跟踪参考，没有额外引导或滞后项')
    if 'dynamics_tolerance' in mpc:
        raise ValueError('mpc.dynamics_tolerance 已取消：状态由同一动力学积分直接生成')
    if 'qp_stopline_margin_factor' in mpc:
        raise ValueError('mpc.qp_stopline_margin_factor 已取消：停止截面改为软惩罚，不再生成QP硬预留')
    penalty_names={'actor_clearance','road_clearance','noncrossable_line_clearance','yield_stop_line','speed_limit','lateral_acceleration'}
    if not isinstance(mpc.get('safety_penalties',{}),dict) or set(mpc.get('safety_penalties',{}))-penalty_names:
        raise ValueError('mpc.safety_penalties 包含未实现的软惩罚项')
    for key,value in mpc.get('safety_penalties',{}).items():_positive(value,'mpc.safety_penalties.'+key)
    difference=mpc.get('geometry_difference_steps',{})
    if not isinstance(difference,dict) or set(difference)-{'position_m','heading_rad'}:
        raise ValueError('几何差分步长只支持 position_m 和 heading_rad')
    for key,value in difference.items():_positive(value,'mpc.geometry_difference_steps.'+key)
    clock=result.get('clock',{})
    physics=clock.get('physics_hz',50);planner=clock.get('planner_hz',10);fast=clock.get('fast_hz',1)
    for name,value in (('physics_hz',physics),('planner_hz',planner),('fast_hz',fast)):
        _positive(value,'clock.'+name)
    if 'episode_seconds' in clock:_positive(clock['episode_seconds'],'clock.episode_seconds')
    if not math.isclose(physics/planner,round(physics/planner),abs_tol=1e-8) or not math.isclose(planner/fast,round(planner/fast),abs_tol=1e-8) or fast>planner:
        raise ValueError('物理频率必须是规划频率的整数倍，规划频率必须是模型频率的整数倍')
    dt=mpc.get('step_s',.1)
    _positive(dt,'mpc.step_s')
    if not math.isclose(dt*planner,1.,abs_tol=1e-8):raise ValueError('MPC 步长必须等于规划周期')
    if mpc.get('physics_substeps',5)!=round(physics/planner):raise ValueError('MPC 子步数必须与物理时钟一致')
    n=mpc.get('horizon_steps',30)
    if type(n) is not int or n<2:raise ValueError('MPC 时域步数必须为至少 2 的整数')
    prediction=result.get('prediction',{})
    if not math.isclose(prediction.get('step_s',dt),dt,abs_tol=1e-8):raise ValueError('预测采样间隔必须与 MPC 步长一致')
    if not math.isclose(prediction.get('horizon_s',n*dt),n*dt,abs_tol=1e-8):
        raise ValueError('基础预测时域必须等于 MPC 时域；TTC 扩展由诊断时域单独定义')
    if prediction.get('diagnostic_horizon_s',3.9)<n*dt+max(result.get('quality',{}).get('ttc_offsets_s',[0,.3,.6,.9]))-1e-8:
        raise ValueError('诊断预测时域必须覆盖 MPC 时域及最长 TTC 外推')
    for group in ('normalization',):
        for key,value in mpc.get(group,{}).items():_positive(value,f'mpc.{group}.{key}')
    for key,value in mpc.get('weights',{}).items():_nonnegative(value,'mpc.weights.'+key)
    if set(mpc.get('weights',{}))-{'contour','heading','speed','acceleration','steering','jerk','steering_rate','risk','progress'}:
        raise ValueError('MPC 权重包含未实现的代价项')
    from .local_reference import DEFAULTS as reference_defaults
    local=result.get('local_reference',{})
    if not isinstance(local,dict) or set(local)-set(reference_defaults):
        raise ValueError('local_reference 必须为已实现的局部参考参数对象')
    for key,value in local.items():_positive(value,'local_reference.'+key)
    if local.get('prediction_check_s',3.)>prediction.get('horizon_s',n*dt)+1e-8:
        raise ValueError('局部候选检查不能超出基础预测时域')
    if local.get('geometry_margin_m',.001)>mpc.get('body_constraint_tolerance_m',.001)+1e-10:
        raise ValueError('局部参考几何容差不能大于原MPC车体容差')
    from .goals import DEFAULTS as goal_defaults
    goal=result.get('local_goal',{})
    if not isinstance(goal,dict) or set(goal)-set(goal_defaults):
        raise ValueError('local_goal 必须为已实现的目标参数对象')
    for key,value in goal.items():_positive(value,'local_goal.'+key)
    if goal.get('passed_tolerance_m',.001)>.005:
        raise ValueError('纵向到达容差不能大于5毫米，不能提前释放目标')
    if goal.get('minimum_delta_s_m',2.)>=goal.get('maximum_delta_s_m',70.):
        raise ValueError('目标纵向最小距离必须小于最大距离')
    if goal.get('projection_window_m',120.)<goal.get('maximum_delta_s_m',70.):
        raise ValueError('目标连续投影窗口不能小于目标纵向范围')
    for key in ('maximum_speed_mps','steering_abs_max_rad','jerk_abs_max_mps3','steering_rate_abs_max_radps','lateral_acceleration_abs_max_mps2','solver_wall_budget_s','cold_start_budget_s'):
        if key in mpc:_positive(mpc[key],'mpc.'+key)
    if 'body_constraint_tolerance_m' in mpc:_nonnegative(mpc['body_constraint_tolerance_m'],'mpc.body_constraint_tolerance_m')
    if 'warm_start_recovery_speed_mps' in mpc:
        _nonnegative(mpc['warm_start_recovery_speed_mps'],'mpc.warm_start_recovery_speed_mps')
    if mpc.get('acceleration_min_mps2',-6)>=0 or mpc.get('acceleration_max_mps2',3)<=0:
        raise ValueError('加速度边界必须包含制动负值和驱动正值')
    if 'restarts' in mpc:raise ValueError('mpc.restarts 已取消；实时SQP不执行多次慢NLP重启')
    for key in ('solver_max_iterations','qp_max_iterations','sqp_line_search_steps'):
        if key in mpc and (type(mpc[key]) is not int or mpc[key]<1):raise ValueError(f'mpc.{key} 必须为正整数')
    for key in ('qp_tolerance','sqp_trust_acceleration_mps2','sqp_trust_steering_rad','sqp_regularization'):
        if key in mpc:_positive(mpc[key],'mpc.'+key)
    geometry=result.get('map_geometry',{})
    for key in ('circular_angle_step_rad','coordinate_precision_m'):
        if key in geometry:_positive(geometry[key],'map_geometry.'+key)
    if geometry.get('circular_angle_step_rad',.02)>.05 or geometry.get('coordinate_precision_m',1e-9)>1e-6:
        raise ValueError('地图圆弧步长不能超过0.05rad，坐标舍入不能超过1微米；不能用粗精度放宽道路')
    result['grid_distance']=grid_distance_config(result.get('grid_distance'))
    result['grid_observation']=grid_observation_config(result.get('grid_observation'))
    result['occupancy_distance']=occupancy_distance_config(result.get('occupancy_distance'))
    if set(result.get('occupancy',{})) & {'min_known_fraction','min_union_area_m2'}:
        raise ValueError('旧occupancy比较阈值已迁至occupancy_distance，不能保留失效配置')
    retrieval=result.get('retrieval',{})
    if 'per_channel_k' in retrieval and type(retrieval['per_channel_k']) is not int:
        raise ValueError('retrieval.per_channel_k 是固定契约，必须为整数1')
    obsolete=set(retrieval)&{'weights','minimum_score','final_k'}
    if obsolete:
        raise ValueError('retrieval旧融合参数已取消：'+','.join(sorted(obsolete))+'；当前为各通道Top-1，合并同ID，不补齐')
    if 'minimum_grid_score' in retrieval:_unit_interval(retrieval['minimum_grid_score'],'retrieval.minimum_grid_score')
    quality=result.get('quality',{})
    if 'minimum_score' in quality:_unit_interval(quality['minimum_score'],'quality.minimum_score')
    for key,value in quality.get('weights',{}).items():_nonnegative(value,'quality.weights.'+key)
    if quality.get('weights') and set(quality['weights'])!={'progress','ttc','comfort'}:
        raise ValueError('质量权重必须完整包含 progress、ttc、comfort')
    if quality.get('weights') and sum(quality['weights'].values())<=0:raise ValueError('质量权重不能全部为零')
    offsets=quality.get('ttc_offsets_s',[0,.3,.6,.9])
    if not offsets or list(offsets)!=sorted(set(offsets)) or offsets[0]!=0:
        raise ValueError('TTC 外推时刻必须从零起严格递增')
    for value in offsets:_nonnegative(value,'quality.ttc_offsets_s')
    for key,value in prediction.items():
        if isinstance(value,dict):
            for sub,item in value.items():_positive(item,f'prediction.{key}.{sub}')
        elif type(value) in (int,float):_positive(value,'prediction.'+key)
    if prediction.get('lane_change_duration_min_s',1)>prediction.get('lane_change_duration_max_s',5):
        raise ValueError('换道候选时长下界不能超过上界')
    initial=prediction.get('pedestrian_initial_variances',[.1]*6)
    if not isinstance(initial,list) or len(initial)!=6:raise ValueError('行人初始方差必须按六维状态提供')
    for value in initial:_positive(value,'prediction.pedestrian_initial_variances')
    for key in ('reference_back_m','reference_front_m','lane_switch_hold_s'):
        if key in result.get('coordinates',{}):_positive(result['coordinates'][key],'coordinates.'+key)
    if 'lane_switch_overlap_fraction' in result.get('coordinates',{}):
        _unit_interval(result['coordinates']['lane_switch_overlap_fraction'],'coordinates.lane_switch_overlap_fraction')
    for name,settings in result.get('risk',{}).get('classes',{}).items():
        _nonnegative(settings.get('eta',1),'risk.classes.'+name+'.eta')
        _positive(settings.get('influence_m',1),'risk.classes.'+name+'.influence_m')
    if 'hard_body_margin_m' in result.get('risk',{}):
        raise ValueError('risk.hard_body_margin_m 已改名 body_clearance_margin_m；该余量用于软惩罚与审计，不是硬避碰约束')
    for key in ('clearance_floor_m','body_clearance_margin_m'):
        if key in result.get('risk',{}):_positive(result['risk'][key],'risk.'+key)
    if 'attention_increment' in result.get('risk',{}):_nonnegative(result['risk']['attention_increment'],'risk.attention_increment')
    for key in ('radius_m','visibility_angular_step_deg'):
        if key in result.get('perception',{}):_positive(result['perception'][key],'perception.'+key)
    if result.get('perception',{}).get('visibility_angular_step_deg',.25)>180:
        raise ValueError('几何可见性角采样间隔不能大于 180 度')
    for key,value in result.get('rules',{}).items():
        if key in ('yield_time_margin_s','stop_speed_mps'):
            raise ValueError(f'rules.{key} 没有对应执行规则，不能作为有效配置保留')
        _positive(value,'rules.'+key)
    for key in ('progress_floor_m','stopped_speed_mps'):
        if key in quality:_positive(quality[key],'quality.'+key)
    for key,value in quality.get('comfort',{}).items():
        if key=='long_accel_min_mps2':
            if value>=0:raise ValueError('舒适制动加速度下界必须为负')
        else:_positive(value,'quality.comfort.'+key)
    models=result.get('models',{})
    for key in ('fast_backend','reflection_backend'):
        if models.get(key,'off') not in ('off','api','server'):raise ValueError(f'models.{key} 只支持 off、api、server')
    for key in ('fast_timeout_s','reflection_timeout_s'):
        if key in models:_positive(models[key],'models.'+key)
    if 'answer_max_characters' in models and (type(models['answer_max_characters']) is not int or models['answer_max_characters']<1):
        raise ValueError('答案字符上限必须为正整数')
    memory=result.get('memory',{})
    if memory.get('mode','collect') not in ('collect','formal'):raise ValueError('记忆模式只支持 collect 或 formal')
    for key in ('read_enabled','write_enabled'):
        if key in memory and type(memory[key]) is not bool:raise ValueError(f'memory.{key} 必须为布尔值')
    if 'post_accident_association_s' in memory:_positive(memory['post_accident_association_s'],'memory.post_accident_association_s')
    env_ids={'highway':'highway-v0','merge':'merge-v1','roundabout':'roundabout-v1','intersection':'intersection-v2'}
    for name,settings in result.get('scenarios',{}).items():
        if name not in env_ids:raise ValueError(f'未实现的 highway 场景：{name}')
        validate_scenario_settings(settings,name)
        if 'env_id' in settings and settings['env_id']!=env_ids[name]:raise ValueError(f'{name} 的环境实现固定为 {env_ids[name]}')
    platform = result.get('platform', 'highway')
    if platform not in ('highway', 'carla'):
        raise ValueError('platform仅支持highway或carla')
    if platform == 'highway':
        highway_settings(result,result.get('scenario','highway'))
    return result
