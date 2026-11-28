"""Keep requested poses, nominal MPC and predicted controller execution distinct."""
from copy import deepcopy
from dataclasses import replace

import numpy as np

from .execution_rpc import validate_poses, validate_states
from .protocol import digest


def native_center_kinematics(states, rear):
    """Physical rigid-body centre quantities; separate from SDK comfort conventions."""
    states = np.asarray(states, float)
    vx = states[..., 3]
    vy = states[..., 4]+rear*states[..., 9]
    speed = np.hypot(vx, vy)
    beta = np.arctan2(vy, vx)
    ax = states[..., 5]-rear*states[..., 9]**2
    ay = states[..., 6]+rear*states[..., 10]
    tangential = ax*np.cos(beta)+ay*np.sin(beta)
    lateral = -ax*np.sin(beta)+ay*np.cos(beta)
    return dict(speed=speed, beta=beta, body_acceleration=np.stack((ax,ay),axis=-1),
                tangential_acceleration=tangential,lateral_acceleration=lateral)


def expected_execution_plan(scene, nominal, refinement):
    states = validate_states([refinement['expected_states']], 1)[0]
    validate_poses([refinement['submitted_poses']])
    rear = float(scene.ego.rear_axle_to_center_m)
    local_heading = states[:, 2]
    local_center = states[:, :2] + rear*np.c_[np.cos(local_heading), np.sin(local_heading)]
    angle = float(scene.ego.heading)
    c,s = np.cos(angle),np.sin(angle)
    origin = np.asarray(scene.ego.position)-rear*np.array([c,s])
    center = local_center@np.array([[c,s],[-s,c]])+origin
    heading = (local_heading+angle+np.pi)%(2*np.pi)-np.pi
    # Native state vx is longitudinal rear speed. Rotation adds lateral centre
    # velocity; no guessed slip angle or ideal-model re-integration is used.
    center_speed = native_center_kinematics(states,rear)['speed']
    numerical_states = np.c_[center, heading, center_speed]
    # This PlanResult's acceleration describes its centre-speed change. Native
    # rear acceleration and all original actuator fields remain in the record
    # and are checked separately below, rather than mislabelled as ideal MPC.
    controls = np.c_[np.diff(center_speed)/.1, states[:-1, 7]]
    diagnostic = deepcopy(nominal.diagnostics)
    diagnostic.update(trajectory_source='predicted_official_controller_execution',
        native_preview_vehicle=dict(rear_axle_to_center_m=rear,
            wheelbase_m=float(scene.ego.wheelbase_m)),
        nominal_mpc_diagnostics_retained_in='planning_evidence.plan_or_reference',
        nominal_mpc_safety_feasible=bool(nominal.diagnostics.get('safety_feasible', False)),
        execution_refinement=deepcopy(refinement['diagnostics']))
    return replace(nominal, states=numerical_states, controls=controls, diagnostics=diagnostic)


def audit_expected_execution(quality, expected, refinement, config):
    gates = quality.metrics.get('gates', {})
    native = validate_states([refinement['expected_states']],1)[0]
    p = config['mpc']
    rear = expected.diagnostics['native_preview_vehicle']['rear_axle_to_center_m']
    kinematics = native_center_kinematics(native,rear)
    control_minima = dict(
        nonnegative_speed=float(np.min(native[1:,3])),
        acceleration_lower=float(np.min(native[1:,5]-p['acceleration_min_mps2'])),
        acceleration_upper=float(np.min(p['acceleration_max_mps2']-native[1:,5])),
        steering=float(np.min(p['steering_abs_max_rad']-np.abs(native[1:,7]))),
        steering_rate=float(np.min(p['steering_rate_abs_max_radps']-np.abs(native[1:,8]))),
        jerk=float(np.min(p['jerk_abs_max_mps3']-np.abs(np.diff(native[:,5])/.1))),
        lateral_acceleration=float(np.min(p['lateral_acceleration_abs_max_mps2']-
                                          np.abs(kinematics['lateral_acceleration'][1:]))))
    controls_ok = all(value >= -1e-6 for value in control_minima.values())
    diagnostic = dict(available=bool(gates), source='causal_native_controller_preview',
        prediction_only=True, actual_official_evaluation=False, veto_enabled=False,
        submission_replaced=False, execution_duration_s=4.0, executed_control_count=40,
        audit_samples=201, sample_step_s=.02,
        sampling='linear interpolation of 41 preview states; not a continuous-time guarantee',
        collision=bool(quality.collision), offroad=not gates.get('on_road', False),
        alarm=bool(not controls_ok or quality.collision or not gates.get('on_road', False)
                   or not gates.get('rules', False) or not gates.get('correct_direction', False)),
        physical_control_minima=control_minima,physical_controls_satisfied=controls_ok,
        evaluated_actor_ids=quality.metrics.get('evaluated_actor_ids', []),
        optimization_actor_ids=expected.diagnostics.get('planning_object_selection', {}).get('selected_ids', []),
        controller_diagnostics=deepcopy(refinement['diagnostics']),
        executed_policy='Expected official execution is diagnosed; no rejection or braking replacement')
    if diagnostic['alarm']:
        expected.diagnostics.update(safety_feasible=False, execution_mode='diagnostic_unverified',
            diagnostic_only_safety_failure=True)
        if quality.metrics.get('gates') is not None:
            quality.metrics['gates']['planner_safety'] = False
            if quality.score is not None:
                quality.score = 0.
    return diagnostic


def validate_refinement_record(value, packet, poses, benchmark_profile=None):
    """Checkpoint proof binds the final request to its own preview evidence."""
    if (not isinstance(value, dict) or value.get('schema') != 'navsim-execution-refinement-evidence-1'
            or value.get('source_token') != packet['identity']['token']
            or value.get('source_timestamp_us') != packet['identity']['timestamp_us']
            or value.get('uses_future_ground_truth') is not False
            or value.get('uses_metric_cache') is not False):
        raise ValueError('Execution refinement is not bound to the causal source packet')
    if benchmark_profile is not None and value.get('benchmark_profile') != benchmark_profile:
        raise ValueError('Execution refinement belongs to another benchmark profile')
    main = value['plan']
    requested = validate_poses([main['submitted_poses']])[0]
    if not np.array_equal(np.asarray(poses, np.float32), requested):
        raise ValueError('Submitted trajectory differs from execution refinement evidence')
    history=packet['ego']['history']
    timestamps=np.asarray(packet['identity']['history_timestamps_us'],np.int64)
    if len(history)!=4 or timestamps.shape!=(4,) or np.any(np.diff(timestamps)<=0):
        raise ValueError('Execution preview requires four bound source-history frames')
    headings=np.asarray([row['heading'] for row in history],float)
    history_time=np.asarray([row['time'] for row in history],float)
    if (timestamps[-1]!=packet['identity']['timestamp_us'] or not np.isfinite(headings).all()
            or any(row.get('observed') is not True for row in history)
            or not np.allclose(history_time,(timestamps-timestamps[-1])/1e6,rtol=0.,atol=1e-8)
            or abs(headings[-1])>1e-8):
        raise ValueError('Execution preview history is not bound to the current rear axle')
    omega=np.diff(np.unwrap(headings))/(np.diff(timestamps)/1e6)
    alpha=(omega[-1]-omega[-2])/((timestamps[-1]-timestamps[-2])/1e6)
    rear=float(packet['vehicle']['rear_axle_to_center_m'])
    expected_initial=np.zeros(11)
    expected_initial[3:5]=np.asarray(packet['ego']['velocity'],float)-[0.,rear*omega[-1]]
    expected_initial[5:7]=np.asarray(packet['ego']['acceleration'],float)-[-rear*omega[-1]**2,rear*alpha]
    for key in ('plan', 'reference'):
        item = value[key]
        validate_poses([item['submitted_poses']])
        native=validate_states([item['expected_states']], 1)[0]
        if not np.allclose(native[0],expected_initial,rtol=0.,atol=1e-7):
            raise ValueError('Archived controller preview initial state differs from its source packet')
        if not isinstance(item.get('diagnostics'), dict):
            raise ValueError('Execution refinement lacks optimizer diagnostics')
        diagnostic = item['diagnostics']
        if (diagnostic.get('enabled') is not True or diagnostic.get('status') != 'complete'
                or diagnostic.get('frame_id') != packet['identity']['token']
                or diagnostic.get('scorer_or_metric_cache_used') is not False
                or diagnostic.get('future_observations_used') is not False
                or diagnostic.get('postsolve_safety_veto') is not False):
            raise ValueError('Execution refinement has inconsistent causal optimizer diagnostics')
    provenance = value.get('controller_provenance')
    if (not isinstance(provenance, dict)
            or provenance.get('input_packet_sha256') != digest(packet)
            or provenance.get('frame_id') != packet['identity']['token']
            or provenance.get('source_timestamp_us') != packet['identity']['timestamp_us']
            or provenance.get('future_observations_used') is not False
            or provenance.get('metric_cache_used') is not False
            or provenance.get('scorer_used') is not False
            or (benchmark_profile is not None and provenance.get('benchmark_profile') != benchmark_profile)):
        raise ValueError('Execution refinement lacks controller provenance')
    return requested
