"""Same-frame feedback for the original paper planner; no driving controls."""
from copy import deepcopy
import math
import numpy as np

from nine_grid.contracts import plain


def trajectory_accepted(plan):
    """Quality scores and QP status alone cannot reject a safe waiting plan."""
    d = plan.diagnostics
    return bool(plan.valid and d.get('safety_feasible') is True
                and not d.get('emergency', False)
                and d.get('execution_mode') != 'emergency_unverified'
                and d.get('dense_validation', {}).get('valid', True))


def reflection_result_executable(plan, scene):
    """Transport/execution contract only, never an extra safety admission.

    MPC retains its own constraints and may return its emergency-braking result.
    Do not reinterpret that result as a certified normal/safe trajectory here.
    """
    if plan is None:
        return False
    try:
        controls=np.asarray(plan.controls,float);states=np.asarray(plan.states,float)
        return bool(plan.valid and plan.frame_id==scene.frame_id
            and math.isfinite(float(plan.time))
            and math.isclose(plan.time,scene.time,rel_tol=0.,abs_tol=1e-8)
            and controls.ndim==2 and controls.shape[1]==2 and len(controls)>0
            and states.shape==(len(controls)+1,4)
            and np.isfinite(controls).all() and np.isfinite(states).all())
    except (TypeError,ValueError,AttributeError):
        return False


def planner_snapshot(planner):
    """Only copy numerical history and reference state, never a CasADi solver."""
    if not callable(getattr(planner, 'export_reference_state', None)):
        return None
    return dict(reference=deepcopy(planner.export_reference_state()),
        previous=deepcopy(getattr(planner, 'previous', None)),
        previous_time=getattr(planner, 'previous_time', None),
        previous_reference_revision=getattr(planner, 'previous_reference_revision', None),
        has_solved=bool(getattr(planner, '_has_solved', False)))


def restore_candidate(config, snapshot, factory):
    candidate = factory(config)
    candidate.restore_reference_state(deepcopy(snapshot['reference']))
    candidate.previous = deepcopy(snapshot['previous'])
    candidate.previous_time = snapshot['previous_time']
    candidate.previous_reference_revision = snapshot['previous_reference_revision']
    candidate._has_solved = snapshot['has_solved']
    return candidate


def failure_feedback(plan, allowed_ids):
    """Summarize observed/forecast constraints without exporting unrelated actors."""
    d = plan.diagnostics
    allowed = set(allowed_ids)
    def visible_key(key):
        return not key.startswith('actor_clearance:') or key.split(':', 1)[1] in allowed
    minima = {str(k): float(v) for k, v in d.get('constraint_minima', {}).items()
              if visible_key(str(k)) and isinstance(v, (int, float)) and math.isfinite(v) and v < 0.}
    initial = {str(k): bool(v) for k, v in d.get('initial_safety_checks', {}).items()
               if visible_key(str(k)) and not v}
    dense = d.get('dense_validation', {})
    contacts = [{key: value for key, value in contact.items()
                 if key in ('actor_id', 'time_s', 'closing_speed_mps', 'ego_speed_mps')}
                for contact in dense.get('predicted_contacts', []) if contact.get('actor_id') in allowed]
    return plain(dict(frame_id=plan.frame_id, time=float(plan.time),
        trajectory_valid=bool(plan.valid), safety_feasible=d.get('safety_feasible'),
        emergency=d.get('emergency', False), execution_mode=d.get('execution_mode'),
        failed_initial_checks=initial, negative_constraint_margins=minima,
        dense_validation={key: dense.get(key) for key in ('valid', 'offroad_samples',
            'line_violation_samples', 'line_clearance_min_m', 'line_buffer_margin_min_m',
            'outside_potential_domain_samples', 'stop_line_passed') if key in dense},
        predicted_contacts=contacts, solver_status=d.get('solver_status', []),
        optimizer_termination=d.get('optimizer_termination'),
        rejected_semantics=d.get('rejected_semantics'),
        interpretation='initial checks are current observations; trajectory contacts are predictions, not future truth; soft buffer shortfall is not a physical crossing',
        instruction='Use the same frozen observations, legal single local goal and attention IDs. Numerical failure alone does not prove the driving intention is wrong; do not remove obstacles or weaken rules.'))
