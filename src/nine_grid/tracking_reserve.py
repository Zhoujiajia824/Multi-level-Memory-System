"""Causal recovery schedules for nonphysical NAVSIM execution reserves.

Witnesses only determine when to demand a reserve. They never select a plan,
read scoring futures, certify safety, or replace an optimizer result by braking.
"""
from __future__ import annotations

import time
import numpy as np
import shapely

from .realtime_solver import circle_gap, rectangle_gap


def smooth_reserve_ramp(times, physical_floor, target, recovery_s, delay_s=0.):
    """C2 quintic ramp with zero first/second derivatives at both ends."""
    times=np.asarray(times,float)
    if np.ndim(recovery_s)==0 and recovery_s<=0.:
        return np.full(times.shape,float(target))
    u=np.clip((times-delay_s)/recovery_s,0.,1.)
    return physical_floor+(target-physical_floor)*u**3*(10.-15.*u+6.*u*u)


def signed_body_road_clearance(polygons,road,boundary=None):
    """Complete footprint gap, including concave corners and holes."""
    boundary=road.boundary if boundary is None else boundary
    polygons=np.asarray(polygons,dtype=object)
    outside=~np.asarray(shapely.covers(road,polygons),bool)
    distances=np.asarray(shapely.distance(polygons,boundary),float)
    if np.any(outside):
        distances=distances.copy()
        distances[outside]=-np.sqrt(np.maximum(0.,np.asarray(
            shapely.area(shapely.difference(polygons[outside],road)),float)))
    return distances


def body_road_clearance_partial(states,body_vertices,length,width,road,boundary,
                                position_step=1e-5,heading_step=1e-5):
    """Differentiate the identical complete-footprint signed gap in state space."""
    partial=np.zeros((len(states),4))
    for axis,epsilon in enumerate((position_step,position_step,heading_step)):
        plus=states.copy();minus=states.copy()
        plus[:,axis]+=epsilon;minus[:,axis]-=epsilon
        positive=signed_body_road_clearance(shapely.polygons(body_vertices(plus,length,width)),road,boundary)
        negative=signed_body_road_clearance(shapely.polygons(body_vertices(minus,length,width)),road,boundary)
        partial[:,axis]=(positive-negative)/(2*epsilon)
    return partial


def build_tracking_reserve_recovery(*, initial, controls, seed_controls, projection,
        dense_rollout, body_vertices, length, width, targets, road, accepted_road,
        boundary, n, dt, substeps, min_a, max_a, max_delta, body_clearance,
        actor_reserve, road_reserve, tolerance, planning_deadline,witness_rule_mask=None):
    """Use current clearance and finite actuator-bounded model witnesses.

    A sufficiently separated entity keeps its full hard reserve from t=0.
    Only initially missing reserves recover from the unchanged physical floor.
    All recovering margins must be jointly met by one bounded witness at every
    physical substep. No joint witness is not an infeasibility proof: retain
    physical hard floors and full soft targets for unproven initial shortfalls.
    """
    started=time.perf_counter();times=np.arange(n+1)*dt
    target_actor=body_clearance+actor_reserve
    initial_polygon=shapely.polygons(body_vertices(initial[None,:],length,width))
    initial_road=float(signed_body_road_clearance(initial_polygon,road,boundary)[0])

    def entity_gap(states,actor,sample):
        if actor.kind=='pedestrian':
            return circle_gap(states,sample[:,:2],actor.radius or actor.width/2,length,width)[0]
        return rectangle_gap(states,sample,length,width,actor.length,actor.width)[0]

    initial_actor={actor.id:float(entity_gap(initial[None,:],actor,sample[:1])[0])
                   for actor,sample,_ in targets}
    specifications={'road':(initial_road,0.,road_reserve)}
    specifications.update({actor.id:(initial_actor[actor.id],body_clearance,target_actor) for actor,_,_ in targets})
    deficient={key for key,(gap,floor,target) in specifications.items() if gap<target-tolerance and target>floor}
    witnesses=[];budget_limited=False;joint_witness=None;joint_curves={};joint_parameters={}
    if deficient:
        # Bound this preparation by a fraction of the same remaining planning
        # budget. This affects evidence coverage, not the requested reserve.
        witness_deadline=started+max(0.,planning_deadline-started)/4.
        dense_times=np.arange(n*substeps+1)*dt/substeps
        # Delays/completions use only the existing control-time grid. The
        # finite search introduces no new risk radius or recovery duration.
        delay_indices,end_indices=np.triu_indices(n,k=0)
        end_indices=end_indices+1
        delays=delay_indices*dt;durations=(end_indices-delay_indices)*dt
        normalized_curves=smooth_reserve_ramp(dense_times[None,:],0.,1.,durations[:,None],delays[:,None])
        samples={actor.id:np.stack([np.interp(dense_times,times,sample[:,j]) for j in range(4)],axis=1)
                 for actor,sample,_ in targets}
        def candidates():
            yield 'current_nominal',controls
            if time.perf_counter()<witness_deadline:
                yield from seed_controls()
            for accel in (min_a,max_a):
                for steer in (-max_delta,0.,max_delta):
                    yield f'actuator_extreme_{accel:g}_{steer:g}',np.tile([accel,steer],(n,1))
        seen=set()
        for name,intent in candidates():
            if time.perf_counter()>=witness_deadline:
                budget_limited=True;break
            bounded,recovery=projection(intent)
            key=bounded.tobytes()
            if key in seen:continue
            seen.add(key)
            states=dense_rollout(bounded,initial)
            polygons=shapely.polygons(body_vertices(states,length,width))
            inside=shapely.covers(accepted_road,polygons)
            road_clear=signed_body_road_clearance(polygons,road,boundary)
            feasible=inside&(road_clear>=-tolerance);gaps={'road':road_clear}
            for actor,_,_ in targets:
                gap=entity_gap(states,actor,samples[actor.id]);gaps[actor.id]=gap
                feasible &= gap>=body_clearance-tolerance
            if witness_rule_mask is not None:
                feasible &= np.asarray(witness_rule_mask(states,polygons),bool)
            if recovery['recovery_required']:
                feasible[:]=False
            for key,(_,floor,target) in specifications.items():
                if key not in deficient:feasible &= gaps[key]>=target-tolerance
            witnesses.append(dict(name=name,physical_and_initially_full_reserves_met=bool(feasible.all())))
            if not feasible.all():continue
            curves={};parameters={}
            for key in sorted(deficient):
                if time.perf_counter()>=witness_deadline:
                    budget_limited=True;break
                _,floor,target=specifications[key]
                valid=np.flatnonzero(np.all(floor+(target-floor)*normalized_curves<=gaps[key][None,:]+tolerance,axis=1))
                if not len(valid):break
                # Earliest completion, then latest feasible start; all selected
                # per-object schedules are checked against this SAME witness.
                chosen=min(valid,key=lambda j:(end_indices[j],-delay_indices[j]))
                curves[key]=floor+(target-floor)*normalized_curves[chosen]
                parameters[key]=(float(delays[chosen]),float(durations[chosen]))
            if len(curves)==len(deficient):
                joint_witness=name;joint_curves=curves;joint_parameters=parameters
                break

    def schedule(initial_gap,physical_floor,target,kind):
        if target<=physical_floor or initial_gap>=target-tolerance:
            return np.full(n+1,target),dict(initial_clearance_m=initial_gap,full_from_start=True)
        if joint_witness is None:
            margin=np.full(n+1,physical_floor);delay=duration=None
        else:
            margin=joint_curves[kind][::substeps];delay,duration=joint_parameters[kind]
        return margin,dict(
            initial_clearance_m=initial_gap,full_from_start=False,recovery_s=duration,recovery_delay_s=delay,
            bounded_model_witness=joint_witness,witness_found=joint_witness is not None,
            recovery_unproven=joint_witness is None,
            fallback='physical_hard_floor_and_unchanged_full_soft_target' if joint_witness is None else None,
            no_witness_is_not_an_infeasibility_proof=True,
            initial_physical_shortfall_m=max(0.,physical_floor-initial_gap))

    road_margin,road_evidence=schedule(initial_road,0.,road_reserve,'road')
    actor_margins={};actor_evidence={}
    for actor,_,_ in targets:
        margin,evidence=schedule(initial_actor[actor.id],body_clearance,target_actor,actor.id)
        actor_margins[actor.id]=margin;actor_evidence[actor.id]=evidence
    evidence=dict(policy='causal_joint_bounded_model_witness_delayed_C2_reserve_recovery_v2',
        formula='floor+(target-floor)*(10u^3-15u^4+6u^5); u=clip((t-delay)/T,0,1)',
        times_s=times.tolist(),road_total_hard_margin_m=road_margin.tolist(),
        actor_total_hard_margin_m={key:value.tolist() for key,value in actor_margins.items()},
        road=road_evidence,actors=actor_evidence,witness_count=len(witnesses),
        joint_bounded_model_witness=joint_witness,joint_recovery_proven_on_grid=joint_witness is not None,
        recovery_unproven=bool(deficient and joint_witness is None),witness_substep_s=dt/substeps,
        witness_budget_limited=budget_limited,preparation_seconds=time.perf_counter()-started,
        physical_boundaries_unchanged=True,postsolve_veto=False,
        current_and_causal_prediction_only=True,
        witness_constraints='projected_actuator_limits_physical_road_entities_and_same_source_rule_mask' if witness_rule_mask is not None else 'projected_actuator_limits_and_physical_road_entities;rule_mask_not_provided',
        witness_rule_mask_provided=witness_rule_mask is not None,
        limitation='Finite bounded bicycle-model witnesses, not official-controller reachability or a safety certificate')
    return road_margin,actor_margins,evidence
