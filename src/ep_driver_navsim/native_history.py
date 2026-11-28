"""Observed ego-history padding for the public NAVSIM v2 comfort convention.

No SDK, metric cache or future data is needed. Coordinates match controller
previews: positions/headings in the current rear-axle frame, dynamics in each
state's vehicle frame. The deliberately unusual endpoint exclusion follows
the pinned official history-comfort implementation without changing it.
"""
from __future__ import annotations

import numpy as np


VERSION='navsim-observed-native-history-1'


def build_native_history(scene):
    """Return the official 0.1 s past prefix from four observed ego frames."""
    identity=scene.diagnostics.get('navsim_identity',{})
    rows=scene.ego.history
    timestamps=np.asarray(identity.get('history_timestamps_us',[]),dtype=np.int64)
    if (scene.scenario!='navsim' or identity.get('token')!=scene.frame_id
            or identity.get('history_frames')!=4 or len(rows)!=4
            or timestamps.shape!=(4,) or np.any(np.diff(timestamps)<=0)
            or timestamps[-1]!=identity.get('timestamp_us')):
        raise ValueError('Native history requires four bound NAVSIM ego observations')
    relative=np.asarray([row['time'] for row in rows],float)
    if (not np.isfinite(relative).all() or np.any(relative>1e-8)
            or not np.allclose(relative,(timestamps-timestamps[-1])/1e6,rtol=0.,atol=1e-8)
            or any(row.get('observed') is not True for row in rows)):
        raise ValueError('Native history cannot consume future, unobserved or unbound states')
    headings=np.asarray([row['heading'] for row in rows],float)
    positions=np.asarray([row['position'] for row in rows],float)
    velocity=np.asarray([row['velocity'] for row in rows],float)
    acceleration=np.asarray([row['acceleration'] for row in rows],float)
    rear=scene.ego.rear_axle_to_center_m
    if (type(rear) not in (int,float) or not np.isfinite(rear) or rear<=0
            or headings.shape!=(4,) or positions.shape!=(4,2)
            or velocity.shape!=(4,2) or acceleration.shape!=(4,2)
            or not np.isfinite(np.r_[headings,positions.ravel(),velocity.ravel(),acceleration.ravel()]).all()):
        raise ValueError('Native history contains invalid vehicle geometry or observed dynamics')
    if (not np.allclose(positions[-1],scene.ego.position,rtol=0.,atol=1e-7)
            or abs((headings[-1]-scene.ego.heading+np.pi)%(2*np.pi)-np.pi)>1e-7):
        raise ValueError('Native history current pose differs from the source ego')
    dt=np.diff(timestamps)/1e6
    rate=np.r_[0.,np.diff(np.unwrap(headings))/dt]
    alpha=np.r_[0.,0.,np.diff(rate[1:])/dt[1:]]
    c,s=np.cos(headings),np.sin(headings)
    rear_positions=positions-rear*np.c_[c,s]
    origin=np.asarray(scene.ego.position)-rear*np.array([np.cos(scene.ego.heading),np.sin(scene.ego.heading)])
    ch,sh=np.cos(scene.ego.heading),np.sin(scene.ego.heading)
    local=(rear_positions-origin)@np.array([[ch,-sh],[sh,ch]])
    body_velocity=np.c_[c*velocity[:,0]+s*velocity[:,1],-s*velocity[:,0]+c*velocity[:,1]]
    body_acceleration=np.c_[c*acceleration[:,0]+s*acceleration[:,1],-s*acceleration[:,0]+c*acceleration[:,1]]
    body_velocity[:,1]-=rear*rate
    body_acceleration-=np.c_[-rear*rate**2,rear*alpha]
    knots=np.zeros((4,11),float)
    knots[:,:2]=local;knots[:,2]=np.unwrap(headings-scene.ego.heading)
    knots[:,3:5]=body_velocity;knots[:,5:7]=body_acceleration
    # NavSimScenario constructs every observed EgoState with steering and
    # angular dynamics zero. Do not inject inferred yaw into those SDK fields.
    query=np.arange(timestamps[0],timestamps[-1],100000,dtype=np.int64)[:-1]
    if len(query)==0:
        raise ValueError('Native history has no observed interpolation interval')
    key_time=timestamps-timestamps[0];query_time=query-timestamps[0]
    states=np.column_stack([np.interp(query_time,key_time,knots[:,axis]) for axis in range(11)])
    states[:,2]=(states[:,2]+np.pi)%(2*np.pi)-np.pi
    return dict(states=states,timestamps_us=query,provenance=dict(version=VERSION,
        frame_id=scene.frame_id,source='four_observed_ego_history_rows_only',
        future_observations_used=False,metric_cache_used=False,
        endpoint_policy='official_arange_start_to_current_exclusive_then_drop_last',
        history_frames=4,prefix_samples=len(query),
        coordinates='current_rear_axle_xy_heading; dynamics_vehicle_frame'))
