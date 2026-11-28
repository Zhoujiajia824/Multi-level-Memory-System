"""Current-state-only preview of the pinned SDK's unchanged vehicle controller.

This SDK-side module deliberately has no scorer, metric-cache, map, annotation
or future-observation imports. It predicts our own response to a candidate
reference, not what other actors will do.
"""
from copy import deepcopy
import numpy as np

VERSION = 'navsim-native-controller-preview-1'


class NativeControllerPreview:
    """Batch Nx8x3 source-rear-axle references -> Nx41x11 native states.

    State columns are the pinned SDK StateIndex layout; positions and headings
    stay in the original current rear-axle frame, dynamics in the vehicle frame.
    """
    def __init__(self, packet, benchmark_profile):
        from .profiles import get_profile
        from .protocol import digest
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        self.profile = get_profile(benchmark_profile)
        identity = packet['identity']
        history = packet['ego']['history']
        timestamps = np.asarray(identity['history_timestamps_us'], dtype=np.int64)
        if (identity['history_frames'] != 4 or len(history) != 4 or timestamps.shape != (4,)
                or timestamps[-1] != identity['timestamp_us'] or np.any(np.diff(timestamps) <= 0)):
            raise ValueError('Controller preview requires the four bound current/history frames')
        history_time = np.asarray([row['time'] for row in history], float)
        if (not np.isfinite(history_time).all() or np.any(history_time > 1e-8)
                or any(row.get('observed') is not True for row in history)
                or not np.allclose(history_time, (timestamps-timestamps[-1])/1e6, rtol=0., atol=1e-8)):
            raise ValueError('Controller preview cannot consume future or unbound history')
        headings = np.asarray([row['heading'] for row in history], float)
        if not np.isfinite(headings).all():
            raise ValueError('Controller preview current headings are invalid')
        vehicle = get_pacifica_parameters()
        dimensions = packet['vehicle']
        expected = [vehicle.length, vehicle.width, vehicle.wheel_base, vehicle.rear_axle_to_center]
        actual = [dimensions[key] for key in ('length_m', 'width_m', 'wheelbase_m', 'rear_axle_to_center_m')]
        if not np.allclose(actual, expected, rtol=0., atol=1e-9):
            raise ValueError('Controller preview vehicle differs from the pinned Pacifica geometry')
        # The packet stores current centre dynamics with causal rigid-body yaw
        # terms. Undo precisely those terms; no estimation from future data.
        omega = np.diff(np.unwrap(headings))/(np.diff(timestamps)/1e6)
        alpha = (omega[-1]-omega[-2])/((timestamps[-1]-timestamps[-2])/1e6)
        rear = float(vehicle.rear_axle_to_center)
        if (not np.allclose(packet['ego']['position'],[rear,0.],rtol=0.,atol=1e-7)
                or abs(float(packet['ego']['heading']))>1e-8 or abs(float(headings[-1]))>1e-8):
            raise ValueError('Controller preview current packet is not in its source rear-axle frame')
        velocity = np.asarray(packet['ego']['velocity'], float)-[0., rear*omega[-1]]
        acceleration = np.asarray(packet['ego']['acceleration'], float)-[-rear*omega[-1]**2, rear*alpha]
        if velocity.shape != (2,) or acceleration.shape != (2,) or not np.isfinite([*velocity, *acceleration]).all():
            raise ValueError('Controller preview current dynamics are invalid')
        # This is exactly NavSimScenario's public initialization convention.
        # We neither change the official initial state nor inject history yaw
        # into the official simulator's zero steering/angular-rate state.
        self.ego = EgoState.build_from_rear_axle(StateSE2(0., 0., 0.),
            tire_steering_angle=0., vehicle_parameters=vehicle,
            time_point=TimePoint(int(timestamps[-1])), rear_axle_velocity_2d=StateVector2D(*velocity),
            rear_axle_acceleration_2d=StateVector2D(*acceleration))
        self.simulator = PDMSimulator(TrajectorySampling(num_poses=40, interval_length=.1))
        self.frame_id = str(identity['token'])
        t0 = int(timestamps[-1]); t0_s = self.ego.time_point.time_s
        # Reproduce original fixed timesteps and their integer-microsecond
        # rounding, then interpolate relative coordinates in a well-conditioned
        # frame. Tests bind this to the SDK transform/interpolation path.
        keys = ((np.arange(0., 4., .5)+.5+t0_s)*1e6).astype(np.int64)
        self._keys = np.r_[t0, keys]-t0
        query = ((np.arange(0., 4.+.1, .1)+t0_s)*1e6).astype(np.int64)
        self._query = np.clip(query, t0, keys[-1])-t0
        if len(self._query) != 41 or np.any(np.diff(self._keys) <= 0):
            raise ValueError('Controller preview time grid differs from the four-second SDK contract')
        self._provenance = dict(version=VERSION, frame_id=self.frame_id,
            input_packet_sha256=digest(packet),source_timestamp_us=int(timestamps[-1]),
            benchmark_profile=self.profile['id'], official_commit=self.profile['sdk_commit'],
            current_state_source='packet_current_dynamics_inverse_of_causal_center_yaw_terms',
            controller='unchanged_pinned_PDMSimulator_BatchLQRTracker',
            future_observations_used=False, metric_cache_used=False, scorer_used=False,
            coordinates='source_current_rear_axle_local_xy_heading; dynamics_vehicle_frame',
            dt_s=.1, samples=41)

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    def __call__(self, poses_batch):
        poses = np.asarray(poses_batch, dtype=np.float32).astype(np.float64)
        if (poses.ndim != 3 or poses.shape[1:] != (8, 3) or not 1 <= len(poses) <= 65
                or not np.isfinite(poses).all()):
            raise ValueError('Controller preview requires 1..65 finite eight-pose candidates')
        batch = len(poses)
        references = np.zeros((batch, 41, 11), dtype=np.float64)
        for index in range(batch):
            knots = np.vstack(([0., 0., 0.], poses[index]))
            knots[:, 2] = np.unwrap(knots[:, 2])
            for coordinate in range(3):
                references[index, :, coordinate] = np.interp(self._query, self._keys, knots[:, coordinate])
            references[index, :, 2] = (references[index, :, 2]+np.pi)%(2*np.pi)-np.pi
        result = np.asarray(self.simulator.simulate_proposals(references, self.ego), float)
        if result.shape != (batch, 41, 11) or not np.isfinite(result).all():
            raise RuntimeError('Native controller preview produced an invalid state batch')
        return result
