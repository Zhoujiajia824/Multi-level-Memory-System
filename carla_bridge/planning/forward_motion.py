"""Keep native forward-gear rollouts from numerically braking into reverse.

The native Euler integrator has no gear/zero-speed boundary. A large measured
braking acceleration can carry its low-pass-filtered velocity through zero,
then simulate reverse travel even though IDM requests only forward targets.
Normal forward steps remain bit-for-bit native. Only zero crossings integrate
to the stopping point and hold zero velocity for the rest of that sample.
"""
import numpy as np


def build_forward_motion_model(api, vehicle, max_steering_angle):
    si = api.StateIndex

    class ForwardGearNativeBicycle(api.BatchKinematicBicycleModel):
        def propagate_state(self, states, command_states, sampling_time):
            result = super().propagate_state(states, command_states, sampling_time)
            dt = sampling_time.time_s
            velocity = states[:, si.VELOCITY_X]
            crossing = (velocity >= 0.) & (result[:, si.VELOCITY_X] < 0.)
            if not crossing.any():
                return result
            v = velocity[crossing]
            accel = result[crossing, si.ACCELERATION_X]
            stop_time = np.clip(-v / accel, 0., dt)
            distance = v * stop_time + .5 * accel * stop_time**2
            heading = states[crossing, si.HEADING]
            result[crossing, si.X] = states[crossing, si.X] + distance * np.cos(heading)
            result[crossing, si.Y] = states[crossing, si.Y] + distance * np.sin(heading)
            # Match the native step's steering update; shorten only travel.
            updated = self._update_commands(states[crossing], command_states[crossing], sampling_time)
            heading += distance * np.tan(updated[:, si.STEERING_ANGLE]) / self._vehicle.wheel_base
            result[crossing, si.HEADING] = (heading + np.pi) % (2*np.pi) - np.pi
            result[crossing, si.VELOCITY_X] = 0.
            result[crossing, si.ACCELERATION_X] = -v / dt
            result[crossing, si.ANGULAR_VELOCITY] = 0.
            result[crossing, si.ANGULAR_ACCELERATION] = -states[crossing, si.ANGULAR_VELOCITY] / dt
            return result

    return ForwardGearNativeBicycle(vehicle=vehicle, max_steering_angle=max_steering_angle)
