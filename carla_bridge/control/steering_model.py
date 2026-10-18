"""CARLA/PhysX inner-wheel commands and equivalent bicycle steering angles.

PhysX 4W gives the inner front wheel the requested angle and corrects the outer
wheel using Ackermann geometry. A bicycle needs the angle at the axle centre.
The geometry-bound mapping follows perfect Ackermann (validated on the default
Tesla); it does not model input smoothing or tire slip. Without axle geometry,
the historical linear wheel-angle mapping is retained.
"""
from bisect import bisect_right
import math


class SteeringModel:
    def __init__(self, max_steer_rad, steering_curve=(), *, wheelbase_m=None, front_track_m=None):
        limit = float(max_steer_rad)
        if not math.isfinite(limit) or not 0 < limit < math.pi / 2:
            raise ValueError("front-wheel steering limit must be between 0 and pi/2")
        curve = tuple((float(speed), float(scale)) for speed, scale in steering_curve)
        if any(not math.isfinite(speed) or speed < 0 or not math.isfinite(scale) or scale <= 0
               for speed, scale in curve):
            raise ValueError("steering curve requires finite nonnegative speeds and positive scales")
        if any(b[0] <= a[0] for a, b in zip(curve, curve[1:])):
            raise ValueError("steering curve speeds must be strictly increasing")
        if any(limit * scale >= math.pi / 2 for _, scale in curve):
            raise ValueError("scaled inner-wheel limit must be below pi/2")
        if (wheelbase_m is None) != (front_track_m is None):
            raise ValueError("Ackermann mapping requires both wheelbase and front track")
        self.wheelbase_m = None if wheelbase_m is None else float(wheelbase_m)
        self.front_track_m = None if front_track_m is None else float(front_track_m)
        if self.wheelbase_m is not None and (
                not math.isfinite(self.wheelbase_m) or not 1 < self.wheelbase_m < 6
                or not math.isfinite(self.front_track_m) or not .5 < self.front_track_m < 4):
            raise ValueError("Ackermann mapping requires valid vehicle axle geometry")
        self.physical_max_steer_rad = limit
        self.max_steer_rad = self.inner_to_bicycle(limit)
        self.curve = curve

    @property
    def mapping(self):
        return "physx_perfect_ackermann" if self.wheelbase_m is not None else "legacy_linear_wheel_angle"

    @staticmethod
    def _angle(value):
        value = float(value)
        if not math.isfinite(value) or abs(value) >= math.pi / 2:
            raise ValueError("steering angle must be finite and within (-pi/2, pi/2)")
        return value

    def inner_to_bicycle(self, angle):
        """Map a signed physical inner-wheel angle to the axle-centre angle."""
        angle = self._angle(angle)
        if self.wheelbase_m is None or angle == 0:
            return angle
        a = abs(angle)
        return math.copysign(math.atan2(self.wheelbase_m * math.sin(a),
            self.wheelbase_m * math.cos(a) + self.front_track_m * .5 * math.sin(a)), angle)

    def bicycle_to_inner(self, angle):
        """Inverse geometry; the caller clamps to the available centre angle."""
        angle = self._angle(angle)
        if self.wheelbase_m is None or angle == 0:
            return angle
        a = abs(angle)
        denominator = self.wheelbase_m * math.cos(a) - self.front_track_m * .5 * math.sin(a)
        if denominator <= 0:
            raise ValueError("bicycle angle exceeds the Ackermann geometry range")
        return math.copysign(math.atan2(self.wheelbase_m * math.sin(a), denominator), angle)

    def physical_limit_at(self, speed_mps):
        """Available inner-wheel angle; CARLA curve abscissas are km/h."""
        speed = abs(float(speed_mps)) * 3.6
        if not math.isfinite(speed):
            raise ValueError("steering conversion requires finite speed")
        if not self.curve:
            return self.physical_max_steer_rad
        index = bisect_right([sample[0] for sample in self.curve], speed)
        if index == 0:
            scale = self.curve[0][1]
        elif index == len(self.curve):
            scale = self.curve[-1][1]
        else:
            s0, f0 = self.curve[index - 1]
            s1, f1 = self.curve[index]
            scale = f0 + (f1 - f0) * (speed - s0) / (s1 - s0)
        return self.physical_max_steer_rad * scale

    def limit_at(self, speed_mps):
        """Available equivalent bicycle angle at this speed."""
        return self.inner_to_bicycle(self.physical_limit_at(speed_mps))

    def normalized_to_bicycle(self, steer, speed_mps):
        steer = float(steer)
        if not math.isfinite(steer) or abs(steer) > 1 + 1e-6:
            raise ValueError("normalized steering must be finite and within [-1, 1]")
        return self.inner_to_bicycle(max(-1, min(1, steer)) * self.physical_limit_at(speed_mps))

    def bicycle_to_normalized(self, angle, speed_mps):
        angle = self._angle(angle)
        limit = self.limit_at(speed_mps)
        bounded = max(-limit, min(limit, angle))
        return max(-1, min(1, self.bicycle_to_inner(bounded) / self.physical_limit_at(speed_mps)))

    def front_wheels_to_bicycle(self, front_left_rad, front_right_rad):
        """Equivalent angle from measured same-sign front wheels, preserving sign.

        For perfect Ackermann, cot(centre) is the mean of the two cotangents.
        Opposite wheel directions/toe or rear steering require a different model.
        """
        left, right = self._angle(front_left_rad), self._angle(front_right_rad)
        if abs(left) < 1e-8 and abs(right) < 1e-8:
            return 0.0
        if left * right <= 0:
            raise ValueError("measured front-wheel angles do not share one Ackermann turn direction")
        a, b = math.tan(left), math.tan(right)
        return math.atan(2 * a * b / (a + b))
