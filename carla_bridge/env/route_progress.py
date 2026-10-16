"""Ordered route checkpoints; geometric proximity alone cannot skip a route loop."""
from __future__ import annotations

import math


def xyz(point):
    return (float(point.x), float(point.y), float(point.z))


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def project(point, start, end):
    delta = tuple(b - a for a, b in zip(start, end))
    length2 = sum(v * v for v in delta)
    t = max(0.0, min(1.0, sum((p - a) * d for p, a, d in zip(point, start, delta)) / length2)) if length2 else 0.0
    closest = tuple(a + t * d for a, d in zip(start, delta))
    return t, distance(point, closest)


class RouteProgress:
    """Visit checkpoints in order, within a configurable metric tolerance.

    Swept motion between ticks avoids missing checkpoints at normal driving speeds.
    The final checkpoint additionally requires the CURRENT position near the end.
    """

    def __init__(self, points, tolerance_m=2.0, corridor_m=5.0):
        if not math.isfinite(tolerance_m) or tolerance_m <= 0:
            raise ValueError("route completion tolerance must be positive and finite")
        if not math.isfinite(corridor_m) or corridor_m <= 0:
            raise ValueError("route tracking corridor must be positive and finite")
        self.points = []
        for point in points:
            p = xyz(point)
            if not all(math.isfinite(v) for v in p):
                raise ValueError("route contains a non-finite point")
            if not self.points or distance(p, self.points[-1]) > 1e-6:
                self.points.append(p)
        if len(self.points) < 2:
            raise ValueError("route must contain at least two distinct points")
        self.tolerance_m = tolerance_m
        self.corridor_m = corridor_m
        self.lengths = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            self.lengths.append(self.lengths[-1] + distance(a, b))
        self.index = 0
        self.started = False
        self.completed = False
        self.fraction = 0.0
        self._last = None

    def update(self, location):
        current = xyz(location)
        if self.completed:
            return 1.0
        if not all(math.isfinite(v) for v in current):
            return self.fraction
        previous = self._last if self._last is not None else current
        self._last = current
        if not self.started:
            if distance(current, self.points[0]) > self.corridor_m:
                return 0.0
            self.started = True
        last_motion_t = -1.0
        while self.index + 1 < len(self.points):
            target = self.points[self.index + 1]
            motion_t, error = project(target, previous, current)
            final = self.index + 1 == len(self.points) - 1
            if error > self.corridor_m or motion_t < last_motion_t:
                break
            if final and distance(current, target) > self.tolerance_m:
                break
            # Interior checkpoints must be passed longitudinally. A lateral
            # corridor permits lane changes without awarding progress at rest.
            direction = tuple(b - a for a, b in zip(self.points[self.index], target))
            if not final and sum((c - p) * d for c, p, d in zip(current, target, direction)) < -1e-6:
                break
            self.index += 1
            last_motion_t = motion_t
            if final:
                self.completed = True
                self.fraction = 1.0
                return 1.0
        segment_t, error = project(current, self.points[self.index], self.points[self.index + 1])
        if error <= self.corridor_m:
            along = self.lengths[self.index] + segment_t * (self.lengths[self.index + 1] - self.lengths[self.index])
            # Keep incomplete values visibly below 100%, including rounded reports.
            self.fraction = max(self.fraction, min(0.999, along / self.lengths[-1]))
        return self.fraction
