"""Measured comfort statistics do not hide sign reversals or raw peaks."""
import json
import math

import pytest

from carla_bridge.metrics.collector import MetricsCollector
from carla_bridge.metrics.run_reporter import _render_md, write_report


@pytest.fixture
def collector():
    result = MetricsCollector(None, None, dt_s=.1)
    result._check_red_light = lambda *_args: None
    result._check_wrong_way = lambda *_args: None
    return result


def sample(collector, t, ax=0, ay=0, yaw=0, speed=2):
    collector.on_tick(None, t, {
        "speed": speed, "acceleration": math.hypot(ax, ay),
        "acceleration_x": ax, "acceleration_y": ay, "yaw_rad": yaw,
    })


def test_equal_magnitude_acceleration_to_braking_is_not_zero_jerk(collector):
    sample(collector, .1, ax=2)
    sample(collector, .2, ax=-2)
    result = collector.summary()
    assert result["max_jerk_mps3"] == 0  # legacy magnitude-only field
    assert result["max_planar_jerk_mps3"] == 40
    assert result["max_longitudinal_jerk_mps3"] == 40
    assert result["max_longitudinal_acceleration_mps2"] == 2
    assert result["max_longitudinal_deceleration_mps2"] == 2
    assert result["max_planar_jerk_time_s"] == .2


def test_signed_acceleration_uses_native_carla_world_heading(collector):
    # Heading along world +y: -y is braking, +x is lateral.
    sample(collector, .1, ax=3, ay=-4, yaw=math.pi / 2)
    result = collector.summary()
    assert result["max_accel_mps2"] == 5
    assert result["max_longitudinal_acceleration_mps2"] == 0
    assert result["max_longitudinal_deceleration_mps2"] == 4
    assert result["max_lateral_acceleration_mps2"] == 3
    assert result["hard_braking_events"] == 1


def test_actual_timestamp_delta_and_two_samples_produce_jerk(collector):
    sample(collector, .1, ax=0)
    sample(collector, .3, ax=2)
    result = collector.summary()
    assert result["max_planar_jerk_mps3"] == 10  # not 20 from configured .1 s
    assert result["max_jerk_mps3"] == 10
    assert result["comfort_sampling"]["max_observed_dt_s"] == .2
    assert result["comfort_sampling"]["gapped_intervals"] == 1


def test_raw_startup_or_emergency_peak_is_retained_with_distribution(collector):
    for i, ax in enumerate([18.9, 0, 0, 0, 0]):
        sample(collector, .1 * (i + 1), ax=ax)
    result = collector.summary()
    assert result["max_accel_mps2"] == 18.9
    assert result["max_accel_time_s"] == .1
    assert result["p95_accel_mps2"] == 15.12
    assert result["rms_accel_mps2"] == pytest.approx(18.9 / math.sqrt(5), abs=.001)
    assert result["max_planar_jerk_mps3"] == 189
    assert result["rms_planar_jerk_mps3"] == 94.5
    assert result["p95_planar_jerk_mps3"] == 160.65


def test_braking_counts_segments_not_individual_ticks(collector):
    for i, ax in enumerate([0, -3, -4, -3, -2, -3, 0]):
        sample(collector, .1 * i, ax=ax)
    assert collector.summary()["hard_braking_events"] == 2


def test_stop_duration_requires_observed_continuous_static_samples(collector):
    for i in range(13):
        sample(collector, .1 * i, speed=0)
    # A gap must not invent 5.8 seconds of stopping, nor merge episodes.
    for i in range(12):
        sample(collector, 7 + .1 * i, speed=.1)
    sample(collector, 8.2, speed=1)
    result = collector.summary()
    assert result["stopped_time_s"] == 2.3
    assert result["stop_episodes"] == 2
    assert result["longest_stop_s"] == 1.2
    assert result["comfort_sampling"]["gapped_intervals"] == 1
    assert collector.summary() == result  # report reads cannot change counters


def test_legacy_input_does_not_fabricate_vector_measurements(collector):
    collector.on_tick(None, .1, {"speed": 1, "acceleration": 1})
    collector.on_tick(None, .2, {"speed": 1.2, "acceleration": 2})
    result = collector.summary()
    assert result["max_jerk_mps3"] == 10
    assert result["max_accel_mps2"] == 2
    for key in ("max_planar_jerk_mps3", "max_longitudinal_acceleration_mps2", "hard_braking_events"):
        assert result[key] is None
    assert result["comfort_sampling"]["longitudinal_acceleration_samples"] == 0


def test_missing_measurements_and_invalid_times_are_visible(collector):
    sample(collector, .1, ax=1)
    collector.on_tick(None, .2, {"speed": None, "acceleration": float("nan")})
    sample(collector, .3, ax=2)
    sample(collector, .3, ax=9)  # no divide by zero or fabricated derivative
    result = collector.summary()
    assert result["max_planar_jerk_mps3"] is None
    assert result["max_accel_mps2"] == 9  # invalid dt does not erase the acceleration peak
    assert result["comfort_sampling"]["invalid_intervals"] == 1
    assert result["comfort_sampling"]["acceleration_samples"] == 3
    json.dumps(result, allow_nan=False)


def test_report_explains_legacy_metric_and_preserves_unavailable_data(collector, tmp_path):
    collector.on_tick(None, .1, {"speed": 0, "acceleration": 0})
    md_path, json_path = write_report(collector.summary(), {"scenario": "test"}, str(tmp_path))
    from pathlib import Path
    md = Path(md_path).read_text(encoding="utf-8")
    assert "最大平面加速度模长" in md
    assert "旧 jerk 字段" in md
    assert "平面向量 jerk 最大值 / P95 / RMS: 未采集 / 未采集 / 未采集" in md
    assert "未平滑或剔除" in md
    assert "不区分合理等待与无必要停车" in md
    assert json.loads(Path(json_path).read_text(encoding="utf-8"))["metrics"]["max_planar_jerk_mps3"] is None
    assert "最大平面加速度模长: 18.90" in _render_md({}, {"max_accel_mps2": 18.9})
