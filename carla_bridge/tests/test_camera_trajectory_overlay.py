"""Camera-extrinsic, clipping and color contracts without a CARLA connection.

The expected pixels below come from the pinhole equation, not the renderer's
own projection helpers. Coordinates use UE x-forward/y-right/z-up.
"""
from copy import deepcopy
import math

import numpy as np
import pytest

from carla_bridge.visualization.camera_overlay import draw_trajectory_overlay


WIDTH, HEIGHT = 640, 360
BACKGROUND = 218
RGB = [240, 80, 40]


class CaptureTransform:
    """Only the transform attached to a captured image is available."""
    def __init__(self, xyz=(0., 0., 0.), yaw_degrees=0.):
        yaw = math.radians(yaw_degrees)
        c, s = math.cos(yaw), math.sin(yaw)
        self.rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        self.translation = np.asarray(xyz, dtype=float)
        self.inverse = np.eye(4)
        self.inverse[:3, :3] = self.rotation.T
        self.inverse[:3, 3] = -self.rotation.T @ self.translation
        self.inverse_reads = 0

    def get_inverse_matrix(self):
        self.inverse_reads += 1
        return self.inverse.tolist()

    def to_world(self, camera_points):
        return (np.asarray(camera_points) @ self.rotation.T + self.translation).tolist()


def image():
    return np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)


def path(points, *, kind="candidate", rgb=None, width_m=.1, dashed=False, **kwargs):
    return {"kind": kind, "rgb": RGB.copy() if rgb is None else list(rgb),
            "points_world": points, "width_m": width_m, "dashed": dashed,
            "phase_m": 0., "marker": False, **kwargs}


def display(paths, **kwargs):
    return {"style": "paper", "paths": paths, "dash_length_m": 1.2,
            "dash_gap_m": .8, "outline_width_m": 0.,
            "endpoint_markers": False, **kwargs}


def changed(pixels):
    return np.any(pixels != BACKGROUND, axis=-1)


def color_mask(pixels, rgb=RGB):
    # An exact full-intensity BGR core must exist; antialiased edges may blend.
    return np.all(pixels == np.asarray(rgb[::-1], dtype=np.uint8), axis=-1)


def near_pixel(mask, x, y, radius=2):
    return mask[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1].any()


@pytest.mark.parametrize("fov,expected_x", [(90., 384), (60., 431)])
def test_known_pinhole_projection_and_rgb_to_bgr(fov, expected_x):
    pixels = image()
    transform = CaptureTransform()
    audit = draw_trajectory_overlay(pixels, display([path([[10., 2., -1.], [10., 2., 1.]])]),
                                    transform, fov_degrees=fov)
    colored = color_mask(pixels)
    # fx = width/(2*tan(horizontal_fov/2)); u=cx+fx*y/x, v=cy-fx*z/x.
    assert near_pixel(colored, expected_x, HEIGHT//2)
    assert near_pixel(colored, expected_x, round(HEIGHT/2 - (WIDTH/2)/math.tan(math.radians(fov/2))/10))
    assert not colored[:, :WIDTH//2].any()
    assert audit["drawn_paths"] == 1 and audit["drawn_segments"] >= 1
    assert transform.inverse_reads >= 1


@pytest.mark.parametrize("xyz,yaw", [((10., 20., 2.), 90.), ((-30., 7., 9.), -135.)])
def test_capture_camera_yaw_translation_leave_camera_relative_pixels_unchanged(xyz, yaw):
    baseline, transformed = image(), image()
    local = [[10., 2., -1.], [10., 2., 1.]]
    draw_trajectory_overlay(baseline, display([path(local)]), CaptureTransform())
    capture = CaptureTransform(xyz, yaw)
    draw_trajectory_overlay(transformed, display([path(capture.to_world(local))]), capture)
    # Tiny floating-point transform error may affect AA on one border, but the
    # independently known line center must remain at (384,180), not world origin.
    assert near_pixel(color_mask(transformed), 384, 180)
    a, b = np.argwhere(changed(baseline)), np.argwhere(changed(transformed))
    assert np.max(np.abs(a.mean(axis=0)-b.mean(axis=0))) < 1.
    assert np.max(np.abs(a.min(axis=0)-b.min(axis=0))) <= 1
    assert np.max(np.abs(a.max(axis=0)-b.max(axis=0))) <= 1


@pytest.mark.parametrize("points", [
    [[-2., -1., 0.], [-2., 1., 0.]],
    [[.1, -.5, 0.], [.1, .5, 0.]],
    [[0., -.5, 0.], [0., .5, 0.]],
])
def test_behind_or_entirely_before_point_two_meter_near_plane_never_paints(points):
    pixels = image()
    before = pixels.copy()
    audit = draw_trajectory_overlay(pixels, display([path(points)]), CaptureTransform())
    np.testing.assert_array_equal(pixels, before)
    assert audit["drawn_paths"] == 0


def test_segment_crossing_near_plane_is_clipped_in_3d_before_projection():
    pixels = image()
    # At x=.2, y=-.52 -> u=-512. The visible segment enters at the LEFT
    # frame and ends at u400; projecting the behind-camera endpoint instead
    # would wrongly draw only between u400 and u640.
    audit = draw_trajectory_overlay(pixels,
        display([path([[-1., -1., 0.], [4., 1., 0.]], width_m=.02)]), CaptureTransform())
    painted = changed(pixels)
    assert painted[176:185, :5].any()
    assert near_pixel(color_mask(pixels), 320, 180)
    assert near_pixel(color_mask(pixels), 400, 180)
    assert not painted[:, 450:].any()
    assert audit["drawn_paths"] == 1


def test_two_offscreen_endpoints_crossing_display_are_clipped_and_visible():
    pixels = image()
    audit = draw_trajectory_overlay(pixels,
        display([path([[10., -20., 0.], [10., 20., 0.]])]), CaptureTransform())
    painted = changed(pixels)
    assert painted[176:185, :4].any() and painted[176:185, -4:].any()
    assert near_pixel(color_mask(pixels), 320, 180)
    assert audit["drawn_paths"] == 1


@pytest.mark.parametrize("points", [
    [[10., 30., -1.], [10., 31., 1.]],
    [[10., -30., -1.], [10., -31., 1.]],
    [[10., -1., 30.], [10., 1., 31.]],
])
def test_wholly_offscreen_path_does_not_clamp_onto_an_image_border(points):
    pixels = image()
    before = pixels.copy()
    audit = draw_trajectory_overlay(pixels, display([path(points)]), CaptureTransform())
    np.testing.assert_array_equal(pixels, before)
    assert audit["drawn_paths"] == 0


@pytest.mark.parametrize("off_display", [None, {}, {"style": "paper", "paths": []},
    display([path([[10., 0., -1.], [10., 0., 1.]])], enabled=False),
    display([path([[10., 0., -1.], [10., 0., 1.]])], style="classic")])
def test_off_display_is_an_exact_image_noop(off_display):
    pixels = image()
    before = pixels.copy()
    audit = draw_trajectory_overlay(pixels, off_display, CaptureTransform())
    np.testing.assert_array_equal(pixels, before)
    assert audit["drawn_paths"] == 0


def test_nonuniform_waypoint_polyline_is_pixel_continuous_despite_legacy_dash_flag():
    pixels = image()
    # Depth is constant, so 1m is exactly32px. The entire real polyline must
    # remain connected, including locations that used to be metric dash gaps.
    data = display([path([[10., -5., 0.], [10., -3.7, 0.],
                          [10., -1.1, 0.], [10., 5., 0.]], dashed=True, width_m=.1)])
    audit = draw_trajectory_overlay(pixels, data, CaptureTransform())
    assert color_mask(pixels)[180, 162:479].all()
    assert audit["drawn_segments"] == 3
    assert audit["line_style"] == "solid" and audit["endpoint_markers"] is False


@pytest.mark.parametrize("reverse_input", [False, True])
def test_muted_blue_selected_outer_and_terracotta_execution_inner_remain_visible(reverse_input):
    pixels = image()
    blue, terracotta = [54, 90, 115], [201, 107, 80]
    points = [[10., 0., -1.5], [10., 0., 1.5]]
    paths = [
        path(points, kind="selected", rgb=blue, width_m=.20),
        path(points, kind="execution", rgb=terracotta, width_m=.09),
    ]
    draw_trajectory_overlay(pixels, display(paths[::-1] if reverse_input else paths), CaptureTransform())
    blue_x = np.where(color_mask(pixels, blue)[180])[0]
    execution_x = np.where(color_mask(pixels, terracotta)[180])[0]
    assert len(blue_x) and len(execution_x)
    assert blue_x.min() < execution_x.min() <= 320 <= execution_x.max() < blue_x.max()
    # With zero outline, AA can only mix source colors with the existing
    # background. A black halo or glow would leave this per-channel range.
    minimum = np.minimum(blue[::-1], terracotta[::-1])
    assert np.all(pixels.min(axis=(0, 1)) >= minimum-1)
    assert pixels.max() <= BACKGROUND


def test_readable_bounded_stroke_width_does_not_grow_with_camera_depth():
    pixels = image()
    # Both are .4m wide; near u200 depth10 and far u440 depth20.
    draw_trajectory_overlay(pixels, display([
        path([[10., -3.75, -1.], [10., -3.75, 1.]], kind="selected", width_m=.4),
        path([[20., 7.5, -2.], [20., 7.5, 2.]], kind="selected", width_m=.4),
    ]), CaptureTransform())
    row = color_mask(pixels)[180]
    near_width, far_width = row[180:220].sum(), row[420:460].sum()
    assert 0 < far_width <= near_width <= 20


@pytest.mark.parametrize("bad", [
    None,
    {"kind": "candidate", "points_world": [[10., 0., 0.], [float("nan"), 1., 0.]],
     "width_m": .1, "rgb": RGB},
    {"kind": "candidate", "points_world": [[10., 0.], [11., 1.]], "width_m": .1, "rgb": RGB},
])
def test_invalid_single_path_cannot_hide_or_recolor_an_independent_valid_path(bad):
    valid = path([[10., 2., -1.], [10., 2., 1.]])
    expected, actual = image(), image()
    draw_trajectory_overlay(expected, display([valid]), CaptureTransform())
    audit = draw_trajectory_overlay(actual, display([bad, valid]), CaptureTransform())
    np.testing.assert_array_equal(actual, expected)
    assert audit["drawn_paths"] == 1


def test_only_image_is_mutated_not_display_paths_rgb_or_capture_matrix():
    pixels = image()
    data = display([path([[10., 2., -1.], [10., 2., 1.]], marker=True)], endpoint_markers=True)
    transform = CaptureTransform((2., 3., 4.), 25.)
    data["paths"][0]["points_world"] = transform.to_world(data["paths"][0]["points_world"])
    before = deepcopy(data)
    inverse = transform.inverse.copy()
    draw_trajectory_overlay(pixels, data, transform)
    assert changed(pixels).any()
    assert data == before
    np.testing.assert_array_equal(transform.inverse, inverse)


def test_missing_capture_transform_cannot_synthesize_a_world_or_live_camera_pose():
    pixels = image()
    before = pixels.copy()
    audit = draw_trajectory_overlay(pixels,
        display([path([[10., 2., -1.], [10., 2., 1.]])]), None)
    np.testing.assert_array_equal(pixels, before)
    assert audit["drawn_paths"] == 0
    assert audit["skipped_reason"] == "missing_capture_transform"


@pytest.mark.parametrize("kind", ["candidate", "selected", "execution", "rejected"])
def test_legacy_marker_dash_phase_flags_cannot_add_arrow_cap_or_gaps(kind):
    normal, legacy_flags = image(), image()
    # A real ground-plane forward line projects vertically below the horizon.
    # Old endpoint arrows/caps would add lateral pixels near its far endpoint.
    points = [[5., 0., -1.], [10., 0., -1.], [20., 0., -1.]]
    draw_trajectory_overlay(normal, display([path(points, kind=kind)]), CaptureTransform())
    audit = draw_trajectory_overlay(legacy_flags,
        display([path(points, kind=kind, dashed=True, marker=True, phase_m=1.3)],
                endpoint_markers=True, dash_length_m=.05, dash_gap_m=100.), CaptureTransform())
    np.testing.assert_array_equal(normal, legacy_flags)
    assert audit["drawn_segments"] == 2
    assert color_mask(legacy_flags)[198:243, 320].all()


def test_zero_outline_is_valid_and_never_introduces_a_black_halo_or_glow():
    pixels = image()
    rgb = [90, 140, 180]
    draw_trajectory_overlay(pixels,
        display([path([[10., -4., 0.], [10., 4., 0.]], rgb=rgb)], outline_width_m=0.), CaptureTransform())
    assert color_mask(pixels, rgb)[180, 194:447].all()
    assert np.all(pixels.min(axis=(0, 1)) >= np.asarray(rgb[::-1])-1)
    assert pixels.max() <= BACKGROUND


def test_stationary_polyline_stays_blank_even_with_legacy_markers_enabled():
    pixels = image()
    before = pixels.copy()
    audit = draw_trajectory_overlay(pixels,
        display([path([[10., 2., 0.]]*3, kind="execution", marker=True, dashed=True)],
                endpoint_markers=True), CaptureTransform())
    np.testing.assert_array_equal(pixels, before)
    assert audit["drawn_paths"] == 0 and audit["drawn_segments"] == 0
    assert any(p["kind"] == "execution" for p in audit["stationary_paths"])


def test_a_billion_meter_two_point_line_is_clipped_as_one_real_segment():
    pixels = image()
    audit = draw_trajectory_overlay(pixels,
        display([path([[10., -1e9, 0.], [10., 1e9, 0.]], dashed=True, marker=True)],
                endpoint_markers=True), CaptureTransform())
    assert audit["drawn_paths"] == 1 and audit["drawn_segments"] == 1
    assert color_mask(pixels)[180, 1:-1].all()


def test_excessive_actual_polyline_segments_cannot_hide_a_normal_path():
    excessive = path(np.column_stack((np.full(5002, 10.), np.linspace(-4., 4., 5002), np.zeros(5002))).tolist())
    valid = path([[10., 2., -1.], [10., 2., 1.]])
    expected, actual = image(), image()
    draw_trajectory_overlay(expected, display([valid]), CaptureTransform())
    audit = draw_trajectory_overlay(actual, display([excessive, valid]), CaptureTransform())
    np.testing.assert_array_equal(actual, expected)
    assert audit["drawn_paths"] == 1 and len(audit["skipped_paths"]) == 1


def box_corners(x_range, y_range, z_range):
    return [[x, y, z] for x in x_range for y in y_range for z in z_range]


def test_ego_bbox_restores_original_pixels_inside_hull_but_keeps_outside_lines():
    yy, xx = np.indices((HEIGHT, WIDTH))
    original = np.stack((xx % 173, yy % 181, (xx+yy) % 191), axis=-1).astype(np.uint8)
    plain, masked = original.copy(), original.copy()
    trajectory = path([[10., -4., 0.], [10., 4., 0.]])
    data = display([trajectory], ego_box_world=box_corners((8., 12.), (-1., 1.), (-.5, .5)))
    before = deepcopy(data)
    draw_trajectory_overlay(plain, display([trajectory]), CaptureTransform())
    audit = draw_trajectory_overlay(masked, data, CaptureTransform())
    # Analytic projected hull is [280,360]x[160,200]; use a strictly interior
    # rectangle so AA/rasterization at its boundary cannot decide this test.
    interior = np.s_[170:191, 290:351]
    np.testing.assert_array_equal(masked[interior], original[interior])
    assert np.any(plain[interior] != original[interior])
    assert near_pixel(color_mask(masked), 240, 180)
    assert near_pixel(color_mask(masked), 400, 180)
    assert audit["ego_box_masked"] is True and audit["drawn_paths"] == 1
    assert data == before


def test_invalid_ego_bbox_isolated_from_an_otherwise_valid_trajectory():
    expected, actual = image(), image()
    trajectory = path([[10., -4., 0.], [10., 4., 0.]])
    bad_box = box_corners((8., 12.), (-1., 1.), (-.5, .5))
    bad_box[3][1] = float("nan")
    draw_trajectory_overlay(expected, display([trajectory]), CaptureTransform())
    audit = draw_trajectory_overlay(actual, display([trajectory], ego_box_world=bad_box), CaptureTransform())
    np.testing.assert_array_equal(actual, expected)
    assert audit["drawn_paths"] == 1 and audit["ego_mask_skipped"] == "invalid_ego_box"


def test_ego_bbox_crossing_near_plane_does_not_throw_or_poison_path_projection():
    pixels = image()
    audit = draw_trajectory_overlay(pixels,
        display([path([[10., -4., 0.], [10., 4., 0.]])],
                ego_box_world=box_corners((.1, 2.), (-.01, .01), (-.01, .01))), CaptureTransform())
    assert audit["drawn_paths"] == 1
    assert near_pixel(color_mask(pixels), 240, 180)
    assert near_pixel(color_mask(pixels), 400, 180)
