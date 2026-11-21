"""Render captured official scoring states, without importing or running NAVSIM.

This module consumes evaluation-side evidence only. It never infers missing
traffic, reruns a planner, or substitutes recorded traffic for the official
policy's actual response. Geometry is in world coordinates; ego states use the
rear axle, while actor boxes use their geometric centers.
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely import wkt


VERSION = "navsim-official-replay-1.0"
COLORS = {
    "background": "#f1f5f9", "road": "#d4dde5", "lane": "#dce5ec",
    "border": "#a9b8c7", "ink": "#183248", "muted": "#506a7e",
    "ego": "#087f8c", "executed": "#087f8c", "planned": "#4056c6",
    "human": "#ba7b2e", "vehicle": "#6e98c7", "pedestrian": "#b661a4",
    "bicycle": "#d49237", "static": "#73818e", "red": "#d33b4b",
}
METRICS = (
    ("NC", "no_at_fault_collisions"), ("DAC", "drivable_area_compliance"),
    ("DDC", "driving_direction_compliance"), ("TLC", "traffic_light_compliance"),
    ("EP", "ego_progress"), ("TTC", "time_to_collision_within_bound"),
    ("LK", "lane_keeping"), ("HC", "history_comfort"),
)


def _array(value, label, minimum_columns):
    array = np.asarray(value, dtype=float)
    if (array.ndim != 2 or len(array) < 2 or array.shape[1] < minimum_columns
            or not np.isfinite(array).all()):
        raise ValueError("Invalid replay " + label)
    return array


def footprint(pose, length_m, width_m, rear_axle_to_center_m=0.0):
    """Four world XY corners; the optional offset is along vehicle heading."""
    pose = np.asarray(pose, dtype=float)
    length_m, width_m, offset = map(float, (length_m, width_m, rear_axle_to_center_m))
    if (pose.shape != (3,) or not np.isfinite(pose).all()
            or not all(math.isfinite(v) for v in (length_m, width_m, offset))
            or min(length_m, width_m) <= 0):
        raise ValueError("Invalid replay footprint")
    c, s = math.cos(pose[2]), math.sin(pose[2])
    center = pose[:2] + offset * np.array([c, s])
    corners = np.array([[length_m / 2, width_m / 2], [length_m / 2, -width_m / 2],
                        [-length_m / 2, -width_m / 2], [-length_m / 2, width_m / 2]])
    return corners @ np.array([[c, s], [-s, c]]) + center


def load_replay(path):
    """Reject incomplete/misaligned captures instead of inventing frames."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("version") != VERSION:
        raise ValueError("Unsupported official replay version")
    dt = float(value["dt_s"])
    if not math.isfinite(dt) or not math.isclose(dt, 0.1, abs_tol=1e-9):
        raise ValueError("Official replay requires actual 0.1s samples")
    states = _array(value["ego"]["states"], "ego states", 5)
    if len(states) != 41:
        raise ValueError("Official replay must contain all 41 samples from 0.0 to 4.0s")
    if len(_array(value["planned_poses_world"], "planned poses", 3)) != len(states):
        raise ValueError("Submitted plan is not aligned to scoring samples")
    if value.get("human_poses_world"):
        if len(_array(value["human_poses_world"], "recorded human poses", 3)) != len(states):
            raise ValueError("Recorded human is not aligned to scoring samples")
    actors = value["actors"]
    if not isinstance(actors, list) or len(actors) != len(states):
        raise ValueError("Official response traffic is not aligned to ego samples")
    footprint(states[0, :3], value["ego"]["length_m"], value["ego"]["width_m"],
              value["ego"]["rear_axle_to_center_m"])
    for frame in actors:
        if not isinstance(frame, list):
            raise ValueError("Invalid official traffic frame")
        ids = []
        for actor in frame:
            ids.append(str(actor["id"]))
            footprint(actor["center"], actor["length_m"], actor["width_m"])
            if not isinstance(actor["type"], str):
                raise ValueError("Invalid official actor type")
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate actor in official traffic frame")
    lights = value.get("red_lights", [])
    if lights and len(lights) != len(states):
        raise ValueError("Red-light geometry is not aligned to scoring samples")
    for row in value["map"]["polygons"]:
        geometry = wkt.loads(row["wkt"])
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Invalid captured map geometry")
    for rows in lights:
        for row in rows:
            geometry = wkt.loads(row["wkt"])
            if geometry.is_empty or not geometry.is_valid:
                raise ValueError("Invalid captured traffic-light geometry")
    if value["map"].get("centerline"):
        _array(value["map"]["centerline"], "map centerline", 2)
    for event in value.get("events", []):
        index = event["frame_index"]
        if type(index) is not int or not 0 <= index < len(states) or not isinstance(event["type"], str):
            raise ValueError("Invalid raw event frame")
    if not isinstance(value.get("metrics", {}).get("official_final"), dict):
        raise ValueError("Missing official final metric components")
    return value


def _font(size, bold=False):
    names = ("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


class _Canvas:
    def __init__(self, replay, size):
        self.replay = replay
        self.size = size
        self.scale_ui = size / 900.0
        self.states = np.asarray(replay["ego"]["states"], float)
        self.planned = np.asarray(replay["planned_poses_world"], float)
        self.human = np.asarray(replay.get("human_poses_world") or [], float)
        # Translation preserves compass orientation and avoids huge map numbers.
        self.origin = self.states[0, :2].copy()
        self.rect = (self.u(22), self.u(125), self.u(878), self.u(658))
        self.width = self.rect[2] - self.rect[0]
        self.height = self.rect[3] - self.rect[1]
        core = np.concatenate((self.states[:, :2], self.planned[:, :2]), axis=0)
        low, high = core.min(axis=0) - 16.0, core.max(axis=0) + 16.0
        nearby = []
        for rows in replay["actors"]:
            for row in rows:
                point = np.asarray(row["center"][:2], float)
                # A distant background object must not make the ego unreadable.
                if np.min(np.linalg.norm(core - point, axis=1)) <= 18.0:
                    nearby.append(point)
        if nearby:
            low = np.minimum(low, np.min(nearby, axis=0) - 4.0)
            high = np.maximum(high, np.max(nearby, axis=0) + 4.0)
        center = (low + high) / 2.0
        extent = np.maximum(high - low, [42.0, 35.0])
        meters_per_pixel = max(extent[0] / self.width, extent[1] / self.height)
        self.ppm = 1.0 / meters_per_pixel
        self.minimum = center - np.array([self.width, self.height]) * meters_per_pixel / 2.0
        self.map_layer = Image.new("RGB", (self.width, self.height), COLORS["background"])
        drawing = ImageDraw.Draw(self.map_layer)
        self.map_polygons = [(wkt.loads(row["wkt"]), row.get("kind", "")) for row in replay["map"]["polygons"]]
        for geometry, kind in self.map_polygons:
            color = COLORS["road"] if "road" in str(kind).lower() else COLORS["lane"]
            self.geometry(drawing, geometry, color, COLORS["border"])
        self.grid(drawing)
        centerline = replay["map"].get("centerline")
        if centerline:
            self.path(drawing, np.asarray(centerline), "#9aacba", dash=True, width=1)
        if len(self.human):
            self.path(drawing, self.human, COLORS["human"], dash=True, width=self.u(2))
        self.path(drawing, self.planned, COLORS["planned"], dash=True, width=self.u(3))
        self.fonts = {name: _font(self.u(points), bold) for name, points, bold in (
            ("title", 24, True), ("normal", 15, False), ("small", 12, False), ("metric", 16, True))}
        self._text_masks = {}

    def u(self, value):
        return max(1, int(round(value * self.scale_ui)))

    def pixel(self, points):
        points = np.asarray(points, float)[..., :2]
        local = (points - self.minimum) * self.ppm
        local[..., 1] = self.height - local[..., 1]
        # Canvas clipping is automatic; cap huge map coordinates for PIL.
        return np.clip(np.rint(local), -1_000_000, 1_000_000).astype(int)

    def geometry(self, drawing, shape, fill, outline=None):
        if shape.geom_type == "Polygon":
            drawing.polygon([tuple(v) for v in self.pixel(shape.exterior.coords)], fill=fill, outline=outline)
            for ring in shape.interiors:
                drawing.polygon([tuple(v) for v in self.pixel(ring.coords)], fill=COLORS["background"])
        elif shape.geom_type in ("LineString", "LinearRing"):
            drawing.line([tuple(v) for v in self.pixel(shape.coords)], fill=outline or fill, width=self.u(3))
        elif hasattr(shape, "geoms"):
            for part in shape.geoms:
                self.geometry(drawing, part, fill, outline)
        else:
            raise ValueError("Unsupported captured geometry: " + shape.geom_type)

    def path(self, drawing, points, color, dash=False, width=2):
        pixels = self.pixel(points)
        if not dash:
            if len(pixels) >= 2:
                drawing.line([tuple(v) for v in pixels], fill=color, width=max(1, width))
            return
        # World-distance dashes remain consistent across trajectory sample rates.
        total = 0.0
        for a, b in zip(pixels[:-1], pixels[1:]):
            distance = float(np.linalg.norm(b - a))
            if distance <= 0:
                continue
            start = 0.0
            while start < distance:
                length = min(distance - start, self.u(7) - total % self.u(7))
                if int(total // self.u(7)) % 2 == 0:
                    first = a + (b - a) * start / distance
                    second = a + (b - a) * (start + length) / distance
                    drawing.line([tuple(first), tuple(second)], fill=color, width=max(1, width))
                start += max(length, 1e-7)
                total += max(length, 1e-7)

    def grid(self, drawing):
        # Small subdued tick labels use a fixed local EN origin for the clip.
        for axis, span in ((0, self.width), (1, self.height)):
            lower = self.minimum[axis] - self.origin[axis]
            upper = lower + span / self.ppm
            for value in np.arange(math.ceil(lower / 10) * 10, upper, 10):
                world = self.origin[axis] + value
                if axis == 0:
                    coordinate = int(round((world - self.minimum[0]) * self.ppm))
                    drawing.line([(coordinate, 0), (coordinate, self.height)], fill="#cbd5df", width=1)
                    drawing.text((coordinate + 3, self.height - self.u(17)), "%gm" % value,
                                 font=_font(self.u(10)), fill=COLORS["muted"])
                else:
                    coordinate = int(round(self.height - (world - self.minimum[1]) * self.ppm))
                    drawing.line([(0, coordinate), (self.width, coordinate)], fill="#cbd5df", width=1)
                    drawing.text((3, coordinate + 1), "%gm" % value,
                                 font=_font(self.u(10)), fill=COLORS["muted"])

    def vehicle(self, drawing, pose, length, width, color, offset=0.0):
        points = self.pixel(footprint(pose, length, width, offset))
        drawing.polygon([tuple(point) for point in points], fill=color, outline="white")
        front = (points[0] + points[1]) / 2.0
        rear = (points[2] + points[3]) / 2.0
        mark = 0.25 * rear + 0.75 * front
        radius = self.u(2)
        drawing.ellipse((mark[0] - radius, mark[1] - radius, mark[0] + radius, mark[1] + radius), fill="white")

    def text(self, drawing, xy, text, font="normal", color=None):
        # A clip repeats its title/legend/final components on every frame. Cache
        # their antialiased masks within this canvas rather than rasterizing
        # the same FreeType text forty-one times. Bounding-box offsets preserve
        # the original ImageDraw.text placement, including baseline padding.
        key = (font, text)
        if key not in self._text_masks:
            bounds = self.fonts[font].getbbox(text)
            mask = Image.new("L", (max(1, bounds[2] - bounds[0]), max(1, bounds[3] - bounds[1])))
            ImageDraw.Draw(mask).text((-bounds[0], -bounds[1]), text, font=self.fonts[font], fill=255)
            self._text_masks[key] = (bounds, mask)
        bounds, mask = self._text_masks[key]
        drawing.bitmap((self.u(xy[0]) + bounds[0], self.u(xy[1]) + bounds[1]), mask,
                       fill=color or COLORS["ink"])

    def frame(self, index):
        layer = self.map_layer.copy()
        drawing = ImageDraw.Draw(layer)
        lights = self.replay.get("red_lights", [])
        if lights:
            for row in lights[index]:
                self.geometry(drawing, wkt.loads(row["wkt"]), "#f3a1a8", COLORS["red"])
        self.path(drawing, self.states[:index + 1], COLORS["executed"], width=self.u(4))
        for actor in self.replay["actors"][index]:
            kind = actor["type"].lower()
            if "pedestrian" in kind:
                color = COLORS["pedestrian"]
            elif "bicycle" in kind or "cyclist" in kind:
                color = COLORS["bicycle"]
            elif "vehicle" in kind:
                color = COLORS["vehicle"]
            else:
                color = COLORS["static"]
            self.vehicle(drawing, actor["center"], actor["length_m"], actor["width_m"], color)
        ego = self.replay["ego"]
        self.vehicle(drawing, self.states[index, :3], ego["length_m"], ego["width_m"],
                     COLORS["ego"], ego["rear_axle_to_center_m"])
        image = Image.new("RGB", (self.size, self.size), "white")
        image.paste(layer, self.rect[:2])
        drawing = ImageDraw.Draw(image)
        self.text(drawing, (22, 16), "NAVSIM | Official scoring replay", "title")
        stage = str(self.replay.get("stage", "unknown"))
        self.text(drawing, (22, 52), "Stage: " + stage + "    Token: " + str(self.replay["token"])[:50], "small")
        self.text(drawing, (22, 75), "t = %.1f / 4.0 s    speed = %.2f m/s    actors = %d" %
                  (index * self.replay["dt_s"], np.linalg.norm(self.states[index, 3:5]),
                   len(self.replay["actors"][index])))
        self.text(drawing, (22, 100), "Official LQR / dynamics + captured traffic-policy response; no replanning within this clip.", "small")
        drawing.rectangle(self.rect, outline=COLORS["border"], width=1)
        # Compass stays fixed as the vehicle turns; this is not an ego-up view.
        nx, ny = self.u(847), self.u(164)
        drawing.line([(nx, ny + self.u(19)), (nx, ny - self.u(10))], fill=COLORS["ink"], width=self.u(2))
        drawing.polygon([(nx, ny - self.u(15)), (nx - self.u(4), ny - self.u(6)),
                         (nx + self.u(4), ny - self.u(6))], fill=COLORS["ink"])
        self.text(drawing, (839, 131), "N", "small")
        self.text(drawing, (22, 667), "World EN axes, translated to anchor; vehicle bodies use correct center / rear-axle geometry.", "small")
        human_label = "Recorded human" if len(self.human) else "Human reference: N/A"
        legends = (("Ego / executed", "executed"), ("Submitted plan", "planned"), (human_label, "human"),
                   ("Vehicles", "vehicle"), ("Pedestrians", "pedestrian"), ("Bikes", "bicycle"),
                   ("Static", "static"), ("Red-light area", "red"))
        for number, (label, key) in enumerate(legends):
            x, y = 24 + (number % 4) * 215, 694 + (number // 4) * 24
            drawing.rectangle((self.u(x), self.u(y + 3), self.u(x + 13), self.u(y + 13)), fill=COLORS[key])
            self.text(drawing, (x + 20, y), label, "small")
        self.text(drawing, (22, 749), "OFFICIAL FINAL COMPONENTS (whole clip, after official filtering)", "normal")
        values = self.replay["metrics"]["official_final"]
        for number, (short, key) in enumerate(METRICS):
            value = values.get(key)
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            label = "--" if not valid else "%.2f" % value
            self.text(drawing, (24 + number * 107, 777), short + " " + label, "metric",
                      COLORS["red"] if valid and value < 0.5 else COLORS["ink"])
        self.text(drawing, (22, 809), "RAW TIMED DIAGNOSTICS (before filtering; not per-frame official scores)", "small")
        events = [str(event["type"]) + ((" [" + str(event["actor_id"]) + "]") if event.get("actor_id") else "")
                  for event in self.replay.get("events", []) if event["frame_index"] == index]
        message = "; ".join(events) if events else "No timed raw diagnostic at this sample."
        self.text(drawing, (22, 830), message[:105], "normal", COLORS["red"] if events else COLORS["muted"])
        self.text(drawing, (22, 866), "0.0-4.0 s state horizon; 41 endpoint-inclusive frames / 10 fps = 4.1 s video. EPDMS aggregates separately.", "small")
        return image


def render_replay(replay_path, output_dir=None, *, size=900):
    """Write and decode-check one MP4; failures are explicit and never simulated.

    ``replay_path`` is gzip JSON using VERSION. Output contains official_bev.mp4,
    official_bev_first.png and render_status.json. Returned status includes the
    exact actual frame count, simulation span and encoded duration.
    """
    replay_path = Path(replay_path).resolve()
    destination = Path(output_dir).resolve() if output_dir is not None else replay_path.parent
    if type(size) is not int or size < 600 or size > 2000 or size % 2:
        raise ValueError("BEV video size must be an even integer between 600 and 2000")
    replay = load_replay(replay_path)
    destination.mkdir(parents=True, exist_ok=True)
    canvas = _Canvas(replay, size)
    token = uuid.uuid4().hex[:12]
    # Worktrees plus sample hashes already make long Windows paths. Keep atomic
    # temporary basenames shorter than the final names, including status files.
    temporary = destination / (".v_" + token + ".mp4")
    video = destination / "official_bev.mp4"
    status_path = destination / "render_status.json"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (size, size))
    count = 0
    try:
        if not writer.isOpened():
            raise RuntimeError("OpenCV mp4v encoder did not open")
        for index in range(len(canvas.states)):
            frame = canvas.frame(index)
            if index == 0:
                frame.save(destination / "official_bev_first.png")
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            count += 1
    finally:
        writer.release()
    if count != len(canvas.states):
        raise RuntimeError("Incomplete official scoring BEV encoding")
    capture = cv2.VideoCapture(str(temporary))
    decoded = 0
    try:
        if not capture.isOpened():
            raise RuntimeError("Encoded official replay could not be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (size, size):
                raise RuntimeError("Encoded BEV frame size mismatch")
            decoded += 1
    finally:
        capture.release()
    if decoded != count or not math.isclose(fps, 10.0, abs_tol=1e-5):
        raise RuntimeError("Encoded official replay failed frame-count/fps validation")
    os.replace(temporary, video)
    status = dict(status="complete", token=replay["token"], replay_version=VERSION,
                  stage=replay.get("stage"), replay_path=str(replay_path),
                  video=str(video), preview=str(destination / "official_bev_first.png"),
                  frames=count, decoded_frames=decoded, playback_verified=True,
                  fps=fps, size=[size, size], codec="mp4v", simulated_span_s=4.0,
                  encoded_duration_s=count / fps, source="actual_official_scoring_execution",
                  note="41 endpoint-inclusive samples over 4.0s; video is 4.1s at 10fps; final metrics are not instantaneous")
    temporary_status = destination / (".s_" + token + ".tmp")
    temporary_status.write_text(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary_status, status_path)
    return status


def main():
    parser = argparse.ArgumentParser(description="Render captured official NAVSIM scoring as BEV MP4")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--size", type=int, default=900)
    args = parser.parse_args()
    print(json.dumps(render_replay(args.replay, args.output_dir, size=args.size), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
