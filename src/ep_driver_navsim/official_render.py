"""Offline SDK plot export of detached scoring evidence, never a scoring run.

NAVSIM's map, annotation, box, trajectory and camera draw functions are called
directly. Scene loading is only a fallback for old traces lacking an archived
input packet; recorded scene futures are never drawn as executed states.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from nine_grid.paths import filesystem_path

from .replay_render import RENDERER_VERSION, _render_captured, load_replay, replay_metrics
from .profiles import get_profile


VIEWS = (("cam_l0", "cam_f0", "cam_r0"),
         ("cam_l1", None, "cam_r1"), ("cam_l2", "cam_b0", "cam_r2"))


def read_packet(path):
    from .archive import read_input_packet
    return read_input_packet(path)


def local_poses(poses, origin):
    """World rear-axle/box poses to the fixed source anchor's official BEV axes."""
    result = np.asarray(poses, dtype=float)[:, :3].copy()
    c, s = math.cos(origin[2]), math.sin(origin[2])
    result[:, :2] = (result[:, :2] - origin[:2]) @ np.array([[c, -s], [s, c]])
    result[:, 2] = np.arctan2(np.sin(result[:, 2] - origin[2]), np.cos(result[:, 2] - origin[2]))
    return result


def _packet_path(replay_path, settings, explicit=None):
    if explicit:
        path = filesystem_path(explicit)
        if path.name == "input_packet.json" and path.with_suffix(".json.gz").is_file():
            path = path.with_suffix(".json.gz")
        return path
    samples = [filesystem_path(replay_path).parent.parent]
    # Old parity exports stored traces only; their frozen settings retain the
    # source planning-run settings path. Do not scan all runs or infer by order.
    frozen_path = settings.get("settings_path")
    if frozen_path and filesystem_path(frozen_path).is_file():
        source = json.loads(filesystem_path(frozen_path).read_text(encoding="utf-8"))
        samples.append(filesystem_path(source["output_dir"]) / "samples" / samples[0].name)
    for sample in samples:
        for name in ("input_packet.json.gz", "input_packet.json"):
            path = sample / name
            if path.is_file():
                return path
    return None


def _camera_path(row, sample):
    # Prefer colocated archive assets after copying a run to a Linux server.
    view = row["view"]
    for path in (filesystem_path(sample) / "cameras" / (view + ".jpg"), filesystem_path(row["path"])):
        if path.is_file():
            return path
    raise FileNotFoundError("Missing archived source camera " + view)


def load_source(replay, replay_path, settings, explicit_packet=None):
    from navsim.common.dataclasses import Camera, SensorConfig
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    packet_path = _packet_path(replay_path, settings, explicit_packet)
    if packet_path:
        packet = read_packet(packet_path)
        if packet["identity"]["token"] != replay["token"]:
            raise ValueError("Archived packet token differs from scoring trace")
        source_pose = packet.get("diagnostics", {}).get("source_rear_axle_global_pose")
        if source_pose is not None and not np.allclose(source_pose, replay["ego"]["states"][0][:3], atol=1e-5, rtol=0):
            raise ValueError("Archived packet source pose differs from scoring trace anchor")
        cameras = {}
        for row in packet["camera_images"]:
            if row.get("frame_id") != replay["token"] or float(row.get("time", 0)) != 0:
                raise ValueError("Camera is not associated with the source scoring anchor")
            path = _camera_path(row, packet_path.parent)
            with Image.open(path) as image:
                cameras[row["view"]] = Camera(image=np.asarray(image.convert("RGB")))
        api = get_maps_api(settings["maps_runtime_root"], "nuplan-maps-v1.0", packet["identity"]["map_name"])
        return api, cameras, dict(source="archived_input_packet", packet=str(packet_path),
            map_name=packet["identity"]["map_name"], timestamp_us=packet["identity"].get("timestamp_us"),
            camera_source="static_source_anchor_original_or_official_synthetic_images")
    from .sdk_runtime import make_loader
    loader = make_loader(settings)
    loader._sensor_config = SensorConfig(**dict({view: [3] for row in VIEWS for view in row if view}, lidar_pc=False))
    scene = loader.get_scene_from_token(replay["token"])
    index = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[index]
    if not np.allclose(frame.ego_status.ego_pose, replay["ego"]["states"][0][:3], atol=1e-5, rtol=0):
        raise ValueError("Source scene pose differs from scoring trace anchor")
    cameras = {view: getattr(frame.cameras, view) for row in VIEWS for view in row if view}
    return scene.map_api, cameras, dict(source="official_scene_loader_current_frame_only",
        map_name=scene.scene_metadata.map_name, timestamp_us=int(frame.timestamp),
        camera_source="static_source_anchor_original_or_official_synthetic_images",
        recorded_future_states_used=False)


class OfficialCanvas:
    def __init__(self, replay, size, map_api):
        import matplotlib.pyplot as plt
        from navsim.visualization import bev, plots
        from navsim.visualization.config import AGENT_CONFIG, BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, TAB_10
        from nuplan.common.actor_state.state_representation import StateSE2

        self.bev, self.plots, self.replay, self.size = bev, plots, replay, size
        profile = get_profile(replay)
        self.agent_config, self.trajectory_config = AGENT_CONFIG, TRAJECTORY_CONFIG
        self.states = np.asarray(replay["ego"]["states"], float)
        self.origin = self.states[0, :3].copy()
        self.local_states = local_poses(self.states, self.origin)
        self.calls = {}
        self.fig = plt.figure(figsize=(size / 100, size / 100), dpi=100, facecolor="white")
        self.ax = self.fig.add_axes([.055, .245, .89, .66])
        self.ax.set_facecolor(BEV_PLOT_CONFIG["background_color"])
        self.call("add_map_to_bev_ax", bev.add_map_to_bev_ax, self.ax, map_api, StateSE2(*self.origin))
        self.call("configure_bev_ax", plots.configure_bev_ax, self.ax)
        self.call("configure_ax", plots.configure_ax, self.ax)
        planned = local_poses(replay["planned_poses_world"], self.origin)
        # Keep official human/agent colors. Line style identifies the submitted
        # plan; executed history uses the SDK blue palette and grows with time.
        planned_config = dict(TRAJECTORY_CONFIG["agent"], line_style="--", marker="", line_width=1.5)
        self.trajectory(planned, planned_config)
        human = replay.get("human_poses_world")
        if human:
            self.trajectory(local_poses(human, self.origin), dict(TRAJECTORY_CONFIG["human"], marker="", line_width=1.5))
        self.executed_config = dict(TRAJECTORY_CONFIG["agent"], line_color=TAB_10[0], marker="", line_width=2.5)
        core = np.concatenate((planned, self.local_states))
        extent = max(32., float(np.abs(core[:, :2]).max()) + 7)
        self.ax.set_xlim(extent, -extent)
        self.ax.set_ylim(-extent, extent)
        self.title = self.fig.text(.055, .946, "", fontsize=13 * size / 900, weight="bold")
        self.fig.text(.055, .918, "NAVSIM SDK plots | fixed source-anchor axes: forward up, left to left", fontsize=9 * size / 900)
        legend = [("Submitted plan --", TRAJECTORY_CONFIG["agent"]["line_color"]),
                  ("Executed history", TAB_10[0]), ("Recorded human reference" if human else "Human: unavailable", TRAJECTORY_CONFIG["human"]["line_color"])]
        for index, (label, color) in enumerate(legend):
            self.fig.text(.06 + index * .30, .218, label, color=color, fontsize=10 * size / 900)
        traffic = "exact scored non-reactive GT occupancy" if profile["is_v1"] else "first traffic-policy response"
        self.fig.text(.055, .189, "Bodies: actual scorer ego / " + traffic + ". Cameras: source anchor only.", fontsize=9 * size / 900)
        metric_label = ("OFFICIAL V1 PDMS COMPONENTS (whole anchor; no human filter)" if profile["is_v1"]
                        else "OFFICIAL FINAL COMPONENTS (whole anchor, after official filtering)")
        self.fig.text(.055, .158, metric_label, fontsize=9 * size / 900)
        values = replay["metrics"]["official_final"]
        labels = [short + " " + ("%.2f" % values[key] if isinstance(values.get(key), (float, int)) else "--")
                  for short, key in replay_metrics(replay)]
        self.fig.text(.055, .130, "    ".join(labels), fontsize=10 * size / 900)
        self.fig.text(.055, .098, "RAW TIMED DIAGNOSTICS (before filtering; not per-frame scores)", fontsize=9 * size / 900)
        self.events = self.fig.text(.055, .071, "", fontsize=10 * size / 900)
        self.fig.text(.055, .032, "41 captured samples / 10 Hz / 0-4 s; 4.1 s video. Independent anchor; no replanning.", fontsize=9 * size / 900)
        self.dynamic = []

    def call(self, name, function, *args, **kwargs):
        self.calls[name] = self.calls.get(name, 0) + 1
        return function(*args, **kwargs)

    def trajectory(self, poses, config):
        # Official function needs only .poses, and prepends the true source
        # origin. Avoid fabricating a sampling rate for growing short histories.
        self.call("add_trajectory_to_bev_ax", self.bev.add_trajectory_to_bev_ax,
                  self.ax, SimpleNamespace(poses=poses[1:]), config)

    def frame(self, index):
        from navsim.common.dataclasses import Annotations
        from navsim.visualization.config import MAP_LAYER_CONFIG
        from nuplan.common.actor_state.oriented_box import OrientedBox
        from nuplan.common.actor_state.state_representation import StateSE2
        from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from shapely import affinity, wkt

        for artist in self.dynamic:
            artist.remove()
        before = set(self.ax.get_children())
        actors = self.replay["actors"][index]
        poses = local_poses([actor["center"] for actor in actors], self.origin) if actors else []
        boxes = [[*pose[:2], 0., actor["length_m"], actor["width_m"], 1., pose[2]] for pose, actor in zip(poses, actors)]
        # The official SDK accepts the exact enum names captured by the scorer.
        annotations = Annotations(boxes=np.asarray(boxes, float).reshape(-1, 7),
            names=[actor["type"].lower() for actor in actors], velocity_3d=np.zeros((len(actors), 3)),
            instance_tokens=[actor["id"] for actor in actors], track_tokens=[actor["id"] for actor in actors])
        self.call("add_annotations_to_bev_ax", self.bev.add_annotations_to_bev_ax, self.ax, annotations, add_ego=False)
        ego = self.replay["ego"]
        pose = self.local_states[index].copy()
        pose[:2] += ego["rear_axle_to_center_m"] * np.array([np.cos(pose[2]), np.sin(pose[2])])
        box = OrientedBox(StateSE2(*pose), ego["length_m"], ego["width_m"], 1.)
        self.call("add_oriented_box_to_bev_ax", self.bev.add_oriented_box_to_bev_ax,
                  self.ax, box, self.agent_config[TrackedObjectType.EGO])
        self.trajectory(self.local_states[:index + 1], self.executed_config)
        red_rows = self.replay.get("red_lights", [])
        for row in red_rows[index] if red_rows else []:
            polygon = affinity.translate(wkt.loads(row["wkt"]), -self.origin[0], -self.origin[1])
            polygon = affinity.rotate(polygon, -self.origin[2], origin=(0, 0), use_radians=True)
            config = dict(MAP_LAYER_CONFIG[SemanticMapLayer.STOP_LINE], fill_color_alpha=.15, line_color_alpha=.7, zorder=1.5)
            self.call("add_polygon_to_bev_ax", self.bev.add_polygon_to_bev_ax, self.ax, polygon, config)
        self.dynamic = [artist for artist in self.ax.get_children() if artist not in before]
        self.title.set_text("%s | %s | t=%.1f / 4.0s | %.2f m/s" %
            (self.replay["token"], self.replay.get("stage", ""), index * .1, np.linalg.norm(self.states[index, 3:5])))
        events = [event["type"] for event in self.replay.get("events", []) if event["frame_index"] == index]
        self.events.set_text(("; ".join(events) or "No timed raw diagnostic at this sample.")[:110])
        self.fig.canvas.draw()
        return Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy())

    def camera_panel(self, cameras, destination):
        import matplotlib.pyplot as plt
        from navsim.visualization.camera import add_camera_ax
        from navsim.visualization.config import CAMERAS_PLOT_CONFIG

        self.frame(0)
        # This center crop is an official-drawn trace frame, not recorded ego
        # annotations. Preserve the official tutorial's eight-camera arrangement.
        bounds = self.ax.get_window_extent().bounds
        image = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        x, y, width, height = [int(round(value)) for value in bounds]
        bev = image[self.size - y - height:self.size - y, x:x + width]
        fig, axes = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"], dpi=100)
        for row, views in enumerate(VIEWS):
            for col, view in enumerate(views):
                ax = axes[row, col]
                if view:
                    camera = cameras.get(view)
                    if camera is None or camera.image is None:
                        raise ValueError("Missing true source-anchor camera: " + view)
                    self.call("add_camera_ax", add_camera_ax, ax, camera)
                    ax.set_title(view + " | source t=0.0s", fontsize=8)
                else:
                    ax.imshow(bev)
                    ax.set_title("Official scoring execution | t=0.0s", fontsize=8)
                self.call("configure_ax", self.plots.configure_ax, ax)
        fig.suptitle("STATIC SOURCE CAMERAS | not simulated future views", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .96))
        path = filesystem_path(destination) / "official_cameras_anchor.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path


def render_official(replay_path, output_dir, settings, size=900, packet=None):
    import inspect
    import matplotlib.pyplot as plt
    from navsim.visualization import bev, camera, config, plots

    replay_path, destination = filesystem_path(replay_path), filesystem_path(output_dir)
    replay = load_replay(replay_path)
    if get_profile(replay)["id"] != get_profile(settings)["id"]:
        raise ValueError("Official renderer SDK profile differs from the captured score")
    source_hash = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    files = [filesystem_path(inspect.getfile(module)) for module in (bev, camera, config, plots)]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    api, cameras, source = load_source(replay, replay_path, settings, packet)
    canvas = OfficialCanvas(replay, size, api)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        panel = canvas.camera_panel(cameras, destination)
        status = _render_captured(replay_path, destination, replay, canvas, size,
            dict(renderer_version=RENDERER_VERSION, official_visualization=True), publish_status=False)
        if hashes != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}:
            raise RuntimeError("Official visualization source changed during export")
        if hashlib.sha256(replay_path.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError("Captured scoring trace changed during export")
        status.update(official_functions_called=canvas.calls, official_visualization_source_sha256=hashes,
            benchmark_profile=get_profile(replay)["id"], official_commit=settings.get("official_commit"),
            official_sdk_sources_unchanged=True, trace_sha256=source_hash,
            renderer_python=sys.version, source_anchor=source, camera_panel=str(panel),
            filesystem_path_policy="absolute_windows_extended_or_native_posix",
            renderer_overrides=["submitted plan dashed", "executed history SDK blue", "red scorer areas translucent", "axes enlarged only when needed to fit captured execution"],
            executed_ego_source="trace.ego.states", traffic_source="trace.actors", human_source="trace.human_poses_world_reference_only",
            planner_called=False, model_called=False, scoring_called=False, camera_future_generated=False)
        from .protocol import write_json
        write_json(destination / "render_status.json", status)
        return status
    finally:
        plt.close(canvas.fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--sdk-root")
    parser.add_argument("--maps-root")
    parser.add_argument("--data-root")
    parser.add_argument("--packet")
    parser.add_argument("--size", type=int, default=900)
    args = parser.parse_args()
    settings = json.loads(filesystem_path(args.settings).read_text(encoding="utf-8"))
    for key, value, environment in (("sdk_root", args.sdk_root, "EP_NAVSIM_SDK_ROOT"),
            ("maps_runtime_root", args.maps_root, "EP_NAVSIM_MAPS_ROOT"),
            ("data_root", args.data_root, "EP_NAVSIM_DATA_ROOT")):
        settings[key] = value or os.environ.get(environment) or settings.get(key)
    os.environ.update(NUPLAN_MAPS_ROOT=settings["maps_runtime_root"], NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
                      OPENSCENE_DATA_ROOT=settings["data_root"], MPLBACKEND="Agg")
    sys.path.insert(0, settings["sdk_root"])
    from .sdk_bootstrap import platform_setup
    platform_setup()
    import matplotlib
    matplotlib.use("Agg")
    print(json.dumps(render_official(args.replay, args.output_dir, settings, args.size, args.packet)), flush=True)


if __name__ == "__main__":
    main()
