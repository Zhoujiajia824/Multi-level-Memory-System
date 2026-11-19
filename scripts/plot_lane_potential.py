"""Reproduce lane-Frenet APF figures from the production numerical kernel.

python scripts/plot_lane_potential.py --output-dir outputs/lane_potential

The road fixtures are illustrations, not CARLA experimental observations. All
energies are actual kernel values with a linear signed color scale. Physical
footprints are drawn only as landmarks and do not determine APF distance.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nine_grid.lane_potential import LanePotentialGeometry


COLORS = {"ego": "#ef5350", "vehicle": "#3186d2", "bicycle": "#eb9027",
          "pedestrian": "#bd6ac6", "static_obstacle": "#d8ad00"}
DEFAULT_CLASSES = {
    "vehicle": {"longitudinal_m": 20., "lateral_m": 6., "peak": 8.},
    "bicycle": {"longitudinal_m": 10., "lateral_m": 3., "peak": 12.},
    "pedestrian": {"longitudinal_m": 10., "lateral_m": 10., "peak": 16.},
    "static_obstacle": {"longitudinal_m": 12., "lateral_m": 4., "peak": 8.}}
DEFAULT_GOAL = {"weight": 3., "longitudinal_scale_m": 30., "lateral_scale_m": 15., "shift": 6.}


@dataclass
class Fixture:
    key: str
    title: str
    reference: np.ndarray
    asphalt: object
    domain: object
    ego_s: float
    goal_s: float
    goal_d: float
    objects: list
    lanes: tuple = (-3.5, 0., 3.5)


def fixtures():
    ref = np.column_stack((np.linspace(-50., 90., 141), np.zeros(141)))
    road = LineString(ref).buffer(5.25, cap_style=2)
    cases = [Fixture("straight", "直道 · 三条同向车道", ref, road, road, 50., 85., 3.5,
                     [("V1", "vehicle", 65., 0.), ("B1", "bicycle", 77., 3.5),
                      ("P1", "pedestrian", 45., -3.5), ("O1", "static_obstacle", 82., -3.5)])]
    theta = np.linspace(-1.2, 1.5, 181)
    ref = np.column_stack((45. * np.sin(theta), 45. * (1. - np.cos(theta))))
    road = LineString(ref).buffer(5.25, cap_style=2)
    cases.append(Fixture("curve", "弯曲车道 · 椭圆势场沿车道弯曲", ref, road, road, 44., 79., 3.5,
                         [("V1", "vehicle", 59., 0.), ("B1", "bicycle", 71., 3.5),
                          ("P1", "pedestrian", 39., -3.5), ("O1", "static_obstacle", 76., -3.5)]))
    x = np.linspace(-50., 90., 201)
    ref = np.column_stack((x, 4. * np.sin(x / 25.) + .7 * np.sin(x / 10.)))
    chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))]
    t = np.gradient(ref, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    n = np.column_stack((-t[:, 1], t[:, 0]))
    blend = np.clip((chord - 55.) / 35., 0., 1.)
    blend = blend ** 2 * (3. - 2. * blend)
    left = 5.15 + .25 * np.sin(chord / 18.)
    right = 5.25 - 3.4 * blend
    road = Polygon(np.vstack((ref + n * left[:, None], (ref - n * right[:, None])[::-1])))
    cases.append(Fixture("irregular", "不规则道路 · 缓 S 弯与渐窄", ref, road, road, 50., 85., 3.5,
                         [("V1", "vehicle", 65., 0.), ("B1", "bicycle", 77., 3.5),
                          ("P1", "pedestrian", 45., 3.5), ("O1", "static_obstacle", 79., -1.)]))
    approach = np.column_stack((np.full(61, 3.5), np.linspace(-70., -11., 61)))
    angle = np.linspace(0., np.pi / 2., 61)
    turn = np.column_stack((-8.5 + 12. * np.cos(angle), -11. + 12. * np.sin(angle)))
    exit_ref = np.column_stack((np.linspace(-8.5, -75., 81), np.ones(81)))
    ref = np.vstack((approach, turn[1:], exit_ref[1:]))
    asphalt = unary_union((box(-7., -70., 7., 65.), box(-75., -7., 70., 7.)))
    corridor = LineString(ref).buffer(5.25, cap_style=2).intersection(asphalt)
    cases.append(Fixture("intersection", "十字路口 · 沿已选导航左转走廊", ref, asphalt, corridor, 49., 84., 0.,
                         [("V1", "vehicle", 60., 0.), ("B1", "bicycle", 43., 1.),
                          ("P1", "pedestrian", 54., .8), ("O1", "static_obstacle", 66., -1.)]))
    return cases


def configure_fonts():
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"),
                      Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")):
        if candidate.exists():
            fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = FontProperties(fname=str(candidate)).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "axes.facecolor": "#eef0f5", "savefig.facecolor": "white",
                         "hatch.linewidth": .35, "pdf.fonttype": 42})


def settings(config):
    data = json.loads(config.read_text(encoding="utf-8"))
    risk = data.get("risk", {})
    classes = {k: {**v, **{name: risk.get("classes", {}).get(k, {}).get(name, value)
                           for name, value in v.items()}}
               for k, v in DEFAULT_CLASSES.items()}
    goal_section = data.get("local_goal", {})
    attraction = {
        "weight": goal_section.get("attraction_weight", DEFAULT_GOAL["weight"]),
        "longitudinal_scale_m": goal_section.get("longitudinal_scale_m", DEFAULT_GOAL["longitudinal_scale_m"]),
        "lateral_scale_m": goal_section.get("lateral_scale_m", DEFAULT_GOAL["lateral_scale_m"]),
        "shift": goal_section.get("potential_shift", DEFAULT_GOAL["shift"])}
    return classes, attraction


def one_point(value):
    return np.asarray(value, float).reshape(-1, 2)[0]


def polygon_parts(shape):
    if shape.is_empty:
        return []
    return [shape] if shape.geom_type == "Polygon" else [p for p in shape.geoms if p.geom_type == "Polygon"]


def draw_shape(ax, shape, **kwargs):
    for polygon in polygon_parts(shape):
        ax.add_patch(PolygonPatch(np.asarray(polygon.exterior.coords), **kwargs))


def landmark(ax, geometry, s, d, name, kind):
    p = one_point(geometry.sample(s, d))
    frame = geometry.frame([s])
    tangent = np.asarray(frame["tangent"]).reshape(-1, 2)[0]
    normal = np.array([-tangent[1], tangent[0]])
    if kind == "pedestrian":
        ax.scatter(*p, s=30, color=COLORS[kind], edgecolor="white", linewidth=1., zorder=9)
    else:
        length, width = {"ego": (4.5, 1.8), "vehicle": (4.5, 1.8),
                         "bicycle": (1.8, .6), "static_obstacle": (.8, .8)}[kind]
        corners = np.array([p + a * length / 2 * tangent + b * width / 2 * normal
                            for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
        ax.add_patch(PolygonPatch(corners, facecolor=COLORS[kind], edgecolor="white", linewidth=1., zorder=8))
    if kind == "ego":
        ax.text(*(p - normal * 2.1), name, ha="center", va="center", color=COLORS[kind], weight="bold", zorder=12)
    else:
        ax.annotate(name, p, xytext=(5, 8), textcoords="offset points", color="white", weight="bold", fontsize=9,
                    bbox={"facecolor": "#172d39", "alpha": .8, "pad": .7, "edgecolor": "none"}, zorder=12)


def compute(fixture, classes, attraction, spacing):
    geometry = LanePotentialGeometry(fixture.reference, fixture.domain)
    local_ref = geometry.sample(np.linspace(max(0., fixture.ego_s - 21.), fixture.goal_s + 18., 200), 0.)
    minx, miny = np.min(local_ref, axis=0) - 7.
    maxx, maxy = np.max(local_ref, axis=0) + 7.
    x = np.arange(minx, maxx + spacing / 2., spacing)
    y = np.arange(miny, maxy + spacing / 2., spacing)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    goal = one_point(geometry.sample(fixture.goal_s, fixture.goal_d))
    att = geometry.attraction(xy, goal, attraction)
    repulsive = np.zeros(len(xy))
    objects = []
    for name, kind, s, d in fixture.objects:
        center = one_point(geometry.sample(s, d))
        field = geometry.repulsion(xy, center, classes[kind])
        repulsive += field["energy"]
        objects.append({"id": name, "kind": kind, "s": s, "d": d, "position": center.tolist(),
                        "settings": classes[kind]})
    valid = np.asarray(att["valid"], bool)
    repulsive[~valid] = np.nan
    total = att["energy"] + repulsive
    return {"fixture": fixture, "geometry": geometry, "x": x, "y": y, "xx": xx, "yy": yy,
            "attraction": np.asarray(att["energy"]).reshape(xx.shape),
            "repulsion": repulsive.reshape(xx.shape), "total": total.reshape(xx.shape),
            "valid": valid.reshape(xx.shape), "goal": goal, "objects": objects,
            "roi": [float(minx), float(maxx), float(miny), float(maxy)]}


def draw_road(ax, data, field, norm, contours=True):
    fixture, geometry = data["fixture"], data["geometry"]
    draw_shape(ax, fixture.asphalt, facecolor="#e4e7ed", edgecolor="#26394e", linewidth=1., zorder=0)
    # Unselected crossroad arms remain visible, but receive no artificial potential.
    excluded = fixture.asphalt.difference(fixture.domain)
    draw_shape(ax, excluded, facecolor="#e4e7ed", edgecolor="#aeb8c5", linewidth=.3, hatch="///", zorder=1)
    im = ax.pcolormesh(data["xx"], data["yy"], field, cmap="viridis", norm=norm,
                       shading="auto", rasterized=True, zorder=2)
    if contours:
        ax.contour(data["xx"], data["yy"], np.ma.masked_invalid(field), levels=np.linspace(norm.vmin, norm.vmax, 19),
                   colors="white", linewidths=.35, alpha=.5, zorder=3)
    ss = np.linspace(max(0., fixture.ego_s - 24.), fixture.goal_s + 21., 350)
    for d in fixture.lanes:
        lane = geometry.sample(ss, d)
        lane[~geometry.project(lane)["valid"]] = np.nan
        ax.plot(lane[:, 0], lane[:, 1], color="white", linewidth=.8, dashes=(4, 4), alpha=.6, zorder=4)
    navigation = geometry.sample(ss, 0.)
    ax.plot(navigation[:, 0], navigation[:, 1], color="#93e3ed", linewidth=1.2, dashes=(7, 4), zorder=5)
    # Draw a genuine constant-energy support contour of a single vehicle kernel.
    # Its world shape follows the road, and does not use the vehicle rectangle.
    vehicle = next(v for v in data["objects"] if v["kind"] == "vehicle")
    angle = np.linspace(0., 2. * np.pi, 160)
    ellipse = geometry.sample(vehicle["s"] + vehicle["settings"]["longitudinal_m"] * np.cos(angle),
                              vehicle["d"] + vehicle["settings"]["lateral_m"] * np.sin(angle))
    # Only the admitted corridor has potential; omit offroad portions of the support.
    support_valid = geometry.project(ellipse)["valid"]
    ellipse[~support_valid] = np.nan
    ax.plot(ellipse[:, 0], ellipse[:, 1], color="#f7fafc", linewidth=.85, linestyle=":", alpha=.95, zorder=5)
    draw_shape(ax, fixture.domain, facecolor="none", edgecolor="#26394e", linewidth=.85, zorder=6)
    landmark(ax, geometry, fixture.ego_s, 0., "E", "ego")
    for name, kind, s, d in fixture.objects:
        landmark(ax, geometry, s, d, name, kind)
    ax.scatter(*data["goal"], marker="*", s=155, color="#ffdb38", edgecolor="#172c47", linewidth=1.1, zorder=11)
    ax.annotate("G", data["goal"], xytext=(6, 6), textcoords="offset points", weight="bold", color="#172c47",
                bbox={"facecolor": "white", "alpha": .8, "pad": .5, "edgecolor": "none"}, zorder=12)
    ax.set(xlim=data["roi"][:2], ylim=data["roi"][2:], xlabel="x / m", ylabel="y / m")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#a8b8c9")
    return im


def save_figure(fig, directory, stem):
    fig.savefig(directory / (stem + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(directory / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def four_roads(results, output):
    vmin = min(float(np.nanmin(r["total"])) for r in results)
    vmax = max(float(np.nanmax(r["total"])) for r in results)
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.4), layout="constrained")
    for ax, data in zip(axes.flat, results):
        im = draw_road(ax, data, data["total"], norm)
        ax.set_title(data["fixture"].title, pad=9)
    bar = fig.colorbar(im, ax=axes, shrink=.76, pad=.025)
    bar.set_label("真实有符号势能 U = Uatt + Σ Urep（线性刻度）", labelpad=10)
    if vmin < 0 < vmax:
        ticks = np.unique(np.r_[vmin, 0., np.linspace(max(0., vmin), vmax, 5)[1:]])
        bar.set_ticks(ticks)
        bar.set_ticklabels([f"{v:.1f}" for v in ticks])
    fig.suptitle("车道 Frenet 叠加势场：目标形成谷地，对象形成平滑山峰", fontsize=20, weight="bold")
    legend = [Line2D([], [], marker="s", linestyle="none", color=COLORS[k], label=label)
              for k, label in (("ego", "E 自车"), ("vehicle", "V 车辆"), ("bicycle", "B 自行车"),
                               ("pedestrian", "P 行人"), ("static_obstacle", "O 道路障碍物"))]
    legend.extend([Line2D([], [], color="#74b9c8", linestyle="--", label="导航参考，非规划轨迹"),
                   Line2D([], [], color="#606f7f", linestyle=":", label="V1 的椭圆作用边界（截于车道）")])
    fig.legend(handles=legend, loc="outside lower center", ncols=4, frameon=False, fontsize=10)
    save_figure(fig, output, "four-road-frenet-potential")
    return {"linear_color_min": vmin, "linear_color_max": vmax}


def components_and_decay(results, output, classes, attraction):
    data = results[0]
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 8.6), layout="constrained")
    norm = Normalize(vmin=float(np.nanmin(data["attraction"])),
                     vmax=max(float(np.nanmax(data[key])) for key in ("attraction", "repulsion", "total")))
    for ax, key, title in zip(axes, ("attraction", "repulsion", "total"),
                               ("吸引势场 Uatt：G 处为负值谷底", "排斥势场 Σ Urep：中心椭圆平滑衰减", "真实叠加 Uatt + Σ Urep")):
        im = draw_road(ax, data, data[key], norm)
        ax.set_title(title, pad=5)
    fig.colorbar(im, ax=axes, shrink=.85, label="势能（线性、同一色标）", pad=.025)
    save_figure(fig, output, "straight-components-linear")

    geometry = data["geometry"]
    source = geometry.sample(65., 0.)
    distance = np.linspace(0., 23., 461)
    points = geometry.sample(65. + distance, 0.)
    base = geometry.repulsion(points, source, classes["vehicle"])
    attended = geometry.repulsion(points, source, classes["vehicle"], attention_gain=2.)
    fig, ax = plt.subplots(figsize=(9., 4.6), layout="constrained")
    ax.plot(distance, base["energy"], color="#3b528b", linewidth=2.5, label="普通车辆：峰值 8")
    ax.plot(distance, attended["energy"], color="#21918c", linewidth=2.5, label="关注车辆：峰值 × 2，作用距离不变")
    ax.axvline(classes["vehicle"]["longitudinal_m"], color="#555", linestyle="--", linewidth=1.)
    ax.set(xlabel="沿车道距对象中心的纵向距离 |Δs| / m", ylabel="排斥势能 Urep（线性）",
           xlim=(0., 23.), ylim=(-.4, 17.), title="车辆中心排斥势能：20 m 边界精确归零")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save_figure(fig, output, "vehicle-longitudinal-decay")


def comparison(output, attraction):
    theta = np.linspace(0., np.pi / 2., 181)
    reference = np.column_stack((25. * np.sin(theta), 25. * (1. - np.cos(theta))))
    domain = LineString(reference).buffer(5.25, cap_style=2)
    geometry = LanePotentialGeometry(reference, domain)
    xx, yy = np.meshgrid(np.linspace(-6., 31., 241), np.linspace(-6., 31., 241))
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    goal = one_point(geometry.sample(37., 0.))
    frenet = geometry.attraction(xy, goal, attraction)
    # Cartesian comparison uses the same quadratic scales, held at entry axes.
    offset = xy - goal
    scales = np.array([attraction["longitudinal_scale_m"], attraction["lateral_scale_m"]])
    cartesian = .5 * attraction["weight"] * np.sum((offset / scales) ** 2, axis=1) - attraction["shift"]
    cartesian[~frenet["valid"]] = np.nan
    norm = Normalize(min(float(np.nanmin(cartesian)), float(np.nanmin(frenet["energy"]))),
                     max(float(np.nanmax(cartesian)), float(np.nanmax(frenet["energy"]))))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.), layout="constrained")
    sample_s = np.linspace(4., 30., 8)
    center = geometry.sample(sample_s, 0.)
    frame = geometry.frame(sample_s)
    f_force = -geometry.attraction(center, goal, attraction)["gradient"]
    c_force = -attraction["weight"] * (center - goal) / scales ** 2
    for ax, field, force, title in zip(axes, (cartesian, frenet["energy"]), (c_force, f_force),
                                      ("笛卡尔二次吸引：固定入口坐标轴", "Frenet 二次吸引：方向沿导航前进")):
        im = ax.pcolormesh(xx, yy, field.reshape(xx.shape), cmap="viridis", norm=norm,
                           shading="auto", rasterized=True)
        draw_shape(ax, domain, facecolor="none", edgecolor="#22354a", linewidth=1.)
        ax.plot(reference[:, 0], reference[:, 1], "--", color="#bbf0f1", linewidth=1.)
        unit = force / np.linalg.norm(force, axis=1)[:, None]
        ax.quiver(center[:, 0], center[:, 1], unit[:, 0], unit[:, 1], color="white",
                  angles="xy", scale_units="xy", scale=.28, width=.007, zorder=6)
        ax.scatter(*goal, s=180, marker="*", color="#ffdb38", edgecolor="#172c47", zorder=7)
        ax.annotate("G", goal, xytext=(7, 4), textcoords="offset points", weight="bold", zorder=8)
        ax.set(aspect="equal", xlim=(-6., 31.), ylim=(-6., 31.), xlabel="x / m", ylabel="y / m", title=title)
    fig.colorbar(im, ax=axes, shrink=.84, label="吸引势能 Uatt（线性、同一色标）")
    fig.suptitle("同一左弯、同一局部引导点；白箭头为 $-\\nabla U_{att}$ 的方向，非规划轨迹", fontsize=15, weight="bold")
    save_figure(fig, output, "cartesian-vs-frenet-left-turn")
    f_unit = f_force / np.linalg.norm(f_force, axis=1)[:, None]
    c_unit = c_force / np.linalg.norm(c_force, axis=1)[:, None]
    return {"sample_s_m": sample_s.tolist(),
            "frenet_force_normal_component_unit": np.sum(f_unit * frame["normal"], axis=1).tolist(),
            "cartesian_force_normal_component_unit": np.sum(c_unit * frame["normal"], axis=1).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lane_potential")
    parser.add_argument("--config", type=Path, default=ROOT / "config/parameters.json")
    parser.add_argument("--spacing", type=float, default=.2, help="World raster spacing in metres")
    args = parser.parse_args()
    if args.spacing <= 0:
        parser.error("--spacing must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_fonts()
    classes, attraction = settings(args.config)
    results = []
    for fixture in fixtures():
        print(f"Evaluating {fixture.key} with production Frenet kernel...", flush=True)
        result = compute(fixture, classes, attraction, args.spacing)
        results.append(result)
        np.savez_compressed(args.output_dir / (fixture.key + "-numeric-field.npz"),
                            x=result["x"], y=result["y"], attraction=result["attraction"],
                            repulsion=result["repulsion"], total=result["total"], valid=result["valid"])
    evidence = {"kernel": "nine_grid.lane_potential.LanePotentialGeometry", "settings": {"classes": classes, "attraction": attraction},
                "field": "raw Uatt + sum(Urep), unit attention gain; no MPC risk weighting, no log transform",
                "distance": "smooth navigation Frenet arc length s and signed lateral offset d; center points",
                "outside_corridor": "invalid/NaN, never zero-energy driveable space", "spacing_m": args.spacing,
                "limitations": "Procedural road fixtures and frozen spatial fields; no CARLA run or closed-loop safety claim.",
                "cases": [{"id": r["fixture"].key, "objects": r["objects"], "roi": r["roi"],
                            "finite_samples": int(np.isfinite(r["total"]).sum()),
                            "offroad_finite_samples": int(np.isfinite(r["total"][~r["valid"]]).sum()),
                            "attraction_min": float(np.nanmin(r["attraction"])),
                            "repulsion_max": float(np.nanmax(r["repulsion"])),
                            "total_min": float(np.nanmin(r["total"])), "total_max": float(np.nanmax(r["total"]))}
                           for r in results]}
    probe_s = 65. + np.array([0., 5., 10., 15., 20., 21.])
    probe = results[0]["geometry"].repulsion(results[0]["geometry"].sample(probe_s, 0.),
                                            results[0]["geometry"].sample(65., 0.), classes["vehicle"])
    evidence["vehicle_longitudinal_probe"] = {"distance_from_source_m": (probe_s - 65.).tolist(),
                                             "energy": probe["energy"].tolist(),
                                             "force_magnitude": np.linalg.norm(probe["gradient"], axis=1).tolist()}
    evidence.update(four_roads(results, args.output_dir))
    components_and_decay(results, args.output_dir, classes, attraction)
    evidence["left_turn_direction_check"] = comparison(args.output_dir, attraction)
    (args.output_dir / "numeric-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved figures, PDF exports and numeric fields to {args.output_dir}")


if __name__ == "__main__":
    main()
