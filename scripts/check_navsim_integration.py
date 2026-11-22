"""Offline regressions for the privileged NAVSIM adapter; no simulator/API calls."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TEMP_ROOT = ROOT / "outputs/navsim_check_temp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

from ep_driver_navsim import input_adapter as adapter
from ep_driver_navsim.protocol import trajectory_from_centers, validate_packet, transform_points
from ep_driver_navsim.launch import verify_scores, normalize_cache_metadata
from ep_driver_navsim.worker import planning_config, snapshot_from_packet, check_memory_disjoint
from ep_driver_navsim.memory import select_memory
from nine_grid.potential_domain import polygonal_area


class HistoryOnlyFrames:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        if type(index) is not int or index >= 4 or index < 0:
            raise AssertionError("Attempted to read future or unbounded frames")
        return self.rows[index]


def fake_scene():
    frames = []
    for i in range(4):
        pose = np.array([-7.5 + 2.5 * i, 0.0, 0.0])
        anns = NS(names=["vehicle", "generic_object"],
                  boxes=np.array([[15.0, 0.0, 0.0, 4.5, 1.8, 1.5, 0.0], [5, 15, 0, 1, 1, 1, 0]]),
                  velocity_3d=np.array([[3.0 + 0.5 * i, 0, 0], [0.0, 0, 0]]),
                  track_tokens=["car", "furniture"])
        status = NS(ego_pose=pose, ego_velocity=np.array([5.0, 0.0]), ego_acceleration=np.zeros(2))
        frames.append(NS(token="anchor" if i == 3 else "past" + str(i), timestamp=1_000_000 + i * 500_000,
                         ego_status=status, annotations=anns, roadblock_ids=["R0", "R1", "R2"],
                         traffic_lights=[("C", True), ("parallel", True)]))
    metadata = NS(num_history_frames=4, initial_token="anchor", scene_token="scene", log_name="log",
                  map_name="map", corresponding_original_scene=None, corresponding_original_initial_token=None)
    scene = NS(scene_metadata=metadata, frames=HistoryOnlyFrames(frames), map_api=None)
    cams = NS(**{name: NS(image=np.zeros((8, 12, 3), dtype=np.uint8)) for name in adapter.CAMERAS})
    agent_input = NS(ego_statuses=[frame.ego_status for frame in frames], cameras=[cams] * 4)
    return scene, agent_input


def fake_map_packet(*args):
    def lane(key, start, end, nxt):
        return dict(id=key, center=[[start, 0.0], [end, 0.0]],
                    polygon=[[start, -1.75], [end, -1.75], [end, 1.75], [start, 1.75]], width=3.5,
                    speed_limit=10.0, successors=nxt, left=None, right=None, priority=0, boundaries=[])
    lanes = [lane("L", -30, 20, ["C"]), lane("C", 20, 35, ["X"]), lane("X", 35, 120, [])]
    connector = NS(stop_lines=[], baseline_path=NS(discrete_path=[NS(x=20.0, y=0.0)]))
    return lanes, np.array([[-30., 0.], [20., 0.], [35., 0.], [120., 0.]]), ["L", "C", "X"], "L", \
        Polygon([[-30, -3], [120, -3], [120, 3], [-30, 3]]), [], {"C": connector}, [], {"current_lane_id": "L"}


def packet_fixture():
    modules = {}
    for name in ("nuplan", "nuplan.common", "nuplan.common.actor_state", "nuplan.common.actor_state.vehicle_parameters"):
        modules[name] = ModuleType(name)
    modules["nuplan.common.actor_state.vehicle_parameters"].get_pacifica_parameters = lambda: NS(
        length=5.176, width=2.297, wheel_base=3.089, rear_axle_to_center=1.461)
    scene, agent_input = fake_scene()
    with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory, patch.dict(sys.modules, modules), \
            patch.object(adapter, "_map_packet", side_effect=fake_map_packet):
        return adapter.build_packet(agent_input, scene, directory)


class NavsimContractTests(unittest.TestCase):
    def test_polygonal_domain_preserves_areas_holes_and_excludes_detached_lines(self):
        road = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                       holes=[[(3, 3), (3, 5), (5, 5), (5, 3)]])
        second = box(20, 0, 25, 5)
        line = LineString([(12, 0), (15, 0)])
        mixed = GeometryCollection([road, GeometryCollection([second, line, Point(18, 0)])])
        area = polygonal_area(mixed)
        self.assertTrue(area.is_valid)
        self.assertEqual(area.geom_type, "MultiPolygon")
        self.assertEqual(area.symmetric_difference(unary_union([road, second])).area, 0)
        self.assertEqual(area.area, mixed.area)
        self.assertFalse(area.covers(Point(4, 4)))
        self.assertFalse(area.covers(Point(13, 0)))
        with self.assertRaisesRegex(ValueError, "有效面"):
            polygonal_area(GeometryCollection([line, Point(18, 0)]))

    def test_synthetic_route_prefix_is_reconnected_by_topology(self):
        def lane(key, block, start, end):
            baseline = LineString([(start, 0), (end, 0)])
            return NS(id=key, get_roadblock_id=lambda: block, polygon=box(start, -1.75, end, 1.75),
                baseline_path=NS(linestring=baseline, discrete_path=[NS(x=start, y=0), NS(x=end, y=0)]),
                left_boundary=NS(linestring=LineString([(start, 1.75), (end, 1.75)])),
                right_boundary=NS(linestring=LineString([(start, -1.75), (end, -1.75)])),
                speed_limit_mps=10., adjacent_edges=(None, None), outgoing_edges=[])
        predecessor, following = lane("L0", "R0", -20, 12), lane("L1", "R1", 12, 80)
        predecessor.outgoing_edges = [following]
        layers = NS(LANE="lane", LANE_CONNECTOR="connector", CROSSWALK="crosswalk",
                    ROADBLOCK="roadblock", INTERSECTION="intersection", CARPARK_AREA="carpark")
        nearby = {value: [] for value in vars(layers).values()}
        nearby[layers.LANE] = [predecessor, following]
        map_api = NS(get_proximal_map_objects=lambda *args: nearby)
        modules = {}
        for name in ("nuplan", "nuplan.common", "nuplan.common.actor_state",
                     "nuplan.common.actor_state.state_representation", "nuplan.common.maps",
                     "nuplan.common.maps.maps_datatypes"):
            modules[name] = ModuleType(name)
        modules["nuplan.common.actor_state.state_representation"].Point2D = lambda *args: args
        modules["nuplan.common.maps.maps_datatypes"].SemanticMapLayer = layers
        origin = np.zeros(3)
        with patch.dict(sys.modules, modules), patch.object(adapter, "_correct_route_order", return_value=["R0", "R1"]) as fix:
            result = adapter._map_packet(map_api, origin, ["R1"], 10., 160.)
            self.assertEqual(result[2], ["L0", "L1"])
            self.assertEqual(result[3], "L0")
            self.assertTrue(result[-1]["topology_correction_changed"])
            self.assertTrue(result[-1]["polygon_covers_rear_axle"])
            fix.assert_called_once_with(map_api, origin, ["R1"])
        # No route connection: the close off-route lane is not a fallback.
        with patch.dict(sys.modules, modules), patch.object(adapter, "_correct_route_order", return_value=["R1"]):
            with self.assertRaisesRegex(ValueError, "after topology correction"):
                adapter._map_packet(map_api, origin, ["R1"], 10., 160.)
        with patch.dict(sys.modules, modules), patch.object(adapter, "_correct_route_order", return_value=["R0", "R1"]) as fix:
            result = adapter._map_packet(map_api, origin, ["R0", "R1"], 10., 160.)
            self.assertTrue(result[-1]["topology_correction_attempted"])
            self.assertFalse(result[-1]["topology_correction_changed"])
            fix.assert_called_once_with(map_api, origin, ["R0", "R1"])

    def test_on_route_initialization_repairs_interior_link_and_counts_only_forward_length(self):
        def lane(key, block_id, start, end):
            return NS(id=key, get_roadblock_id=lambda: block_id, polygon=box(start, -1.75, end, 1.75),
                baseline_path=NS(linestring=LineString([(start, 0), (end, 0)]),
                                 discrete_path=[NS(x=start, y=0), NS(x=end, y=0)]),
                left_boundary=NS(linestring=LineString([(start, 1.75), (end, 1.75)])),
                right_boundary=NS(linestring=LineString([(start, -1.75), (end, -1.75)])),
                speed_limit_mps=10., adjacent_edges=(None, None), outgoing_edges=[])
        current, link, following = lane("L0", "R0", -300, 5), lane("L1", "R1", 5, 20), lane("L2", "R2", 20, 300)
        current.outgoing_edges, link.outgoing_edges = [link], [following]
        layers = NS(LANE="lane", LANE_CONNECTOR="connector", CROSSWALK="crosswalk",
                    ROADBLOCK="roadblock", INTERSECTION="intersection", CARPARK_AREA="carpark")
        nearby = {value: [] for value in vars(layers).values()}
        nearby[layers.LANE] = [current, link, following]
        modules = {name: ModuleType(name) for name in ("nuplan", "nuplan.common", "nuplan.common.actor_state",
            "nuplan.common.actor_state.state_representation", "nuplan.common.maps", "nuplan.common.maps.maps_datatypes")}
        modules["nuplan.common.actor_state.state_representation"].Point2D = lambda *args: args
        modules["nuplan.common.maps.maps_datatypes"].SemanticMapLayer = layers
        origin = np.zeros(3)
        with patch.dict(sys.modules, modules), patch.object(adapter, "_correct_route_order", return_value=["R0", "R1", "R2"]) as fix:
            result = adapter._map_packet(NS(get_proximal_map_objects=lambda *args: nearby), origin, ["R0", "R2"], 10., 160.)
        self.assertEqual(result[2], ["L0", "L1", "L2"])
        self.assertEqual(result[3], "L0")
        self.assertTrue(result[-1]["topology_correction_changed"])
        self.assertTrue(result[-1]["polygon_covers_rear_axle"])
        self.assertEqual(result[-1]["used_roadblock_ids"], ["R0", "R1", "R2"])
        self.assertGreater(LineString(result[1]).length - adapter._project(result[1], np.zeros(2))[0], 240.)
        fix.assert_called_once_with(unittest.mock.ANY, origin, ["R0", "R2"])

    def test_invalid_connector_coverage_cannot_mask_nearby_directed_route(self):
        def lane(key, block_id, points, bounds):
            line = LineString(points)
            return NS(id=key, get_roadblock_id=lambda: block_id, polygon=box(*bounds),
                baseline_path=NS(linestring=line, discrete_path=[NS(x=x, y=y) for x,y in points]),
                left_boundary=NS(linestring=LineString([(x,y+1.75) for x,y in points])),
                right_boundary=NS(linestring=LineString([(x,y-1.75) for x,y in points])),
                speed_limit_mps=10., adjacent_edges=(None,None), outgoing_edges=[])
        directed = lane("directed", "R1", [(-10,3.3),(300,3.3)], [-10,1.55,300,5.05])
        layers = NS(LANE="lane", LANE_CONNECTOR="connector", CROSSWALK="crosswalk",
                    ROADBLOCK="roadblock", INTERSECTION="intersection", CARPARK_AREA="carpark")
        nearby = {value: [] for value in vars(layers).values()}
        modules = {name: ModuleType(name) for name in ("nuplan", "nuplan.common", "nuplan.common.actor_state",
            "nuplan.common.actor_state.state_representation", "nuplan.common.maps", "nuplan.common.maps.maps_datatypes")}
        modules["nuplan.common.actor_state.state_representation"].Point2D = lambda *args: args
        modules["nuplan.common.maps.maps_datatypes"].SemanticMapLayer = layers
        # The second case models an observed wide intersection connector whose
        # polygon covers ego but baseline residual already exceeds the 8m gate.
        for points in ([(10,0),(-10,0)], [(-10,9.3),(300,9.3)]):
            with self.subTest(points=points):
                invalid = lane("invalid", "R0", points, [-10,-1.75,300,12])
                nearby[layers.LANE_CONNECTOR] = [invalid,directed]
                with patch.dict(sys.modules, modules), patch.object(adapter,"_correct_route_order",return_value=["R0","R1"]):
                    result = adapter._map_packet(NS(get_proximal_map_objects=lambda *args: nearby),np.zeros(3),["R0","R1"],10.,160.)
                self.assertEqual(result[3],"directed")
                self.assertFalse(result[-1]["polygon_covers_rear_axle"])
                self.assertAlmostEqual(result[-1]["projection_error_m"],3.3)
                self.assertEqual(result[-1]["heading_error_rad"],0.)

    def _correction_loses_valid_public_route_fixture(self):
        def lane(key, block, points, bounds):
            line = LineString(points)
            return NS(id=key, get_roadblock_id=lambda: block, polygon=box(*bounds),
                baseline_path=NS(linestring=line, discrete_path=[NS(x=x, y=y) for x, y in points]),
                left_boundary=NS(linestring=LineString([(x, y + 1.75) for x, y in points])),
                right_boundary=NS(linestring=LineString([(x, y - 1.75) for x, y in points])),
                speed_limit_mps=10., adjacent_edges=(None, None), outgoing_edges=[])
        # Hazelwood 38e9599c6c91e204f: an overlapping opposing connector makes
        # official correction prepend a wrong branch, then loop removal drops
        # the valid public-route suffix. Public residual remains within the
        # original gate and its direction agrees with the current pose.
        current = lane("public_current", "R0", [(-30, 3.7), (15, 3.7)], [-30, 1.95, 15, 5.45])
        following = lane("public_following", "R1", [(15, 3.7), (300, 3.7)], [15, 1.95, 300, 5.45])
        wrong = lane("opposing", "wrong", [(10, 0), (-10, 0)], [-10, -1.75, 10, 1.75])
        current.outgoing_edges = [following]
        layers = NS(LANE="lane", LANE_CONNECTOR="connector", CROSSWALK="crosswalk",
                    ROADBLOCK="roadblock", INTERSECTION="intersection", CARPARK_AREA="carpark")
        nearby = {value: [] for value in vars(layers).values()}
        nearby[layers.LANE] = [current, following]
        nearby[layers.LANE_CONNECTOR] = [wrong]
        modules = {name: ModuleType(name) for name in ("nuplan", "nuplan.common", "nuplan.common.actor_state",
            "nuplan.common.actor_state.state_representation", "nuplan.common.maps", "nuplan.common.maps.maps_datatypes")}
        modules["nuplan.common.actor_state.state_representation"].Point2D = lambda *args: args
        modules["nuplan.common.maps.maps_datatypes"].SemanticMapLayer = layers
        return modules, NS(get_proximal_map_objects=lambda *args: nearby)

    def test_invalid_correction_retains_only_connected_valid_public_route(self):
        modules, map_api = self._correction_loses_valid_public_route_fixture()
        origin = np.zeros(3)
        public_order, attempted_order = ["R0", "R1"], ["wrong"]
        with patch.dict(sys.modules, modules), \
                patch.object(adapter, "_correct_route_order", return_value=attempted_order) as correction, \
                patch.object(adapter, "_link_public_route", return_value=public_order) as linked:
            result = adapter._map_packet(map_api, origin, public_order, 10., 160.)
        correction.assert_called_once_with(map_api, origin, public_order)
        linked.assert_called_once_with(map_api, public_order)
        self.assertEqual(result[3], "public_current")
        self.assertEqual(result[2], ["public_current", "public_following"])
        audit = result[-1]
        self.assertTrue(audit["topology_correction_attempted"])
        self.assertFalse(audit["topology_correction_changed"])
        self.assertEqual(audit["public_roadblock_ids"], public_order)
        self.assertEqual(audit["used_roadblock_ids"], public_order)
        self.assertEqual(audit["attempted_roadblock_ids"], attempted_order)
        self.assertEqual(audit["topology_correction_rejection_reason"],
            "corrected_route_has_no_valid_current_association;connected_public_route_retained")
        self.assertEqual(audit["initial_lane_id"], "public_current")
        self.assertEqual(audit["current_lane_id"], "public_current")
        self.assertAlmostEqual(audit["projection_error_m"], 3.7)
        self.assertEqual(audit["heading_error_rad"], 0.)
        self.assertFalse(audit["polygon_covers_rear_axle"])

    def test_invalid_correction_cannot_retain_disconnected_public_route(self):
        modules, map_api = self._correction_loses_valid_public_route_fixture()
        with patch.dict(sys.modules, modules), \
                patch.object(adapter, "_correct_route_order", return_value=["wrong"]), \
                patch.object(adapter, "_link_public_route", return_value=None) as linked:
            with self.assertRaisesRegex(ValueError, "after topology correction"):
                adapter._map_packet(map_api, np.zeros(3), ["R0", "R1"], 10., 160.)
        linked.assert_called_once_with(map_api, ["R0", "R1"])

    def test_public_route_connectivity_checks_directed_edges_without_global_deduplication(self):
        first, connector, last = [NS(id=key, outgoing_edges=[]) for key in ("A", "B", "C")]
        first.outgoing_edges, connector.outgoing_edges, last.outgoing_edges = [connector], [last], [first]
        layers = NS(ROADBLOCK="roadblock", ROADBLOCK_CONNECTOR="connector")
        objects = {("A", layers.ROADBLOCK): first, ("B", layers.ROADBLOCK_CONNECTOR): connector,
                   ("C", layers.ROADBLOCK): last}
        queries = []
        def get_object(key, layer):
            queries.append((key, layer))
            return objects.get((key, layer))
        map_api = NS(get_map_object=get_object)
        modules = {name: ModuleType(name) for name in
            ("nuplan", "nuplan.common", "nuplan.common.maps", "nuplan.common.maps.maps_datatypes")}
        modules["nuplan.common.maps.maps_datatypes"].SemanticMapLayer = layers
        cases = [([], False), (["A"], True), (["A", "A", "B", "B", "C"], True),
                 (["A", "C"], False), (["B", "A"], False), (["A", "missing"], False),
                 (["A", "B", "A"], False), (["A", "B", "C", "A"], True)]
        with patch.dict(sys.modules, modules):
            for order, expected in cases:
                with self.subTest(order=order):
                    self.assertEqual(adapter._route_order_connected(map_api, order), expected)
            # A graph edge alone does not authenticate the corresponding map object.
            del objects[("B", layers.ROADBLOCK_CONNECTOR)]
            self.assertFalse(adapter._route_order_connected(map_api, ["A", "B"]))
        self.assertIn(("B", layers.ROADBLOCK), queries)
        self.assertIn(("B", layers.ROADBLOCK_CONNECTOR), queries)

    def _public_route_link_fixture(self, path_ids=None, found=True):
        nodes = {key: NS(id=key, outgoing_edges=[]) for key in ("R0", "link", "R1", "R2", "wrong")}
        nodes["R0"].outgoing_edges = [nodes["link"]]
        nodes["link"].outgoing_edges = [nodes["R1"]]
        nodes["R1"].outgoing_edges = [nodes["R2"]]
        layers = NS(ROADBLOCK="roadblock", ROADBLOCK_CONNECTOR="connector")
        def get_object(key, layer):
            expected_layer = layers.ROADBLOCK_CONNECTOR if key == "link" else layers.ROADBLOCK
            return nodes.get(key) if layer == expected_layer else None
        map_api = NS(get_map_object=get_object)
        path_ids = ["R0", "link", "R1"] if path_ids is None else path_ids
        search = Mock(return_value=(([nodes[key] for key in path_ids], path_ids), found))
        constructor = Mock(return_value=NS(search=search))
        modules = {name: ModuleType(name) for name in (
            "nuplan", "nuplan.common", "nuplan.common.maps", "nuplan.common.maps.maps_datatypes",
            "navsim", "navsim.planning", "navsim.planning.simulation", "navsim.planning.simulation.planner",
            "navsim.planning.simulation.planner.pdm_planner", "navsim.planning.simulation.planner.pdm_planner.utils",
            "navsim.planning.simulation.planner.pdm_planner.utils.graph_search",
            "navsim.planning.simulation.planner.pdm_planner.utils.graph_search.bfs_roadblock")}
        modules["nuplan.common.maps.maps_datatypes"].SemanticMapLayer = layers
        modules["navsim.planning.simulation.planner.pdm_planner.utils.graph_search.bfs_roadblock"].BreadthFirstSearchRoadBlock = constructor
        return modules, map_api, constructor, search, nodes

    def test_public_route_link_inserts_only_missing_forward_blocks_in_public_order(self):
        modules, map_api, constructor, search, _ = self._public_route_link_fixture()
        public_order = ["R0", "R1", "R2"]
        with patch.dict(sys.modules, modules):
            linked = adapter._link_public_route(map_api, public_order)
        self.assertEqual(linked, ["R0", "link", "R1", "R2"])
        self.assertEqual([key for key in linked if key in public_order], public_order)
        constructor.assert_called_once_with("R0", map_api, forward_search=True)
        search.assert_called_once_with("R1", max_depth=30)

    def test_public_route_link_rejects_failed_invalid_or_reordered_paths(self):
        cases = [
            ("not_found", ["R0", "link", "R1"], False),
            ("wrong_start", ["wrong", "link", "R1"], True),
            ("wrong_end", ["R0", "link", "wrong"], True),
            ("public_waypoint_inserted_early", ["R0", "R2", "R1"], True),
            ("path_not_directed", ["R0", "wrong", "R1"], True),
        ]
        for name, path_ids, found in cases:
            with self.subTest(reason=name):
                modules, map_api, constructor, search, _ = self._public_route_link_fixture(path_ids, found)
                with patch.dict(sys.modules, modules):
                    self.assertIsNone(adapter._link_public_route(map_api, ["R0", "R1", "R2"]))
                constructor.assert_called_once_with("R0", map_api, forward_search=True)
                search.assert_called_once_with("R1", max_depth=30)

    def test_public_route_link_rejects_mismatched_path_objects_and_final_graph(self):
        modules, map_api, _, search, nodes = self._public_route_link_fixture()
        search.return_value = (([nodes["R0"], nodes["wrong"], nodes["R1"]], ["R0", "link", "R1"]), True)
        with patch.dict(sys.modules, modules):
            self.assertIsNone(adapter._link_public_route(map_api, ["R0", "R1", "R2"]))
        modules, map_api, _, _, _ = self._public_route_link_fixture()
        # Even plausible returned IDs cannot bypass final adjacency validation.
        with patch.dict(sys.modules, modules), patch.object(adapter, "_route_order_connected", return_value=False):
            self.assertIsNone(adapter._link_public_route(map_api, ["R0", "R1", "R2"]))

    def test_official_cache_path_normalization(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            root = Path(directory).resolve()
            cache = root / "log/original/token/metric_cache.pkl"
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b"cache")
            metadata = root / "metadata/worker.csv"
            metadata.parent.mkdir()
            with metadata.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows([["file_name"], [str(cache)]])
            normalize_cache_metadata(dict(metric_cache_path=str(root)))
            with metadata.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[1][0].split("/")[-2], "token")
            cache.unlink()
            with self.assertRaisesRegex(ValueError, "missing/unsupported"):
                normalize_cache_metadata(dict(metric_cache_path=str(root)))

    def test_four_second_rear_axle_straight(self):
        times = np.arange(41) * .1
        states = np.c_[1.461 + 5 * times, np.zeros(41), np.zeros(41), np.full(41, 5.)]
        poses = trajectory_from_centers(states, .1, 1.461)
        np.testing.assert_allclose(poses[:, 0], np.arange(1, 9) * 2.5)
        np.testing.assert_allclose(poses[:, 1:], 0.)

    def test_rear_axle_turn_conversion(self):
        times = np.arange(41) * .1
        headings = .2 * times
        rear = np.c_[20 * np.sin(headings), 20 * (1 - np.cos(headings))]
        centers = rear + 1.461 * np.c_[np.cos(headings), np.sin(headings)]
        states = np.c_[centers, headings, np.full(41, 4.)]
        poses = trajectory_from_centers(states, .1, 1.461)
        np.testing.assert_allclose(poses[:, :2], rear[5::5], atol=1e-6)
        np.testing.assert_allclose(poses[:, 2], headings[5::5], atol=1e-6)

    def test_three_second_plan_is_not_padded(self):
        with self.assertRaisesRegex(ValueError, "four seconds"):
            trajectory_from_centers(np.zeros((31, 4)), .1, 1.461)

    def test_global_xy_rotation(self):
        actual = transform_points([[10, 22], [8, 20]], [10, 20, np.pi / 2])
        np.testing.assert_allclose(actual, [[2, 0], [0, 2]], atol=1e-10)

    def test_future_scene_frames_are_never_read(self):
        packet = packet_fixture()
        self.assertEqual(packet["identity"]["history_frames"], 4)
        self.assertEqual(packet["ego"]["position"], [1.461, 0.0])
        self.assertEqual(len(packet["camera_images"]), 8)
        self.assertEqual(packet["diagnostics"]["available_observation_history_s"], 1.5)

    def test_future_actor_history_rejected(self):
        packet = packet_fixture()
        packet["actors"][0]["history"][-1]["time"] = .5
        with self.assertRaisesRegex(ValueError, "Future"):
            validate_packet(packet)

    def test_only_current_tracks_and_named_obstacles(self):
        packet = packet_fixture()
        self.assertEqual([actor["id"] for actor in packet["actors"]], ["car"])
        self.assertIn("generic_object", packet["diagnostics"]["ignored_annotation_types"])
        np.testing.assert_allclose(packet["actors"][0]["acceleration"], [1., 0.])
        self.assertEqual([row["time"] for row in packet["actors"][0]["history"]], [-1.5, -1., -.5, 0.])

    def test_route_red_light_and_parallel_signal(self):
        packet = packet_fixture()
        self.assertAlmostEqual(packet["rules"]["stop_s"], 50.0)
        self.assertTrue(packet["rules"]["must_wait"])
        parallel = next(row for row in packet["rules"]["traffic_lights"] if row["id"] == "parallel")
        self.assertNotIn("stop_s", parallel)

    def test_snapshot_and_horizon(self):
        config = planning_config(json.loads((ROOT / "config/parameters.json").read_text(encoding="utf-8")), ROOT / "models/dinov2-base")
        self.assertEqual(config["prediction"]["horizon_s"], 4.0)
        self.assertEqual(config["mpc"]["horizon_steps"], 40)
        scene = snapshot_from_packet(packet_fixture(), config)
        self.assertEqual(scene.ego.wheelbase_m, 3.089)
        self.assertEqual(scene.ego.rear_axle_to_center_m, 1.461)
        self.assertEqual(scene.coordinates, "world_xy_m")

    def test_same_log_memory_is_rejected(self):
        record = dict(id="leak", provenance=dict(log_name="test_log", token="another_frame"))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            check_memory_disjoint([record], ["test_log"], ["anchor"])
        with self.assertRaisesRegex(ValueError, "no auditable"):
            check_memory_disjoint([dict(id="unknown")], ["test_log"], ["anchor"])
        with self.assertRaisesRegex(ValueError, "no auditable"):
            check_memory_disjoint([dict(id="unknown", situation={"scenario": "roundabout"})], ["test_log"], ["anchor"])
        check_memory_disjoint([dict(id="sim", platform="highway-env")], ["test_log"], ["anchor"])
        check_memory_disjoint([dict(id="train", provenance=dict(log_name="train_log", token="train_token"))],
                              ["test_log"], ["anchor"])

    def test_missing_failed_and_aggregate_scores_rejected(self):
        settings = dict(evaluation_tokens=["a", "b"], split="warmup_two_stage")
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            path = Path(directory) / "official.csv"
            def save(rows):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["token", "valid", "score"])
                    writer.writeheader()
                    writer.writerows(rows)
            rows = [dict(token="a", valid="True", score=.8), dict(token="b", valid="True", score=.7),
                    dict(token="extended_pdm_score_combined", valid="True", score=.6)]
            save(rows)
            self.assertEqual(verify_scores(settings, path)["official_two_stage_epdms"], .6)
            save(rows[1:])
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                verify_scores(settings, path)
            rows[1]["valid"] = "False"
            save(rows)
            with self.assertRaisesRegex(RuntimeError, "not a complete"):
                verify_scores(settings, path)

    def test_offline_memory_uses_authenticated_source_and_excludes_whole_log(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            root = Path(directory)
            def record(identity, log, token):
                folder = root / identity
                (folder / "audit").mkdir(parents=True)
                task = dict(id=identity, anchor_frame_id=log+":"+token,
                            capture=dict(source="openscene", source_log_name=log),
                            timeline=[dict(frame_id=log+":"+token)])
                raw = json.dumps(task).encode()
                (folder / "audit/task.json").write_bytes(raw)
                return dict(id=identity, frame_id=task["anchor_frame_id"], source="openscene",
                    record_origin="offline_retrospective", _asset_root=str(folder),
                    checksums={"audit/task.json":hashlib.sha256(raw).hexdigest()})
            train, test = record("train", "train_log", "train_token"), record("test", "test_log", "another_token")
            # A different anchor from the same test log is still excluded.
            selected, snapshot, audit = select_memory([train, test], "source_hash", ["test_log"], ["test_token"])
            self.assertEqual([r["id"] for r in selected], ["train"])
            self.assertEqual(audit["excluded_records"][0]["id"], "test")
            check_memory_disjoint(selected, ["test_log"], ["test_token"])
            with self.assertRaisesRegex(ValueError, "overlaps"):
                check_memory_disjoint([test], ["test_log"], ["test_token"])
            self.assertNotEqual(snapshot, select_memory([train, test], "changed_source", ["test_log"], ["test_token"])[1])
            (root / "train/audit/task.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audit changed"):
                select_memory([train], "source_hash", ["test_log"], [])

    def test_model_fallback_is_reported_separately_from_official_validity(self):
        from ep_driver_navsim.protocol import digest
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            root = Path(directory)
            path = root / "official.csv"
            path.write_text("token,valid,score\na,True,0.5\nb,True,0.5\nextended_pdm_score_combined,True,0.5\n", encoding="utf-8")
            settings = dict(evaluation_tokens=["a", "b"], split="warmup_two_stage", fast="api",
                            output_dir=str(root), memory_snapshot_id="frozen", memory_record_count=1)
            for token in ("a", "b"):
                target = root / "samples" / digest(token)[:24] / "trajectory.json"
                target.parent.mkdir(parents=True)
                target.write_text(json.dumps(dict(condition_audit=dict(model_requested=True,
                    model_answer_valid=token == "a", retrieved_memory_ids=["memory"]))), encoding="utf-8")
            result = verify_scores(settings, path)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["condition_audit"]["valid_model_answers"], 1)
            self.assertEqual(result["condition_audit"]["base_fallback_samples"], 1)
            self.assertFalse(result["condition_audit"]["all_samples_received_valid_model_answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
