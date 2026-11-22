"""HTTP contract and read-only asset boundaries, using a disposable episode."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from epviz.server import ViewerServer, safe_json, library_wire_frame


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.episode = cls.root / "场景 seed1"
        cls.episode.mkdir()
        (cls.episode / "configuration.json").write_text(
            json.dumps({"api_key": "private", "scenario": "test"}), encoding="utf-8")
        cls.plan_path = cls.episode / "plans.jsonl"
        cls.plan_path.write_text(json.dumps(dict(frame_id="test:0", time=0,
            plan=dict(frame_id="test:0", time=0, states=[], controls=[]))) + "\n", encoding="utf-8")
        (cls.episode / "video.mp4").write_bytes(b"0123456789abcdef")
        cls.server = ViewerServer(("127.0.0.1", 0), cls.root)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.identifier = cls.get_json("/api/episodes")["episodes"][0]["id"]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=3)
        cls.temporary.cleanup()

    @classmethod
    def get_json(cls, path):
        with urlopen(cls.url + path, timeout=5) as response:
            return json.load(response)

    def test_health_manifest_and_no_credential_disclosure(self):
        self.assertTrue(self.get_json("/api/health")["read_only"])
        manifest = self.get_json("/api/episodes/" + self.identifier)
        self.assertEqual(manifest["configuration"]["api_key"], "[redacted]")
        self.assertEqual(manifest["times"][0], 0)

    def test_optional_memory_library_is_explicitly_unconfigured(self):
        document = self.get_json('/api/library')
        self.assertFalse(document['available'])
        self.assertIn('--memory-root', document['reason'])

    def test_static_page_and_head(self):
        request = Request(self.url + "/", method="HEAD")
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/html", response.headers["Content-Type"])
            self.assertGreater(int(response.headers["Content-Length"]), 0)
            self.assertEqual(response.read(), b"")

    def test_existing_listener_is_not_shared_or_replaced(self):
        with self.assertRaises(OSError):
            ViewerServer(self.server.server_address, self.root)
        self.assertEqual(self.get_json("/api/health")["status"], "ok")

    def test_bundled_car_and_episode_asset_namespaces_do_not_collide(self):
        with urlopen(self.url + "/assets/ego-car.glb", timeout=5) as response:
            self.assertEqual(response.read(4), b"glTF")
        with urlopen(self.url + "/assets/" + self.identifier + "/video.mp4", timeout=5) as response:
            self.assertEqual(response.read(), b"0123456789abcdef")

    def test_video_partial_content_and_suffix(self):
        url = self.url + "/api/asset?episode=" + self.identifier + "&path=video.mp4"
        for spec, expected in (("bytes=2-6", b"23456"), ("bytes=-3", b"def"), ("bytes=12-", b"cdef")):
            with urlopen(Request(url, headers={"Range": spec}), timeout=5) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), expected)
                self.assertIn("Content-Range", response.headers)
        with self.assertRaises(HTTPError) as failure:
            urlopen(Request(url, headers={"Range": "bytes=100-200"}), timeout=5)
        self.assertEqual(failure.exception.code, 416)

    def test_asset_and_static_traversal_rejected(self):
        for path in ("/api/asset?episode=" + self.identifier + "&path=" + quote("../secret"),
                     "/%2e%2e/epviz/server.py"):
            with self.assertRaises(HTTPError) as failure:
                urlopen(self.url + path, timeout=5)
            self.assertEqual(failure.exception.code, 400)

    def test_live_file_append_visible_without_restart(self):
        with self.plan_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(frame_id="test:1", time=.1,
                plan=dict(frame_id="test:1", time=.1, states=[], controls=[]))) + "\n")
        manifest = self.get_json("/api/episodes/" + self.identifier)
        self.assertIn(.1, manifest["times"])
        self.assertTrue(manifest["frames"][-1]["pending"])

    def test_json_nonfinite_and_nested_secrets(self):
        result = safe_json({"values": [float("nan"), float("inf"), 1],
                            "auth": {"authorization": "Bearer private"}})
        self.assertEqual(result["values"], [None, None, 1])
        self.assertEqual(result["auth"]["authorization"], "[redacted]")

    def test_library_payload_keeps_render_geometry_without_duplicate_archive(self):
        polygon = dict(outer=[[0, 0], [10, 0], [10, 4], [0, 4]], holes=[])
        frame = dict(frame_id='f', scene=dict(frame_id='f', lanes=['large-map']),
            ego=dict(position=[0, 0], history=['historical-only']),
            visualization=dict(roads=['duplicate-road'], render_geometry=dict(
                road_surface=dict(polygons=[polygon], recorded_polygons=[polygon]),
                markings=[dict(points=[[0, 0], [10, 0]], kind='solid', source_lane_ids=['a', 'b'])])))
        result = library_wire_frame(frame)
        self.assertEqual(result['visualization']['render_geometry']['road_surface']['polygons'], [polygon])
        self.assertEqual(result['visualization']['render_geometry']['markings'][0]['kind'], 'solid')
        self.assertNotIn('lanes', result['scene'])
        self.assertNotIn('history', result['ego'])
        self.assertEqual(frame['scene']['lanes'], ['large-map'])
        self.assertEqual(frame['ego']['history'], ['historical-only'])

    def test_library_wire_sends_one_copy_of_road_topology(self):
        topology = dict(edges=[dict(path=[[0, 0, 0], [10, 0, 0]], successors=['b'])])
        frame = dict(visualization=dict(topology=topology, render_geometry=dict(topology=topology)))
        result = library_wire_frame(frame)
        self.assertEqual(result['visualization']['render_geometry']['topology'], topology)
        self.assertNotIn('topology', result['visualization'])
        self.assertIn('topology', frame['visualization'])

    def test_unfinished_video_is_not_transcoded_and_general_writes_are_refused(self):
        response = self.get_json("/api/episodes/" + self.identifier + "/video")
        self.assertEqual(response["status"], "unavailable")
        with self.assertRaises(HTTPError) as failure:
            urlopen(Request(self.url + "/api/episodes", method="POST"), timeout=5)
        self.assertEqual(failure.exception.code, 405)


class LibraryHTTPTests(unittest.TestCase):
    """Route binding is independent of simulator records and their root."""

    @classmethod
    def setUpClass(cls):
        class Library:
            def summary(self):
                return dict(read_only=True, count=1, frozen=True, available=True)

            def records(self, **kwargs):
                return dict(records=[], total=0, request=kwargs)

            def detail(self, identity, evidence_index=0):
                return dict(id=identity, selected_evidence=evidence_index)

            def scene_data(self, identity, evidence_index=0):
                return dict(available=False, record_id=identity, evidence_index=evidence_index,
                            reason='该证据没有归档场景')

        cls.server = ViewerServer(('127.0.0.1', 0))
        cls.server.library = Library()
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=3)

    def get_json(self, path):
        with urlopen(self.url + path, timeout=5) as response:
            return json.load(response)

    def test_library_only_does_not_require_experiment_files(self):
        self.assertTrue(self.get_json('/api/library')['available'])
        self.assertFalse(self.get_json('/api/library')['experiments_configured'])
        self.assertEqual(self.get_json('/api/episodes')['episodes'], [])
        with self.assertRaises(HTTPError) as failure:
            self.get_json('/api/episodes/missing/frames/0')
        self.assertEqual(failure.exception.code, 404)

    def test_library_only_home_opens_library_instead_of_empty_experiment(self):
        for path in ('/', '/index.html?record=record_a'):
            with urlopen(self.url + path, timeout=5) as response:
                suffix = '?record=record_a' if '?' in path else ''
                self.assertEqual(response.geturl(), self.url + '/memory' + suffix)
                self.assertIn('驾驶经验库', response.read().decode('utf-8'))

    def test_search_and_evidence_are_bound_to_requested_library_record(self):
        query = quote('变道 经验')
        result = self.get_json(f'/api/library/records?q={query}&source=highway&offset=40&limit=20')
        self.assertEqual(result['request'], dict(query='变道 经验', source='highway', status='', offset=40, limit=20))
        detail = self.get_json('/api/library/records/record_a?evidence=2')
        self.assertEqual(detail, dict(id='record_a', selected_evidence=2))
        scene = self.get_json('/api/library/records/record_a/scene?evidence=3')
        self.assertEqual(scene['record_id'], 'record_a')
        self.assertEqual(scene['evidence_index'], 3)
        self.assertFalse(scene['available'])

    def test_library_mutations_and_invalid_paging_are_rejected(self):
        with self.assertRaises(HTTPError) as failure:
            urlopen(Request(self.url + '/api/library/records/record_a', method='POST'), timeout=5)
        self.assertEqual(failure.exception.code, 405)
        with self.assertRaises(HTTPError) as failure:
            self.get_json('/api/library/records?offset=not-a-number')
        self.assertEqual(failure.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
