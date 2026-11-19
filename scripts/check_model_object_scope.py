"""验证模型文字、关注ID与基础规划共享范围；不连接服务器或调用模型。"""
from pathlib import Path
import sys
import json
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from check_lane_object_scope import actor, straight, lane
from nine_grid.models import scene_text, attention_ids, parse_answer, _reflection_diagnostics
from nine_grid.object_scope import scope_scene
from nine_grid.planning_objects import planning_scene
from nine_grid.representations import build_grid, NineGridBuilder


def bind(scene, config=None):
    _, diagnostic = scope_scene(scene, config)
    scene.diagnostics['object_scope'] = diagnostic
    scene.diagnostics['visibility_wkt'] = scene.drivable.wkt
    scene.rules.update(desired_speed=12., speed_limit=15., must_wait=False, map_available=True)
    return scene


def payload(scene):
    # 故意给出较宽的检索图，模型范围仍须独立受约束。
    return json.loads(scene_text(scene, SimpleNamespace(grid=build_grid(scene))))


def model_ids(value):
    return {row['id'] for row in value['八向图']['对象'] + value['范围内其他对象']}


class ModelScopeTests(unittest.TestCase):
    def test_no_whole_scene_payload_and_static_retained(self):
        s = straight()
        s.actors = [actor('near', 15), actor('second', 23), actor('far', 65),
                    actor('rear', -18), actor('rear_far', -22),
                    actor('bike', 12, 3.5, kind='bicycle', lane='l'),
                    actor('pedestrian', 5, -3.5, kind='pedestrian', lane='r'),
                    actor('road_obstacle', 26, kind='static_obstacle'),
                    actor('offroad_obstacle', 10, 12, kind='static_obstacle'),
                    actor('pole', 10, kind='pole'), actor('two_lanes', 12, 7, lane='far')]
        bind(s)
        value = payload(s)
        expected = {'near', 'second', 'rear', 'bike', 'pedestrian', 'road_obstacle'}
        self.assertEqual(model_ids(value), expected)
        self.assertEqual(attention_ids(s), expected)
        self.assertNotIn('其他可见对象', value)
        self.assertEqual(value['对象输入范围']['对象数量'], 6)
        self.assertEqual(len(s.actors), 11)  # 原始审计快照不被原地删改。
        from nine_grid.representations import _bev_label_actors
        self.assertLessEqual({a.id for a in _bev_label_actors(s)} - {'ego'}, expected)
        planned, _ = planning_scene(s)
        self.assertLessEqual({a.id for a in planned.actors}, expected)

    def test_configured_range_applies_to_text_attention_and_plan(self):
        s = straight(); s.actors = [actor('inside', 18), actor('beyond', 22), actor('rear_out', -6)]
        config = {'planning_objects': {'front_m': 20., 'rear_m': 5.}}
        bind(s, config)
        self.assertEqual(model_ids(payload(s)), {'inside'})
        self.assertEqual(attention_ids(s), {'inside'})
        planned, _ = planning_scene(s, config)
        self.assertEqual({a.id for a in planned.actors}, {'inside'})

    def test_forbidden_lane_and_unknown_permission(self):
        s = straight(False); s.actors = [actor('neighbor', 12, 3.5, lane='l')]
        bind(s); value = payload(s)
        self.assertFalse(model_ids(value))
        self.assertIn('自车左侧相邻车道前方区域', value['八向图']['相邻车道存在但当前位置禁止跨越的区域'])
        s = straight(); s.lanes[0].left = 'r'; s.lanes[0].right = None
        bind(s); value = payload(s)
        self.assertIn('自车左侧相邻车道前方区域', value['八向图']['未知区域'])
        self.assertNotIn('自车左侧相邻车道前方区域', value['八向图']['未发现可见对象的区域'])

    def test_invalid_attention_rejected_before_goal(self):
        s = straight(); s.actors = [actor('far', 65)]; bind(s)
        raw = json.dumps({'attention': [{'id': 'far', 'rank': 1, 'reason': 'test'}],
                          'local_goal': {'delta_s_m': 15., 'd_m': 0.}, 'explanation': {key: 'test' for key in ('情境', '依据', '参考答案', '适用条件')}})
        with self.assertRaisesRegex(ValueError, '关注 ID'):
            parse_answer(raw, s)

    def test_body_intrusion_gets_same_region_as_planner(self):
        s = straight(); s.lanes = s.lanes[:1]; s.lanes[0].left = s.lanes[0].right = None
        s.actors = [actor('intrusion', 10, 2.2, kind='static_obstacle', lane=None)]
        bind(s); value = payload(s)
        self.assertEqual(value['八向图']['对象'][0]['所在区域'], ['自车车道前方区域'])
        _, d = planning_scene(s)
        self.assertEqual(d['region_objects']['front'], 'intrusion')

    def test_topology_grid_follows_curve_and_disambiguates_junction(self):
        s = straight()
        cross = lane('cross', [[10, -30], [10, 30]], width=16)
        s.lanes.append(cross)
        s.actors = [actor('cross_traffic', 10, 0, heading=np.pi/2, lane='cross', speed=3),
                    actor('own', 12, speed=3)]
        bind(s)
        g = build_grid(s)
        self.assertEqual(g['slots']['front']['id'], 'own')
        self.assertEqual(g['assignment_protocol'], 'lane-topology-role-assignment-1.0')
        self.assertEqual(model_ids(payload(s)), {'own'})

    def test_reflection_does_not_repeat_excluded_objects(self):
        result = _reflection_diagnostics({'plan': {'planning_object_selection': {'excluded': [{'id': 'far'}]},
                                                  'safety_feasible': True},
                                          'quality': {'metrics': {'planning_scope': {'excluded': [{'id': 'far'}]}}}})
        self.assertNotIn('far', json.dumps(result))
        self.assertTrue(result['plan']['safety_feasible'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
