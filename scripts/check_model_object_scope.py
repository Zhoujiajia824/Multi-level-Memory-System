"""验证模型BEV候选、八向范围与关注ID；不连接服务器或调用模型。"""
from pathlib import Path
import sys
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import yaml

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
    # 检索八向与模型动态候选范围独立；数值真值只发送一次。
    return json.loads(scene_text(scene, SimpleNamespace(grid=build_grid(scene))))


def model_ids(value):
    return {row['id'] for row in value['候选对象']}


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
        expected = {'near', 'second', 'far', 'rear', 'rear_far', 'bike', 'pedestrian',
                    'road_obstacle', 'two_lanes'}
        self.assertEqual(model_ids(value), expected)
        self.assertEqual(attention_ids(s), expected)
        self.assertNotIn('其他可见对象', value)
        self.assertNotIn('八向', json.dumps(value, ensure_ascii=False))
        rows=value['候选对象']
        self.assertEqual(len(rows),len(expected))
        self.assertFalse(any('所在区域' in row for row in rows))
        self.assertEqual(len(s.actors), 11)  # 原始审计快照不被原地删改。
        from nine_grid.representations import _bev_label_actors
        self.assertLessEqual({a.id for a in _bev_label_actors(s)} - {'ego'}, expected)
        planned, _ = planning_scene(s)
        self.assertLessEqual({a.id for a in planned.actors}, expected)

    def test_configured_lane_range_does_not_crop_bev_candidates(self):
        s = straight(); s.actors = [actor('inside', 18), actor('beyond', 22), actor('rear_out', -6)]
        config = {'planning_objects': {'front_m': 20., 'rear_m': 5.}}
        bind(s, config)
        value=payload(s)
        self.assertEqual(model_ids(value), {'inside', 'beyond', 'rear_out'})
        self.assertEqual(attention_ids(s), {'inside', 'beyond', 'rear_out'})
        _, diagnostic=scope_scene(s, config)
        self.assertEqual(diagnostic['settings'], {'front_m':20., 'rear_m':5.})
        self.assertEqual(diagnostic['selected_ids'],['inside'])
        planned, _ = planning_scene(s, config)
        self.assertEqual({a.id for a in planned.actors}, {'inside'})

    def test_forbidden_lane_and_unknown_permission(self):
        s = straight(False); s.actors = [actor('neighbor', 12, 3.5, lane='l')]
        bind(s); value = payload(s)
        self.assertEqual(model_ids(value),{'neighbor'})
        _, diagnostic=scope_scene(s)
        self.assertFalse(diagnostic['roads']['left']['crossable'])
        self.assertEqual(diagnostic['selected_ids'],[])
        s = straight(); s.lanes[0].left = 'r'; s.lanes[0].right = None
        s.actors=[actor('neighbor',12,3.5,lane='l')]
        bind(s); value = payload(s)
        _, diagnostic=scope_scene(s)
        self.assertFalse(diagnostic['roads']['left']['known'])
        self.assertEqual(model_ids(value),{'neighbor'})

    def test_invalid_attention_rejected_before_goal(self):
        s = straight(); s.actors = [actor('far', 85)]; bind(s)
        raw = json.dumps({'attention': [{'id': 'far', 'rank': 1, 'reason': 'test'}],
                          'local_goal': {'delta_s_m': 15., 'd_m': 0.}, 'explanation': {key: 'test' for key in ('情境', '依据', '参考答案', '适用条件')}})
        with self.assertRaisesRegex(ValueError, '关注 ID'):
            parse_answer(raw, s)

    def test_outside_lane_bev_attention_accepted_without_fabricated_region(self):
        s=straight(False)
        s.actors=[actor('opposing',12,3.5,heading=np.pi,lane='l',speed=8)]
        bind(s)
        value=payload(s)
        self.assertEqual([row['id'] for row in value['候选对象']],['opposing'])
        self.assertNotIn('所在区域',value['候选对象'][0])
        raw=json.dumps({'attention':[{'id':'opposing','rank':1,'reason':'接近左转通道'}],
                        'local_goal':{'delta_s_m':15.,'d_m':0.},
                        'explanation':{key:'test' for key in ('情境','依据','参考答案','适用条件')}})
        goal=dict(local_goal={'delta_s_m':15.,'d_m':0.},world_goal=np.array([15.,0.]),
                  goal_reference={},goal_id='test-goal',target_lane_id='c')
        with patch('nine_grid.models._goal_reference',return_value={}), \
                patch('nine_grid.goals.validate_local_goal',return_value=goal):
            answer=parse_answer(raw,s)
        self.assertEqual(answer.attention[0]['id'],'opposing')
        self.assertEqual(answer.frame_id,s.frame_id)

    def test_body_intrusion_gets_same_region_as_planner(self):
        s = straight(); s.lanes = s.lanes[:1]; s.lanes[0].left = s.lanes[0].right = None
        s.actors = [actor('intrusion', 10, 2.2, kind='static_obstacle', lane=None)]
        bind(s); value = payload(s)
        self.assertEqual([row['id'] for row in value['候选对象']],['intrusion'])
        self.assertNotIn('所在区域',value['候选对象'][0])
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
        # 横穿车真实占用当前车道且更近，不能因原属车道方向不同而忽略。
        self.assertEqual(g['slots']['front']['id'], 'cross_traffic')
        self.assertEqual(g['assignment_protocol'], 'lane-topology-role-assignment-1.1')
        self.assertEqual(model_ids(payload(s)), {'own', 'cross_traffic'})

    def test_reflection_does_not_repeat_excluded_objects(self):
        result = _reflection_diagnostics({'plan': {'planning_object_selection': {'excluded': [{'id': 'far'}]},
                                                  'potential_field': {'source': '八向势场内部诊断'},
                                                  'rule_observation_scope': {'source': '八向规则内部诊断'},
                                                  'safety_feasible': True},
                                          'quality': {'metrics': {'planning_scope': {'excluded': [{'id': 'far'}]}}}})
        self.assertNotIn('far', json.dumps(result))
        self.assertNotIn('八向', json.dumps(result, ensure_ascii=False))
        self.assertTrue(result['plan']['safety_feasible'])

    def test_system_prompts_are_compact_and_focus_on_task_and_output(self):
        prompts=yaml.safe_load((ROOT/'config/prompts.yaml').read_text(encoding='utf-8'))
        for name in ('fast','reflection'):
            with self.subTest(prompt=name):
                system=prompts[name]['system']
                self.assertLessEqual(len(system),2000)
                for term in ('八向','势场','检索算法'):
                    self.assertNotIn(term,system)
                for field in ('scene','decision','attention','local_goal','delta_s_m','d_m'):
                    self.assertIn(field,system)


if __name__ == '__main__':
    unittest.main(verbosity=2)
