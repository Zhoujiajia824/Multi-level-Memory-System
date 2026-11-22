"""Frozen/synthetic regression for distinct retrieval, perception and MPC object sets.

No simulator step, model request, sensor connection or experiment-file mutation.
An optional --scene path checks an existing archived intersection frame as well.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace
import argparse
import json
import os
import sys
import unittest

os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import unary_union
from check_lane_object_scope import actor, lane, straight
from nine_grid.contracts import Actor, Lane, SceneSnapshot, PlanResult, PredictionBundle
from nine_grid.models import attention_ids, parse_answer, scene_text
from nine_grid.object_scope import scope_scene
from nine_grid.perception_scope import perception_scene, bev_viewport_settings, bev_pixels
from nine_grid.planning_context import prepare_planning_context
from nine_grid.planning_objects import planning_scene, planning_view
from nine_grid.planner import Planner, propagate
from nine_grid.quality import evaluate
from nine_grid.representations import build_grid, _bev_label_actors

ARCHIVED_SCENE = None


def config():
    return json.loads((ROOT / 'config/parameters.json').read_text(encoding='utf-8'))


def bind(scene):
    for item in scene.lanes:
        for index, edge in enumerate(item.boundaries):
            edge.setdefault('id', item.id + ':boundary:' + str(index))
    scene.drivable = unary_union([Polygon(item.polygon) for item in scene.lanes])
    scene.diagnostics['visibility_wkt'] = scene.drivable.wkt
    scene.rules.update(desired_speed=5., speed_limit=15., must_wait=False, map_available=True)
    return scene


def candidates(scene, settings=None):
    result, diagnostic = perception_scene(scene, settings)
    return {item.id for item in result.actors}, diagnostic


def answer(scene, ids):
    raw = json.dumps(dict(attention=[dict(id=identity, rank=index + 1, reason='current observed conflict')
                                    for index, identity in enumerate(ids)],
                          local_goal=dict(delta_s_m=15., d_m=0.),
                          explanation={key: 'current-frame evidence' for key in ('情境', '依据', '参考答案', '适用条件')}))
    return parse_answer(raw, scene, config=config())


def model_objects(scene):
    text = json.loads(scene_text(scene, SimpleNamespace(grid=build_grid(scene))))
    rows = text['候选对象']
    return {row['id'] for row in rows}, rows, text


def read_frozen(path):
    """Only deserialize public archived observations; never construct an environment."""
    value = json.loads(Path(path).read_text(encoding='utf-8'))

    def entity(row):
        result = {key: deepcopy(item) for key, item in row.items() if key in {f.name for f in fields(Actor)}}
        for key in ('position', 'velocity', 'acceleration'):
            result[key] = np.asarray(result[key], float)
        return Actor(**result)

    lanes = []
    for row in value['lanes']:
        data = {key: deepcopy(item) for key, item in row.items() if key in {f.name for f in fields(Lane)}}
        data['center'] = np.asarray(data['center'], float)
        data['polygon'] = np.asarray(wkt.loads(row['polygon_wkt']).exterior.coords)
        lanes.append(Lane(**data))
    return SceneSnapshot(value['frame_id'], value['time'], entity(value['ego']),
                         [entity(row) for row in value['actors']], lanes, np.asarray(value['route'], float),
                         value['route_lane_ids'], wkt.loads(value['drivable_wkt']), value['scenario'],
                         rules=deepcopy(value['rules']), assets=deepcopy(value.get('sensor_assets', {})),
                         diagnostics=deepcopy(value['diagnostics']), version=value['version'],
                         coordinates=value['coordinates'])


class PerceptionSetTests(unittest.TestCase):
    def test_bev_dynamics_are_not_limited_by_lane_permission_or_eight_slots(self):
        scene = straight(False)
        scene.actors = [actor('near', 10., speed=5.), actor('farther_same_region', 24., speed=5.),
                        actor('opposing', 16., 3.5, heading=np.pi, lane='l', speed=4.),
                        actor('walker', 13., 10., kind='pedestrian', lane=None),
                        actor('bike', 18., -9., kind='bicycle', lane=None),
                        actor('front_bev', 65.), actor('rear_bev', -27.),
                        actor('offroad_obstacle', 15., 10., kind='static_obstacle', lane=None),
                        actor('road_obstacle', 30., kind='static_obstacle'),
                        actor('pole', 8., kind='pole'), actor('outside', 80., 10.)]
        bind(scene)
        original = [a.id for a in scene.actors]
        expected = {'near', 'farther_same_region', 'opposing', 'walker', 'bike',
                    'front_bev', 'rear_bev', 'road_obstacle'}
        chosen, d = candidates(scene)
        self.assertEqual(chosen, expected)
        self.assertEqual(set(d['lane_scope_ids']), {'near', 'farther_same_region', 'road_obstacle'})
        self.assertIn('opposing', d['outside_lane_ids'])
        self.assertEqual(d['objects']['opposing']['memberships'], {})
        self.assertEqual(d['objects']['opposing']['sources'], ['bev_dynamic'])
        self.assertEqual(model_objects(scene)[0], expected)
        self.assertEqual(attention_ids(scene), expected)
        self.assertEqual(len(model_objects(scene)[1]), len(expected))  # one truth record per ID
        self.assertNotIn('八向',json.dumps(model_objects(scene)[2],ensure_ascii=False))
        self.assertEqual([a.id for a in scene.actors], original)
        self.assertFalse(d['lane_scope']['roads']['left']['crossable'])

    def test_exact_rendered_rectangle_and_partial_body_visibility(self):
        scene = straight(False)
        scene.actors = [actor('front_touch', 72., 10.), actor('front_out', 72.01, 10.),
                        actor('rear_touch', -32., 10.), actor('rear_out', -32.01, 10.),
                        actor('left_touch', 10., 25.9), actor('left_out', 10., 25.91),
                        actor('right_touch', 10., -25.9), actor('right_out', 10., -25.91)]
        chosen, d = candidates(scene)
        self.assertEqual(chosen, {'front_touch', 'rear_touch', 'left_touch', 'right_touch'})
        self.assertEqual(d['settings']['bev'], bev_viewport_settings())
        np.testing.assert_allclose(bev_pixels([[70., 25.], [-30., -25.]], scene.ego), [[0., 0.], [500., 1000.]])

    def test_viewport_uses_ego_orientation_and_world_translation(self):
        scene = straight(False)
        scene.actors = [actor('inside', 16., 10.), actor('outside', 16., 29.)]
        theta = 1.2
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        origin = np.array([112., -75.])
        expected = candidates(scene)[0]
        for obj in [scene.ego] + scene.actors:
            obj.position = obj.position @ rotation.T + origin
            obj.velocity = obj.velocity @ rotation.T
            obj.heading += theta
        for item in scene.lanes:
            item.center = item.center @ rotation.T + origin
            item.polygon = item.polygon @ rotation.T + origin
            for edge in item.boundaries:
                edge['points'] = np.asarray(edge['points']) @ rotation.T + origin
        scene.route = scene.route @ rotation.T + origin
        bind(scene)
        self.assertEqual(candidates(scene)[0], expected)

    def test_invisible_and_history_only_objects_are_not_restored(self):
        scene = straight()
        hidden = actor('hidden', 12.)
        hidden.visible = False
        hidden.history = [dict(time=0., position=[12., 0.])]
        scene.actors = [hidden, actor('visible', 15., 8.)]
        scene.diagnostics['historical_only_objects'] = {'old': [dict(time=0., position=[10., 0.])]}
        self.assertEqual(candidates(scene)[0], {'visible'})

    def test_lane_arc_scope_is_retained_when_curve_leaves_bev_rectangle(self):
        scene = straight(False)
        scene.lanes = [lane('c', [[-60., 0.], [0., 0.], [2., 2.], [2., 60.]])]
        scene.route = scene.lanes[0].center
        scene.route_lane_ids = ['c']
        scene.actors = [actor('curved_road_obstacle', 2., 35., kind='static_obstacle', heading=np.pi/2)]
        bind(scene)
        chosen, d = candidates(scene)
        self.assertEqual(chosen, {'curved_road_obstacle'})
        self.assertEqual(d['lane_scope_ids'], ['curved_road_obstacle'])
        self.assertEqual(d['bev_dynamic_ids'], [])

    def test_retrieval_representatives_do_not_expand_with_bev_candidates(self):
        scene = straight(False)
        scene.actors = [actor('near', 10.), actor('far', 24.)]
        before = build_grid(scene)
        scene.actors += [actor('crossing', 18., 10., heading=-np.pi/2, lane=None, speed=4.)]
        after = build_grid(scene)
        self.assertEqual(after['slots']['front']['id'], 'near')
        self.assertEqual({key: item.get('id') for key, item in before['slots'].items()},
                         {key: item.get('id') for key, item in after['slots'].items()})
        self.assertIn('crossing', attention_ids(scene))
        self.assertIn('crossing', {item.id for item in _bev_label_actors(scene)})

    def test_perception_reentry_rejects_missing_or_mutated_frozen_candidates(self):
        scene = straight(False)
        scene.actors = [actor('outside_lane', 14., 10.), actor('outside_bev', 90., 10.)]
        scoped, d = perception_scene(scene)
        repeated, other = perception_scene(scoped)
        self.assertEqual(other, d)
        self.assertEqual({a.id for a in repeated.actors}, {'outside_lane'})
        missing = deepcopy(scoped)
        missing.actors = []
        with self.assertRaises(ValueError):
            perception_scene(missing)
        changed = deepcopy(scoped)
        changed.actors[0].position[0] += 1.
        with self.assertRaises(ValueError):
            perception_scene(changed)

    def test_attention_accepts_bev_candidates_but_not_unprovided_ids(self):
        scene = bind(straight(False))
        scene.actors = [actor('outside_lane', 12., 8.), actor('bike', 18., 8., kind='bicycle'),
                        actor('walker', 20., 10., kind='pedestrian'), actor('fourth', 23., 10.),
                        actor('outside_bev', 90., 10.)]
        accepted = answer(scene, ['outside_lane', 'bike', 'walker'])
        self.assertEqual(len(accepted.attention), 3)
        for identities in (['outside_bev'], ['outside_lane', 'bike', 'walker', 'fourth']):
            with self.subTest(ids=identities), self.assertRaises(ValueError):
                answer(scene, identities)


class PredictionAndPlannerSetTests(unittest.TestCase):
    def synthetic(self):
        scene = bind(straight(False))
        scene.ego.velocity[:] = [5., 0.]
        scene.actors = [actor('near', 22., speed=5.), actor('far', 32., speed=5.),
                        actor('outside_lane', 15., 3.5, lane='l'), actor('outside_bev', 90., 10.)]
        return scene

    def test_ekf_all_candidates_but_base_mpc_all_lane_objects(self):
        scene = self.synthetic()
        prepared, predictions = prepare_planning_context(scene, config())
        self.assertEqual({a.id for a in prepared.actors}, {'near', 'far', 'outside_lane'})
        self.assertEqual(set(predictions.trajectories), {'near', 'far', 'outside_lane'})
        self.assertEqual(set(predictions.actors), set(predictions.trajectories))
        for identity, trajectory in predictions.trajectories.items():
            np.testing.assert_allclose(trajectory[0, :2], predictions.actors[identity].position)
        planned, subset, selected = planning_view(prepared, predictions, config())
        self.assertEqual({a.id for a in planned.actors}, {'near', 'far'})
        self.assertEqual(set(subset.trajectories), {'near', 'far'})
        self.assertEqual(selected['base_selected_ids'], ['near'])
        self.assertEqual(set(selected['lane_scope_ids']), {'near', 'far'})
        self.assertEqual(set(selected['selected_ids']), {'near', 'far'})

    def test_attention_adds_external_target_without_forging_eight_membership(self):
        scene = self.synthetic()
        selected, d = planning_scene(scene, config(), answer(scene, ['outside_lane']))
        self.assertEqual({a.id for a in selected.actors}, {'near', 'far', 'outside_lane'})
        self.assertEqual(d['selected_state']['outside_lane']['memberships'], {})
        self.assertIn('outside_lane', d['attention_added_ids'])
        self.assertEqual(d['region_objects']['front'], 'near')
        self.assertNotIn('outside_lane', set(d['region_objects'].values()))

    def test_original_frame_prediction_reuse_and_wrong_frame_rejection(self):
        scene = self.synthetic()
        prepared, original = prepare_planning_context(scene, config())
        other, reused = prepare_planning_context(scene, config(), answer(scene, ['outside_lane']), existing=original)
        self.assertEqual(set(reused.trajectories), set(original.trajectories))
        for identity in original.trajectories:
            np.testing.assert_array_equal(reused.trajectories[identity], original.trajectories[identity])
        bad = deepcopy(original)
        bad.frame_id = 'future:1'
        with self.assertRaises(ValueError):
            prepare_planning_context(scene, config(), existing=bad)
        changed = deepcopy(scene)
        changed.actors[0].position[0] += .2
        with self.assertRaises(ValueError):
            prepare_planning_context(changed, config(), existing=original)

    def test_old_planning_cache_cannot_preserve_nearest_only_selection(self):
        scene = self.synthetic()
        subset, old = planning_scene(scene, config())
        subset.diagnostics['planning_scope']['version'] = 'frenet-eight-planning-3.0'
        with self.assertRaises(ValueError):
            planning_scene(subset, config())

    def test_actual_mpc_cost_and_collision_constraints_include_all_targets(self):
        scene = self.synthetic()
        c = config()
        observed, predictions = prepare_planning_context(scene, c)
        base = Planner(c).solve(observed, predictions)
        self.assertEqual(set(base.diagnostics['potential_field']['source_ids']), {'near', 'far'})
        for identity in ('near', 'far'):
            self.assertIn('actor_clearance:' + identity, base.diagnostics['constraint_minima'])
        self.assertNotIn('actor_clearance:outside_lane', base.diagnostics['constraint_minima'])
        semantic = answer(scene, ['outside_lane', 'near'])
        attended = Planner(c).solve(observed, predictions, semantic)
        self.assertEqual(set(attended.diagnostics['potential_field']['source_ids']), {'near', 'far', 'outside_lane'})
        self.assertIn('actor_clearance:outside_lane', attended.diagnostics['constraint_minima'])
        gain = attended.diagnostics['potential_field']['attention_gain_by_id']
        self.assertEqual(gain['outside_lane'], 2.)
        self.assertEqual(gain['near'], 1.5)
        self.assertEqual(gain['far'], 1.)
        # Formula is independent of optimizer choices: evaluate its accepted rollout,
        # including prediction motion, and verify each physical source exactly once.
        expected = 0.
        for identity in ('near', 'far', 'outside_lane'):
            item = predictions.actors[identity]
            source = predictions.trajectories[identity][:len(attended.states)]
            xy = attended.states[1:, :2] - source[1:, :2]
            risk = c['risk']['classes'][item.kind]
            rho = np.sqrt((xy[:, 0] / risk['longitudinal_m']) ** 2 + (xy[:, 1] / risk['lateral_m']) ** 2)
            t = np.maximum(1. - rho, 0.)
            expected += c['mpc']['weights']['risk'] * risk['peak'] * gain[identity] * np.mean(t**3 * (1. + 3.*rho + 6.*rho**2))
        self.assertAlmostEqual(attended.diagnostics['cost_terms']['actor_risk'], expected, places=6)
        brake=Planner(c).emergency_brake(observed,predictions,'offline regression',answer=semantic)
        self.assertIn('outside_lane',brake.diagnostics['potential_field']['source_ids'])
        self.assertEqual(brake.diagnostics['potential_field']['attention_gain_by_id']['outside_lane'],2.)
        self.assertIn('actor_clearance:outside_lane',brake.diagnostics['constraint_minima'])

    def test_reference_collision_with_unattended_bev_candidate_invalidates_quality(self):
        scene = bind(straight(False))
        scene.ego.velocity[:] = [5., 0.]
        crossing = actor('unattended_crossing', 8., 5., heading=-np.pi/2, lane=None, speed=5.)
        scene.actors = [crossing]
        observed, d = perception_scene(scene)
        self.assertEqual(d['lane_scope_ids'], [])
        times = np.arange(40) * .1
        trajectory = np.c_[8. + np.zeros_like(times), 5. - 5.*times,
                            np.full_like(times, -np.pi/2), np.full_like(times, 5.)]
        predictions = PredictionBundle(scene.frame_id, scene.time, times,
                                       {crossing.id: trajectory}, {crossing.id: crossing})
        baseline_states = np.c_[np.arange(31)*.5, np.zeros((31, 2)), np.full(31, 5.)]
        baseline = PlanResult(scene.frame_id, scene.time, True, baseline_states, np.zeros((30, 2)),
                              diagnostics={'safety_feasible': True, 'emergency': False})
        states = [np.array([0., 0., 0., 5.])]
        controls = []
        for _ in range(30):
            control = np.array([-min(3., states[-1][3]/.1), 0.])
            controls.append(control)
            states.append(propagate(states[-1], control, scene.ego.length))
        plan = PlanResult(scene.frame_id, scene.time, True, np.asarray(states), np.asarray(controls),
                          diagnostics={'safety_feasible': True, 'emergency': False})
        quality = evaluate(observed, plan, baseline, predictions, config())
        self.assertFalse(quality.valid)
        self.assertIsNone(quality.score)
        self.assertEqual(quality.metrics['reference_perception_collision_ids'], [crossing.id])
        self.assertEqual(quality.metrics['evaluated_actor_ids'], [crossing.id])
        self.assertFalse(quality.collision)  # invalid reference is independent of the safer candidate

    def test_archived_intersection_missing_vehicle_is_candidate_and_attention_target(self):
        if ARCHIVED_SCENE is None:
            self.skipTest('pass --scene to check an existing archived frame')
        scene = read_frozen(ARCHIVED_SCENE)
        raw_ids = [a.id for a in scene.actors]
        _, original_lane = scope_scene(scene)
        observed, d = perception_scene(scene, config())
        self.assertNotIn('object_00003', original_lane['selected_ids'])
        self.assertIn('object_00003', d['outside_lane_ids'])
        self.assertIn('object_00003', attention_ids(observed))
        model_ids, _, text = model_objects(scene)
        self.assertIn('object_00003', model_ids)
        self.assertIn('object_00003', {row['id'] for row in text['候选对象']})
        self.assertFalse(any('所在区域' in row for row in text['候选对象']))
        prepared, predictions = prepare_planning_context(scene, config())
        self.assertIn('object_00003', predictions.trajectories)
        base, base_d = planning_scene(scene, config())
        self.assertNotIn('object_00003', base_d['selected_ids'])
        semantic = SimpleNamespace(attention=[dict(id='object_00003', rank=1)])
        selected, selected_d = planning_scene(scene, config(), semantic)
        self.assertIn('object_00003', selected_d['selected_ids'])
        self.assertEqual(selected_d['selected_state']['object_00003']['memberships'], {})
        self.assertEqual([a.id for a in scene.actors], raw_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--scene', type=Path)
    args, remaining = parser.parse_known_args()
    if args.scene is not None:
        if not args.scene.is_file():
            parser.error('archived scene does not exist: ' + str(args.scene))
        ARCHIVED_SCENE = args.scene
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
