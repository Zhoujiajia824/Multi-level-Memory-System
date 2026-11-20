"""Physical evaluation regression checks. No CARLA server, model, or world tick."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nine_grid.carla_evaluation import EpisodeEvaluator, CarlaEvaluationProbe, _boundary_topology


def state(x=0., speed=2., identity='ego'):
    return {'id': identity, 'kind': 'vehicle', 'position': [x, 0.], 'z': .8,
            'heading': 0., 'length': 4., 'width': 2., 'velocity': [speed, 0.],
            'acceleration': [0., 0.], 'lane_id': '1|0|-1'}


def road():
    return {'body_on_road': True, 'wrong_way': False, 'speed_limit_mps': 10.,
            'coverage': {k: True for k in ('collision_events', 'lane_markings', 'traffic_lights',
                                          'stop_signs', 'public_priority')}}


def observe(e, tick, ego=None, actors=(), events=(), scenario=None, measured_road=None):
    return e.observe(tick, 100+tick, 123.+tick*.019999999552965164,
                     ego or state(tick*.04), list(actors), measured_road or road(), events,
                     scenario or {'scenario_valid': True, 'events': []})


class EvaluationTests(unittest.TestCase):
    def evaluator(self, **kwargs):
        return EpisodeEvaluator([[0., 0.], [100., 0.]], **kwargs)

    def test_physical_metrics_independent_of_planner_frequency(self):
        reports = []
        for planner_hz in (1, 2, 10):
            e = self.evaluator()
            observed_by_planner = []
            for tick in range(151):
                r = road()
                r['body_on_road'] = not 71 <= tick <= 78
                out = observe(e, tick, measured_road=r)
                if tick % (50//planner_hz) == 0:
                    observed_by_planner.append(out)
            result = e.report('timeout')
            self.assertIn('offroad', result['violations'])
            self.assertAlmostEqual(result['violation_seconds']['offroad'], .16)
            self.assertTrue(result['evaluation_valid'])
            reports.append(result)
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[1], reports[2])

    def test_timeout_never_counts_as_success(self):
        e = self.evaluator()
        observe(e, 0)
        self.assertFalse(e.report('timeout')['route_completed'])
        self.assertFalse(e.report('timeout')['strict_success'])

    def test_loop_projection_loss_freezes_progress_and_preserves_real_violations(self):
        e = EpisodeEvaluator([[0., 0.], [100., 0.], [100., 100.], [0., 100.], [0., 0.]])
        observe(e, 0, state(0.))
        observe(e, 1, state(.2))
        before = e.progress_s
        lost = state(.3); lost['position'] = [80., 70.]
        result = observe(e, 2, lost, events=[{'kind': 'red_light', 'frame': 102, 'id': '38'}])
        self.assertFalse(result['route_projection']['valid'])
        self.assertEqual(e.progress_s, before)
        self.assertIsNone(result['route_completion'])
        self.assertIsNone(result['progress_m'])
        self.assertIn('red_light', result['violations'])
        self.assertTrue(result['infrastructure_valid'])
        self.assertFalse(result['route_measurement_valid'])
        self.assertFalse(result['arrival'])

    def test_parallel_return_leg_cannot_relocate_to_another_loop_segment(self):
        e = EpisodeEvaluator([[0., 0.], [50., 0.], [50., 3.5], [0., 3.5]])
        observe(e, 0, state(0.))
        ego = state(.1); ego['position'][1] = 3.5
        result = observe(e, 1, ego)
        self.assertLess(e.progress_s, 1.)
        self.assertFalse(result['arrival'])

    def test_projection_can_correct_backward_within_supported_local_branch(self):
        e = self.evaluator()
        observe(e, 0, state(2.))
        observe(e, 1, state(1.9))
        self.assertAlmostEqual(e.progress_s, 1.9)
        self.assertTrue(e.report()['route_projection']['valid'])

    def test_destination_lane_and_original_three_metre_margin_are_required(self):
        e = EpisodeEvaluator([[0., 0.], [4., 0.]], settings={'arrival_lane_id': '1|0|-1'})
        wrong = state(2.); wrong['lane_id'] = '1|0|-2'
        self.assertFalse(observe(e, 0, wrong)['arrival'])
        outside = state(.9)
        self.assertFalse(observe(e, 1, outside)['arrival'])
        self.assertTrue(observe(e, 2, state(2.))['arrival'])

    def test_destination_heading_must_match_original_route(self):
        e = EpisodeEvaluator([[0., 0.], [4., 0.]], settings={'arrival_lane_id': '1|0|-1'})
        ego = state(2.);ego['heading'] = math.pi
        result = observe(e, 0, ego)
        self.assertFalse(result['arrival'])
        self.assertGreater(result['arrival_evidence']['heading_error_rad'], math.pi/3)

    def test_control_execution_is_bound_to_the_completed_physical_observation(self):
        output_root = Path(__file__).resolve().parents[1]/'outputs'
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='verify_actuation_', dir=str(output_root)) as directory:
            e = self.evaluator(directory=directory)
            evidence = {'source_frame': 95, 'applied_tick': 0, 'observed_tick': 1,
                        'planned_acceleration_mps2': -4., 'actual_actuation': {'throttle': 0., 'brake': .7}}
            e.observe(0, 100, 123., state(), [], road(), scenario_events={'scenario_valid': True},
                      control_execution=evidence)
            row = json.loads((Path(directory)/'physical_evaluation.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(row['control_execution'], evidence)
            self.assertEqual(row['frame'], 100)

    def test_boundary_links_come_from_native_directed_nearby_topology(self):
        calls=[]
        def waypoint(road,yaw=0.,z=0.):
            return NS(road_id=road,section_id=0,lane_id=2,lane_width=3.5,lane_type='Driving',
                transform=NS(rotation=NS(yaw=yaw),location=NS(x=0.,y=0.,z=z)))
        start=waypoint(2);before=waypoint(676);reversed_lane=waypoint(9,180.);overpass=waypoint(10,z=5.)
        start.previous=lambda distance:calls.append(('previous',distance)) or [before,reversed_lane,overpass]
        start.next=lambda distance:calls.append(('next',distance)) or [waypoint(2)]
        links=_boundary_topology([start],[{'index':0}])
        self.assertEqual(calls,[('previous',3.),('next',3.)])
        self.assertIn('676|0|2',links[0])
        self.assertNotIn('9|0|2',links[0]);self.assertNotIn('10|0|2',links[0])

    def test_final_frame_collision_received_after_arrival_is_retained(self):
        e = EpisodeEvaluator([[0., 0.], [4., 0.]])
        observe(e, 0, state(0.))
        observe(e, 1, state(2.))
        self.assertTrue(e.report('arrival')['strict_success'])
        event = {'kind': 'collision', 'frame': 101, 'other_id': 9}
        e.ingest_events([event, event])
        final = e.report('arrival')
        self.assertTrue(final['evaluation_valid'])
        self.assertTrue(final['route_completed'])
        self.assertFalse(final['strict_success'])
        self.assertEqual(final['driving_outcome'], 'collision')
        self.assertEqual(len(final['events']), 1)

    def test_warmup_collision_not_a_driving_collision(self):
        e = self.evaluator()
        e.ingest_events([{'kind': 'collision', 'frame': 95, 'other_id': 2}])
        observe(e, 0)
        self.assertFalse(e.report()['collision'])
        self.assertEqual(len(e.report()['initialization_events']), 1)
        self.assertFalse(e.report()['infrastructure_valid'])

    def test_late_warmup_or_source_frame_collision_revokes_success_without_driving_collision(self):
        for frame in (95, 100):
            e = EpisodeEvaluator([[0., 0.], [4., 0.]])
            observe(e, 0, state(0.))
            observe(e, 1, state(2.))
            self.assertTrue(e.report('arrival')['strict_success'])
            event = {'kind': 'collision', 'frame': frame, 'other_id': 9}
            e.ingest_events([event, event])
            e.ingest_events([event])
            result = e.report('arrival')
            self.assertTrue(result['route_completed'])
            self.assertFalse(result['collision'])
            self.assertFalse(result['strict_success'])
            self.assertFalse(result['infrastructure_valid'])
            self.assertEqual(len(result['initialization_events']), 1)
            self.assertEqual([row['kind'] for row in result['infrastructure_errors']], ['initialization_collision'])

    def test_late_event_keeps_occurrence_frame(self):
        e = self.evaluator()
        for tick in range(11):
            observe(e, tick)
        e.ingest_events([{'kind': 'collision', 'frame': 103, 'other_id': 3}])
        row = e.report()['events'][0]
        self.assertEqual(row['physical_tick'], 3)
        self.assertEqual(row['late_by_ticks'], 7)

    def test_frame_gap_invalidates_evidence_without_erasing_collision(self):
        e = self.evaluator()
        observe(e, 0)
        observe(e, 1, events=[{'kind': 'collision', 'frame': 101, 'other_id': 1}])
        with self.assertRaises(ValueError):
            observe(e, 3)
        self.assertTrue(e.report()['collision'])
        self.assertFalse(e.report()['evaluation_valid'])

    def test_changed_physics_clock_cannot_be_counted_as_50hz(self):
        e = self.evaluator()
        observe(e, 0)
        with self.assertRaises(ValueError):
            e.observe(1, 101, 123.04, state(), [], road())
        self.assertFalse(e.report()['infrastructure_valid'])
        self.assertEqual(e.report()['infrastructure_errors'][0]['kind'], 'physical_clock_step_changed')

    def test_scenario_transitions_preserve_trigger_and_completion_frames(self):
        output_root = Path(__file__).resolve().parents[1]/'outputs'
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='verify_events_', dir=str(output_root)) as directory:
            e = self.evaluator(directory=directory)
            row = {'id': 'crossing', 'required': True, 'state': 'active', 'trigger_tick': 0,
                   'action_start_tick': 1, 'completed_tick': None}
            for tick in range(3):
                if tick == 2:
                    row.update(state='completed', completed_tick=2)
                observe(e, tick, scenario={'scenario_valid': True, 'events': [row]})
            rows = [json.loads(line) for line in (Path(directory)/'scenario_events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['frame'] for r in rows], [100, 102])
            self.assertEqual(e.report()['scenario_events'][0]['completed_tick'], 2)

    def test_evaluation_reads_actors_outside_mpc_subset(self):
        e = self.evaluator()
        actors = [state(50+i, 10., str(i)) for i in range(12)]
        actors.append(state(14., 10., 'not_in_eight_representatives'))
        observe(e, 0, state(0., 10.), actors=actors)
        self.assertEqual(e.report()['minimum_net_gap_m'], 10.)
        self.assertEqual(e.report()['minimum_time_headway_s'], 1.)

    def test_unresolved_rule_evidence_cannot_certify_success(self):
        e = EpisodeEvaluator([[0., 0.], [4., 0.]])
        r = road()
        r['coverage']['public_priority'] = False
        observe(e, 0, state(2.), measured_road=r)
        self.assertTrue(e.report()['route_completed'])
        self.assertFalse(e.report()['evaluation_valid'])
        self.assertFalse(e.report()['strict_success'])

    def test_unreached_required_event_is_method_outcome_not_environment_failure(self):
        e = EpisodeEvaluator([[0., 0.], [4., 0.]])
        observe(e, 0, state(2.), scenario={'scenario_valid': True,
            'events': [{'id': 'cut_in', 'required': True, 'state': 'waiting'}]})
        result = e.report()
        self.assertTrue(result['evaluation_valid'])
        self.assertFalse(result['strict_success'])
        self.assertEqual(result['required_events_incomplete'], ['cut_in'])

    def test_goal_without_required_maneuver_is_not_strict_success(self):
        e = EpisodeEvaluator([[0., 0.], [4., 0.]])
        observe(e, 0, state(2.), scenario={'scenario_valid': True, 'events': [],
            'task_criteria': {'completed': False, 'kind': 'overtake_return',
                'route_checkpoints_completed': 4, 'route_checkpoints_total': 4}})
        result = e.report('arrival')
        self.assertTrue(result['route_completed'])
        self.assertTrue(result['infrastructure_valid'])
        self.assertTrue(result['measurement_complete'])
        self.assertFalse(result['strict_success'])
        self.assertEqual(result['strict_success_status'], 'task_incomplete')
        self.assertEqual(result['driving_outcome'], 'goal_reached_task_incomplete')

    def test_endpoint_projection_cannot_replace_ordered_route_gates(self):
        e = EpisodeEvaluator([[0., 0.], [4., 0.]])
        observe(e, 0, state(2.), scenario={'scenario_valid': True, 'events': [],
            'task_criteria': {'completed': False, 'kind': 'roundabout',
                'route_checkpoints_completed': 1, 'route_checkpoints_total': 4}})
        result = e.report('arrival')
        self.assertTrue(result['arrival'])
        self.assertFalse(result['route_completed'])
        self.assertFalse(result['strict_success'])
        self.assertTrue(result['infrastructure_valid'])

    def test_mixed_line_marking_keeps_unknown_not_false_violation(self):
        e = self.evaluator()
        observe(e, 0)
        observe(e, 1, events=[{'kind': 'lane_invasion', 'frame': 101, 'markings': ['SolidBroken']}])
        result = e.report()
        self.assertEqual(result['violations'], [])
        self.assertFalse(result['evaluation_valid'])
        self.assertTrue(result['infrastructure_valid'])
        self.assertFalse(result['measurement_complete'])
        self.assertEqual(len(result['unknown_rule_events']), 1)

    def test_mixed_line_permission_uses_crossing_side_not_label_alone(self):
        for permitted in (True, False):
            e = self.evaluator()
            observe(e, 0)
            observe(e, 1, events=[{'kind': 'lane_invasion', 'frame': 101,
                'markings': ['SolidBroken'], 'crossing_permission': permitted}])
            report = e.report()
            self.assertTrue(report['evaluation_valid'])
            self.assertEqual('lane_invasion' in report['violations'], not permitted)

    def test_legal_wait_not_stall_but_permanent_standstill_is(self):
        e = self.evaluator(settings={'stall_seconds': .1})
        for tick in range(10):
            r = road()
            r['expected_wait'] = True
            observe(e, tick, state(0., 0.), measured_road=r)
        self.assertFalse(e.report()['stalled'])
        for tick in range(10, 16):
            observe(e, tick, state(0., 0.))
        self.assertTrue(e.report()['stalled'])

    def test_report_is_json_and_cleanup_error_preserves_driving_result(self):
        output_root = Path(__file__).resolve().parents[1]/'outputs'
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='verify_evaluation_', dir=str(output_root)) as directory:
            e = self.evaluator(directory=directory)
            observe(e, 0)
            observe(e, 1, events=[{'kind': 'collision', 'frame': 101, 'other_id': 1}])
            e.infrastructure_error('cleanup_failure')
            e.save_report('runtime_error')
            result = json.loads((Path(directory)/'evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(result['driving_outcome'], 'collision')
            self.assertEqual(result['runtime_termination_reason'], 'runtime_error')
            self.assertFalse(result['evaluation_valid'])


class MutatingLocation:
    def __init__(self, x=0., y=0., z=0.):
        self.x, self.y, self.z = x, y, z


class MutatingTransform:
    location = NS(x=10., y=20., z=0.)
    rotation = NS(yaw=0.)
    def transform(self, point):
        point.x += self.location.x
        point.y += self.location.y
        return point


class ProbeTests(unittest.TestCase):
    def test_native_same_road_cross_section_supports_seven_metre_legal_lateral_offset(self):
        def waypoint(lane,y):
            return NS(road_id=1,section_id=0,lane_id=lane,lane_width=3.5,lane_type='Driving',
                transform=NS(rotation=NS(yaw=0.),location=NS(x=0.,y=-y,z=0.)))
        a,b,c=waypoint(1,0.),waypoint(2,3.5),waypoint(3,7.)
        a.get_left_lane=lambda:b;a.get_right_lane=lambda:None
        b.get_left_lane=lambda:c;b.get_right_lane=lambda:a
        c.get_left_lane=lambda:None;c.get_right_lane=lambda:b
        for identity,y,heading,expected in (('1|0|3',7.,0.,True),('9|0|3',7.,0.,False),
                                            ('1|0|3',20.,0.,False),('1|0|3',7.,math.pi,False)):
            probe=CarlaEvaluationProbe.__new__(CarlaEvaluationProbe)
            probe.evaluator=EpisodeEvaluator([[0.,0.],[100.,0.]])
            probe._route_waypoints=[a,a];probe._projection_corridors={}
            probe.evaluator.projection_metadata=probe._projection_corridor
            ego=state(0.);ego.update(lane_id=identity,position=[0.,y],heading=heading)
            result=observe(probe.evaluator,0,ego)
            with self.subTest(lane=identity,y=y,heading=heading):
                self.assertEqual(result['route_projection']['valid'],expected)
                self.assertFalse(result['arrival'])
                if expected:
                    self.assertGreater(result['route_projection']['max_distance_m'],7.)
                    self.assertAlmostEqual(probe.evaluator.progress_s,0.)
                    probe._route_boundary_evidence=[];probe._route_boundary_published=False
                    evidence=probe._pop_route_map_evidence(0)
                    self.assertEqual(evidence['version'],'carla-route-map-evidence-1.0')
                    self.assertIn('boundary_gates',evidence)
                    native=evidence['corridors']['0']
                    self.assertEqual(native['route_waypoint']['position'],[0.,0.])
                    self.assertEqual(native['neighbour_links'][1]['to_waypoint']['position'],[0.,7.])
                    self.assertIsNone(probe._pop_route_map_evidence(1))
        # Numeric IDs cannot invent a connection or widen an independent road.
        a.get_left_lane=lambda:None
        probe._projection_corridors={}
        ego.update(lane_id='1|0|3',position=[0.,7.],heading=0.)
        self.assertFalse(probe._projection_corridor(0,ego,0.)['contains_observed_center'])

    def test_priority_actor_identity_survives_observed_junction_lane_transition(self):
        p = CarlaEvaluationProbe.__new__(CarlaEvaluationProbe)
        p.evaluator = EpisodeEvaluator([[0., 0.], [100., 0.]])
        p._lights, p._stops, p._stop_regions = [], [], []
        p.public_rules = [{'id': 'yield', 'kind': 'yield', 'conflict_route_s_m': [10., 12.],
                           'priority_lane_ids': ['approach']}]
        p._conflict_polygons = {'yield': p._zone_polygons({'route_s_m': [10., 12.]})}
        p._priority_seen, p._priority_violated = {}, set()
        p._previous_ego = None
        other = state(11., 2., 'priority_car')
        other['lane_id'] = 'approach'
        other['position'][1] = 6.
        self.assertEqual(p._rules(state(0.), [other], 100, {}), [])
        other['lane_id'] = 'junction_successor'
        other['position'][1] = 0.
        rows = p._rules(state(9.), [other], 101, {})
        self.assertEqual(rows[0]['kind'], 'yield_violation')

    def test_side_evidence_is_bound_to_occurrence_frame(self):
        p = CarlaEvaluationProbe.__new__(CarlaEvaluationProbe)
        p.carla = NS(Location=MutatingLocation, LaneType=NS(Driving=1), LaneChange=NS(Left=2, Right=1))
        wp = NS(road_id=1, section_id=0, lane_id=-1, is_junction=False, lane_width=3.5, lane_change=3,
                transform=NS(location=NS(x=0., y=0.), rotation=NS(yaw=0.)),
                left_lane_marking=NS(type='SolidBroken', lane_change=2),
                right_lane_marking=NS(type='Broken', lane_change=1))
        p.map = NS(get_waypoint=lambda *a, **kw: wp)
        p._previous_waypoint = wp
        p._previous_ego = state()
        p._previous_ego['position'][1] = .74
        p._crossing_evidence = {}
        current = state()
        current['position'][1] = .76
        p._record_crossing_evidence(current, 101)
        events = p._annotate_events([{'kind': 'lane_invasion', 'frame': 101, 'markings': ['SolidBroken']}])
        self.assertTrue(events[0]['crossing_permission'])
        self.assertEqual(events[0]['crossing_evidence']['side'], 'left')
        unaligned = p._annotate_events([{'kind': 'lane_invasion', 'frame': 102, 'markings': ['SolidBroken']}])
        self.assertNotIn('crossing_permission', unaligned[0])

    def test_pet_uses_actual_box_zone_occupancy_and_exit_time(self):
        p = CarlaEvaluationProbe.__new__(CarlaEvaluationProbe)
        p.evaluator = EpisodeEvaluator([[0., 0.], [100., 0.]])
        p._conflict_zones = [{'id': 'crossing', 'route_s_m': [10., 12.], 'half_width_m': 2.}]
        p._conflict_polygons = {'crossing': p._zone_polygons(p._conflict_zones[0])}
        p._zone_presence, p._zone_exits, p._pet, p._pet_keys = {}, {}, [], set()
        other = state(11., 0., 'crossing_actor')
        p._measure_conflict_zones(state(0.), [other], 0, 100)
        other['position'][1] = 6.
        p._measure_conflict_zones(state(0.), [other], 10, 110)
        p._measure_conflict_zones(state(9.), [other], 35, 135)
        self.assertEqual(len(p._pet), 1)
        self.assertAlmostEqual(p._pet[0]['pet_s'], .5)
        self.assertFalse(p._pet[0]['overlapping_occupancy'])

    def test_snapshot_bbox_transform_does_not_mutate_sdk_metadata(self):
        probe = CarlaEvaluationProbe.__new__(CarlaEvaluationProbe)
        probe.carla = NS(Location=MutatingLocation, LaneType=NS(Driving=1))
        probe.map = NS(get_waypoint=lambda *args, **kw: None)
        probe._omega_history = {}
        box = NS(location=MutatingLocation(2., 1., .8), rotation=NS(yaw=0.), extent=NS(x=2., y=1.))
        actor = NS(id=1, bounding_box=box, type_id='vehicle.test')
        source = NS(get_transform=lambda: MutatingTransform(),
                    get_velocity=lambda: NS(x=1., y=2.),
                    get_acceleration=lambda: NS(x=0., y=0.),
                    get_angular_velocity=lambda: NS(z=180./math.pi))
        snapshot = NS(find=lambda identity: source, timestamp=NS(elapsed_seconds=1.))
        first = probe._state(actor, snapshot)
        second = probe._state(actor, snapshot)
        self.assertEqual((box.location.x, box.location.y, box.location.z), (2., 1., .8))
        self.assertEqual(first['position'], second['position'])
        self.assertEqual(first['position'], [12., -21.])
        self.assertAlmostEqual(first['velocity'][0], 0.)
        self.assertAlmostEqual(first['velocity'][1], -4.)
        self.assertEqual(first['acceleration'], [-2., 1.])

    def test_red_light_is_finite_directed_stop_line_crossing(self):
        p = CarlaEvaluationProbe.__new__(CarlaEvaluationProbe)
        p.evaluator = EpisodeEvaluator([[0., 0.], [100., 0.]])
        p._stops, p._stop_regions, p.public_rules = [], [], []
        wp = NS(transform=NS(rotation=NS(yaw=0.), location=NS(x=10., y=0., z=0.)), lane_width=3.5,
                road_id=1, section_id=0, lane_id=-1)
        p._lights = [(NS(id=7, get_state=lambda: 'Red'), wp)]
        p._previous_ego = state(7.9)
        events = p._rules(state(8.1), [], 5, {})
        self.assertEqual(events[0]['kind'], 'red_light')
        self.assertEqual(events[0]['crossing_evidence']['controlled_lane_id'], '1|0|-1')
        self.assertGreater(events[0]['crossing_evidence']['front_signed_distance_m'], 0.)
        p._previous_ego = state(8.1)
        self.assertEqual(p._rules(state(8.2), [], 6, {}), [])
        previous, current = state(7.9), state(8.1)
        previous['position'][1] = current['position'][1] = 20.
        p._previous_ego = previous
        self.assertEqual(p._rules(current, [], 7, {}), [])

    def test_stop_volume_requires_real_hold_before_exit(self):
        def setup():
            p = CarlaEvaluationProbe.__new__(CarlaEvaluationProbe)
            p.evaluator = EpisodeEvaluator([[0., 0.], [100., 0.]])
            p._stops, p._lights, p.public_rules = [], [], []
            p._stop_regions = [{'id': 'stop', 'center': [10., 0.], 'z': 0., 'half': [2., 2., 1.], 'heading': 0.}]
            p._stop_last_stopped, p._stop_dwell, p._region_inside = {}, {}, {}
            p._previous_ego = None
            return p
        p = setup()
        p._rules(state(7., 2.), [], 1, {})
        rows = p._rules(state(11., 2.), [], 2, {})
        self.assertEqual(rows[0]['kind'], 'stop_sign')
        p = setup()
        for frame in range(51):
            self.assertEqual(p._rules(state(7., 0.), [], frame, {}), [])
        self.assertEqual(p._rules(state(11., 2.), [], 51, {}), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
