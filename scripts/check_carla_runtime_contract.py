"""Regression checks for source-bound integer control and observation clocks; no CARLA server."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nine_grid.carla_execution import ControlTimeline, SignedAccelerationController
from nine_grid.carla_bridge import CarlaSession


class RuntimeContract(unittest.TestCase):
    def test_braking_intent_cannot_leave_positive_throttle(self):
        controller = SignedAccelerationController(1.2)
        accelerating = controller.step(0, 2.1, 0., 2.482, 2.44, 0.)
        self.assertGreater(accelerating['throttle'], 0.)
        braking = controller.step(1, -4.4, 0., 2.352, 2.44, .1)
        self.assertEqual(braking['throttle'], 0.)
        self.assertGreaterEqual(braking['brake'], 4.4/8.)
        self.assertLess(braking['requested_acceleration_mps2'], 0.)

    def test_stop_hold_releases_immediately_for_positive_acceleration(self):
        controller = SignedAccelerationController(1.2)
        stopped = controller.step(0, -1., 0., 0., 0., 0.)
        self.assertTrue(stopped['stop_hold'])
        self.assertEqual(stopped['throttle'], 0.)
        self.assertGreaterEqual(stopped['brake'], .5)
        restart = controller.step(1, 2.1, 0., .042, 0., 0.)
        self.assertFalse(restart['stop_hold'])
        self.assertGreaterEqual(restart['throttle'], 2.1/6.)
        self.assertEqual(restart['brake'], 0.)

    def test_feedback_is_idempotent_per_tick_and_uses_real_wheel_limit(self):
        controller = SignedAccelerationController(1.2)
        first = controller.step(0, 2., .6, .04, 0., 0.)
        duplicate = controller.step(0, 2., .6, .04, 0., 0.)
        self.assertEqual(first, duplicate)
        self.assertAlmostEqual(first['steering_rad_left'], .8*.02)
        self.assertAlmostEqual(first['steer_normalized_right'], -(.8*.02)/1.2)
        self.assertLessEqual(first['throttle'], 1.)
        self.assertLessEqual(first['brake'], 1.)

    def test_same_tick_new_plan_replaces_command_without_double_integral(self):
        controller = SignedAccelerationController(1.2)
        controller.step(0, 1., .1, .02, 0., 0.)
        replaced = controller.step(0, -1., -.1, 0., 0., 0.)
        fresh = SignedAccelerationController(1.2).step(0, -1., -.1, 0., 0., 0.)
        self.assertEqual(replaced, fresh)
        self.assertEqual(replaced['throttle'], 0.)
        self.assertGreater(replaced['brake'], 0.)

    def test_plan_updates_keep_bounded_feedback_state_without_integrating_ghost_speed(self):
        controller = SignedAccelerationController(1.2)
        previous_integral = 0.
        for tick in range(10):
            result = controller.step(tick, .5, 0., .01, 0., 0.)
            self.assertGreater(result['integral_mps2'], previous_integral)
            self.assertLessEqual(result['integral_mps2'], 1.)
            # 同意图的新源计划仍以真实0速度为初值；执行前馈不会随小v误差消失。
            self.assertEqual(result['target_speed_mps'], .01)
            self.assertGreater(result['throttle'], .5/6.)
            previous_integral = result['integral_mps2']

    def test_ten_hz_source_speed_resets_do_not_starve_start_or_braking(self):
        # 执行映射的理想踏板模型，仅检查闭环符号/起步；不代表真实CARLA标定。
        controller = SignedAccelerationController(1.2)
        timeline = ControlTimeline(physics_hz=50)
        speed, observed_acceleration = 0., 0.
        for tick in range(250):
            planned_acceleration = 2.1 if tick < 150 else -4.4
            if tick % 5 == 0:
                timeline.install([[planned_acceleration, 0.]]*30, tick*.02,
                    .1, speed, source_tick=tick)
            state = timeline.at(tick*.02, tick)
            self.assertEqual(state['index'], 0)
            elapsed = .1-state['remaining_segment_s']
            target = max(0., state['segment_start_speed_mps']+planned_acceleration*(elapsed+.02))
            control = controller.step(tick, planned_acceleration, 0., target, speed, observed_acceleration)
            if tick >= 150:
                self.assertEqual(control['throttle'], 0.)
            acceleration = 6.*control['throttle']-8.*control['brake']-.15
            next_speed = max(0., speed+acceleration*.02)
            observed_acceleration, speed = (next_speed-speed)/.02, next_speed
            if tick == 149:
                self.assertGreater(speed, 5.)
        self.assertLess(speed, .1)

    def test_bridge_records_actual_tick_response_and_original_source(self):
        session = CarlaSession()
        session.physics_hz, session.control_step_s = 50, .1
        session.timeline = ControlTimeline(physics_hz=50)
        session._initial_source_frame, session.frame = 100, 107
        session._sim_time = .14
        session._runtime_metrics['physical_ticks'] = 7
        session._published_sources = {100: (0., 0)}
        actual = SimpleNamespace(throttle=.2, brake=0., steer=0., hand_brake=False, reverse=False, gear=1)
        commands = []
        vector = SimpleNamespace(x=1., y=0., z=0.)
        session.ego = SimpleNamespace(get_velocity=lambda: SimpleNamespace(x=2., y=0., z=0.),
            get_acceleration=lambda: SimpleNamespace(x=0., y=0., z=0.),
            get_transform=lambda: SimpleNamespace(get_forward_vector=lambda: vector),
            get_control=lambda: actual, apply_control=commands.append)
        session.carla = SimpleNamespace(VehicleControl=lambda **kwargs: kwargs)
        session._execution_controller = SignedAccelerationController(1.2)
        session._show_plan = lambda plan: {}
        controls = [[2., 0.], [-4., .1]]+[[-3., 0.]]*28
        session._install(controls, .1, 0., None, 2., 100)
        self.assertTrue(session._apply_current())
        self.assertTrue(session._apply_current())
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]['throttle'], 0.)
        self.assertGreater(commands[0]['brake'], 0.)
        session.frame, session._sim_time = 108, .16
        session._runtime_metrics['physical_ticks'] = 8
        response = session._read_control_response()
        self.assertEqual(response['source_frame'], 100)
        self.assertEqual(response['source_tick'], 0)
        self.assertEqual(response['applied_tick'], 7)
        self.assertEqual(response['observed_tick'], 8)
        self.assertEqual(response['sequence_index'], 1)
        # 实际SDK读回必须原样记录，不能拿requested踏板伪装执行结果。
        self.assertEqual(response['actual_actuation']['throttle'], .2)
        self.assertEqual(response['actual_actuation']['brake'], 0.)
        self.assertFalse(response['pedal_readback_matches_request'])
        self.assertTrue(response['braking_intent_with_actual_throttle'])

    def test_float32_world_time_never_changes_segment(self):
        for planner_hz in (1, 2, 10):
            timeline = ControlTimeline(physics_hz=50)
            period = 50//planner_hz
            for tick in range(151):
                if tick % period == 0:
                    timeline.install([[0., .001*i] for i in range(30)],
                        tick*.019999999552965164, .1, 5., source_tick=tick)
                state = timeline.at(tick*.019999999552965164, tick)
                self.assertEqual(state['index'], (tick % period)//5)
                self.assertGreater(state['remaining_segment_s'], 0.)

    def test_exact_expiration_and_async_partial_segment(self):
        timeline = ControlTimeline(physics_hz=50)
        timeline.install([[0., 0.]]*30, 0., .1, 2., source_tick=0)
        self.assertEqual(timeline.at(.499999988824, 25)['index'], 5)
        self.assertEqual(timeline.at(.519999988, 26)['index'], 5)
        self.assertAlmostEqual(timeline.at(.519999988, 26)['remaining_segment_s'], .08)
        self.assertTrue(timeline.at(2.97999993, 149)['available'])
        self.assertEqual(timeline.at(2.99999993, 150)['reason'], 'sequence_exhausted')
        with self.assertRaises(ValueError):
            timeline.at(.5)

    def test_rejected_plan_does_not_replace_running_sequence(self):
        session = CarlaSession()
        session.physics_hz, session.control_step_s = 50, .1
        session.timeline = ControlTimeline(physics_hz=50)
        session._published_sources = {100: (0., 0), 105: (.0999999978, 5)}
        session._install([[0., 0.]]*30, .1, 0., None, 2., 100)
        original = session.timeline
        with self.assertRaises(ValueError):
            session._install([[0., 0.]]*30, .1, .1, None, 2., 100)
        with self.assertRaises(ValueError):
            session._install([[0., 0.]]*30, .1, 0., None, 2., 999)
        self.assertIs(session.timeline, original)
        session._sim_time = .1399999969
        session._runtime_metrics['physical_ticks'] = 7
        session._install([[0., 0.]]*30, .1, .0999999978, None, 2., 105)
        self.assertEqual(session.timeline.source_tick, 5)
        self.assertAlmostEqual(session.timeline.at(session._sim_time, 7)['remaining_segment_s'], .06)

    def test_history_uses_ticks_even_with_large_raw_timestamp(self):
        session = CarlaSession()
        session.physics_hz, session.t0 = 50, 1685.158112
        state = {'id': 'ego', 'kind': 'vehicle', 'position': [0., 0.],
                 'velocity': [1., 0.], 'acceleration': [0., 0.], 'heading': 0.}
        session._entities = lambda: (state, [], 0)
        for tick in range(51):
            session.frame = 400000+tick
            session._runtime_metrics['physical_ticks'] = tick
            session.world = SimpleNamespace(get_snapshot=lambda tick=tick: SimpleNamespace(
                timestamp=SimpleNamespace(elapsed_seconds=session.t0+tick*.019999999552965164)))
            session._sample_history()
        self.assertEqual([row['physical_tick'] for row in session._history['ego']], [0,25,50])


if __name__ == '__main__':
    unittest.main(verbosity=2)
