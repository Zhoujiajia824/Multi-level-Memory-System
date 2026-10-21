"""Complete constraints retain per-actor geometry when rectangles are batched."""
from dataclasses import replace

import numpy as np
import pytest

from carla_bridge.planning import mpc_geometry
from carla_bridge.planning.mpc_solver import MPCConfig, MPCInitialState, MPCObstacle, _Evaluation
from carla_bridge.tests.test_mpc_solver import straight


@pytest.mark.parametrize("steps", [3, 6, 20])
@pytest.mark.parametrize("body_offset_yaw", [(0., 0.), (.17, .12)])
def test_rectangle_batch_matches_per_actor_full_constraints_and_jacobians_exactly(
        monkeypatch, steps, body_offset_yaw):
    lateral, yaw = body_offset_yaw
    cfg = MPCConfig(horizon_steps=steps, ego_body_center_lateral=lateral,
                    ego_body_yaw=yaw, collision_margin=.27)
    problem = straight(cfg, initial=MPCInitialState(.1, -.2, .07, 2.1, .01, -.3),
                       target=4., width=4.)
    count = steps+1
    times = np.arange(count)*cfg.dt
    obstacle = MPCObstacle("first_rectangle", np.c_[9.-.3*times, 2.+.1*times],
        np.tile([2.1, .8], (count, 1)), .2+.03*times,
        rectangle_half_extents=[2.1, .8], uncertainty_m=.12+.02*times)
    # Interleaving legacy shapes makes row order part of the comparison.
    circle = MPCObstacle("middle_circle", np.tile([6., -2.], (count, 1)), .45)
    ellipse = MPCObstacle("middle_ellipse", np.tile([13., 3.], (count, 1)),
        np.tile([2.8, 1.2], (count, 1)), -.4)
    sampled = replace(obstacle, id="sampled_rectangle", centers=np.c_[-9.+times, -1.-.2*times],
        rectangle_half_extents=np.c_[2.+.01*times, .7+.02*times],
        uncertainty_m=.07+.03*times, yaws=-.1+.02*times)
    problem = replace(problem, obstacles=(obstacle, circle, ellipse, sampled))
    geometry = mpc_geometry.rectangle_signed_distance
    calls = []

    def one_batch(*args, **kwargs):
        calls.append((np.shape(args[0]), np.shape(args[1])))
        return geometry(*args, **kwargs)

    monkeypatch.setattr(mpc_geometry, "rectangle_signed_distance", one_batch)
    batched = _Evaluation(cfg, problem)
    rng = np.random.default_rng(27100+steps)
    commands = [np.c_[rng.uniform(-3.8, 1.8, steps),
                      rng.uniform(-.35, .35, steps)].ravel() for _ in range(8)]
    expected = []
    for control in commands:
        batched.evaluate(control)
        expected.append((batched.constraints.copy(), batched.jacobian.copy(),
                         batched.cost, batched.gradient.copy(), list(batched.constraint_names)))
    assert calls == [((1, 2*steps, 3), (2, 2*steps, 2))]*len(commands)

    def independent_per_actor(poses, centers, yaws, ego_half, other_half, **kwargs):
        result = [geometry(poses[0], centers[index], yaws[index], ego_half,
                           other_half[index], **kwargs) for index in range(len(centers))]
        return np.stack([item[0] for item in result]), np.stack([item[1] for item in result])

    monkeypatch.setattr(mpc_geometry, "rectangle_signed_distance", independent_per_actor)
    separate = _Evaluation(cfg, problem)
    for control, (values, jacobian, cost, gradient, names) in zip(commands, expected):
        separate.evaluate(control)
        np.testing.assert_array_equal(separate.constraints, values)
        np.testing.assert_array_equal(separate.jacobian, jacobian)
        np.testing.assert_array_equal(separate.gradient, gradient)
        assert separate.cost == cost
        assert separate.constraint_names == names
        names = [name for name in names if name.startswith("obstacle:")]
        assert names == (["obstacle:first_rectangle"]*(2*steps)
            + ["obstacle:middle_circle"]*(2*steps)
            + ["obstacle:middle_ellipse"]*(6*steps)
            + ["obstacle:sampled_rectangle"]*(2*steps))


def test_no_rectangle_problem_keeps_legacy_paths_without_geometry_batch(monkeypatch):
    cfg = MPCConfig(horizon_steps=3)
    problem = straight(cfg)
    circle = MPCObstacle("circle", np.tile([10., 2.], (4, 1)), .4)
    def unexpected_batch(*args, **kwargs):
        raise AssertionError("No rectangle helper needed for circular-only obstacles")
    monkeypatch.setattr(mpc_geometry, "rectangle_signed_distance", unexpected_batch)
    evaluation = _Evaluation(cfg, replace(problem, obstacles=(circle,)))
    evaluation.evaluate(np.zeros(6))
    assert evaluation.rectangles is None
    assert evaluation.constraint_names.count("obstacle:circle") == 6
