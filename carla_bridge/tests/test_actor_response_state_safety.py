"""A follower must not transfer an old response across a changed real leader."""
import numpy as np

from carla_bridge.planning.actor_motion import ActorMotionHistory, predict_actor_motion
from carla_bridge.tests.test_actor_interaction import Actor, Road


def test_old_braking_response_does_not_transfer_to_a_preexisting_more_distant_leader():
    history = ActorMotionHistory()
    follower, near, far = Actor(90,-9.56,speed=2.2),Actor(1,0.),Actor(2,6.)
    for now in np.arange(8)*.2:
        remaining = 1.4-now
        follower.x = -9.56-2.2*remaining-remaining**2
        follower.speed = 2.2+2.*remaining
        history.update(actors=(follower,near,far),sim_time_s=float(now))
    times = np.arange(21)*.2
    before = predict_actor_motion(follower,history.motion_for(follower),times,Road(),
        history=history,leader_actors=(near,far))[2]["following_response"]
    assert before["applied"] and before["leader_id"] == 1
    # The closest real front vehicle leaves. The farther vehicle was present
    # throughout the history, but the old braking event belonged to the old
    # leader relation; no new braking response has been observed since the switch.
    follower.x += follower.speed*.2
    history.update(actors=(follower,far),sim_time_s=1.6)
    after = predict_actor_motion(follower,history.motion_for(follower),times,Road(),
        history=history,leader_actors=(far,))[2]["following_response"]
    assert not after["applied"], after
