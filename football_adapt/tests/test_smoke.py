"""Minimal smoke tests. Run from the project root:  python -m pytest -q"""
import numpy as np
from football import FootballEnv, load_config
from football.geometry import to_frame, closest_point_on_segment
from football.mechanics import pass_interception, tackle_probability
from agents import RandomAgent


def test_frame_conversion_is_self_inverse():
    p = np.array([30.0, 10.0])
    assert np.allclose(to_frame(1, to_frame(1, p, 100, 60), 100, 60), p)
    assert np.allclose(to_frame(0, p, 100, 60), p)


def test_closest_point_on_segment():
    q = closest_point_on_segment(np.array([5.0, 3.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0]))
    assert np.allclose(q, [5.0, 0.0])


def test_defender_on_the_pass_line_intercepts():
    res = pass_interception([0, 0], [20, 0], [("d", np.array([10.0, 0.0]))], 3.0, 1.0, 0.5)
    assert res is not None and res[0] == "d"


def test_far_defender_does_not_intercept():
    res = pass_interception([0, 0], [20, 0], [("d", np.array([10.0, 30.0]))], 3.0, 1.0, 0.5)
    assert res is None


def test_equal_ratings_give_even_tackle():
    assert abs(tackle_probability(0.8, 0.8) - 0.5) < 1e-9


def test_episode_runs_and_is_reproducible():
    cfg = load_config(overrides={"time.T": 100})
    env = FootballEnv(cfg)
    outs = []
    for _ in range(2):
        obs, _ = env.reset(seed=5)
        ag = RandomAgent(1)
        done = False
        while not done:
            obs, r, done, info = env.step(ag.act(obs))
        outs.append((info["final_score"], len(env.event_log)))
    assert outs[0] == outs[1]
    assert obs["mask"].shape == (5, env.space.n)


def test_opponent_cannot_see_beyond_radius():
    cfg = load_config(overrides={"vision.radius": 5.0})
    env = FootballEnv(cfg)
    env.reset(seed=0)
    from football.observation import build_observation
    o = build_observation(env, 1, 0)
    for j in range(5):
        if o.opp_visible[j]:
            assert np.linalg.norm(o.opp_pos[j] - o.pos) <= 5.0 + 1e-6
