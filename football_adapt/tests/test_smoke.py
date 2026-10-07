"""Minimal smoke tests. Run from the project root:  python -m pytest -q"""
import numpy as np
from football import FootballEnv, load_config, apply_style
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


def test_normal_style_is_the_plain_config():
    cfg = load_config()
    assert apply_style(cfg, "normal").to_dict() == cfg.to_dict()


def test_styles_change_how_a_team_plays_but_not_the_rules():
    cfg = load_config()
    env = FootballEnv(cfg)
    line = {}                                                  # depth of the first defender in the defending shape
    for name in ("aggressive", "normal", "defensive"):
        s = apply_style(cfg, name)
        for rules in ("pitch", "time", "speeds", "vision", "mechanics", "control", "reward"):
            assert s[rules].to_dict() == cfg[rules].to_dict()
        line[name] = env.book.anchors_team_frame("2-2-1", -1.0, s.formations.phases)[0][0]
    assert line["aggressive"] > line["normal"] > line["defensive"]
    for bad in ("no_such_style",):
        try:
            apply_style(cfg, bad)
            assert False, "an unknown style must be rejected"
        except KeyError:
            pass


def test_styled_match_is_reproducible_and_style_can_change_mid_match():
    cfg = load_config(overrides={"time.T": 150})
    env = FootballEnv(cfg, scripted_ours=True)
    outs = []
    for _ in range(2):
        env.reset(seed=3, our_style="aggressive", opp_style="defensive")
        info = {}
        while not env.done:
            if env.t == 75:
                env.set_style(1, "normal")
            _, _, _, info = env.step()
        outs.append((info["final_score"], len(env.event_log), info["styles"]))
    assert outs[0] == outs[1]
    assert outs[0][2] == ("aggressive", "normal")


def test_every_formation_plays_and_can_change_mid_match():
    cfg = load_config(overrides={"time.T": 150})
    env = FootballEnv(cfg, scripted_ours=True)
    names = env.book.names
    assert len(names) == 5
    for i, g in enumerate(names):
        f = names[(i + 2) % len(names)]
        env.reset(seed=i, our_formation=f, opp_formation=g)
        assert env.formation == [f, g]
        nxt = names[(i + 1) % len(names)]
        info = {}
        while not env.done:
            if env.t == 60:
                before = env.pos[1].copy()
                env.set_formation(1, nxt)
                assert np.allclose(env.pos[1], before)         # nobody teleports: they walk to the new anchors
                assert env.controllers[1].in_transition
            _, _, _, info = env.step()
        assert env.formation == [f, nxt] and info["g"] == nxt
        assert [e for e in env.event_log if e["type"] == "switch"] == [
            {"type": "switch", "team": 1, "from_": g, "to": nxt, "t": 60}]
        assert sum(env.stats["passes"]) > 0


def test_aggressive_team_stands_in_the_middle_of_passing_lanes():
    cfg = load_config(overrides={"time.T": 300})
    env = FootballEnv(cfg, scripted_ours=True)
    env.reset(seed=3, our_style="aggressive", opp_style="normal")
    blue, red = env.controllers[0], env.controllers[1]
    checked = 0
    while not env.done:
        holder, ball, reds = env.holder, env.ball_pos.copy(), env.pos[1].copy()    # what the players see this step
        env.step()
        assert red._lanes == {}                                # the normal team does not cut lanes
        if holder is not None and holder[0] == 1 and blue._lanes:
            middles = [(ball + reds[k]) / 2 for k in range(5) if k != holder[1]]
            for point in blue._lanes.values():                 # team 0's frame is the world frame
                assert min(np.linalg.norm(point - m) for m in middles) < 1e-9
            assert len(blue._lanes) <= 4
            checked += 1
    assert checked > 20



def _goal_kick_match(seed=0):
    """Play until the first goal kick. Returns the env just after it, and the kicker (team, player)."""
    env = FootballEnv(load_config(), scripted_ours=True)
    env.reset(seed=seed)
    while not env.done:
        _, _, _, info = env.step()
        gk = [e for e in info["events"] if e["type"] == "goal_kick"]
        if gk:
            return env, gk[0]["by"]
    return env, None


def test_nobody_of_the_other_team_is_in_the_box_at_a_goal_kick():
    seen = 0
    for seed in range(4):
        env, kicker = _goal_kick_match(seed)
        if kicker is None:
            continue
        seen += 1
        team = kicker[0]
        assert env.goal_kick == kicker and env._in_box(team, env.ball_pos)
        while env.goal_kick is not None and not env.done:       # until the kick is taken they stay out
            assert not any(env._in_box(team, p) for p in env.pos[1 - team])
            env.step()
        assert env.holder != kicker or not env._in_box(team, env.ball_pos)
    assert seen > 0


def test_out_of_box_moves_a_player_to_the_nearest_edge():
    env = FootballEnv(load_config(), scripted_ours=True)
    env.reset(seed=0)
    assert np.allclose(env._out_of_box(0, np.array([10.0, 30.0])), [16.5, 30.0])     # front edge
    assert np.allclose(env._out_of_box(0, np.array([3.0, 50.0])), [3.0, 52.5])       # side edge
    assert np.allclose(env._out_of_box(1, np.array([95.0, 30.0])), [83.5, 30.0])     # the other goal
    assert np.allclose(env._out_of_box(0, np.array([40.0, 30.0])), [40.0, 30.0])     # outside: unchanged


def test_a_clearance_flies_upfield_over_everyone_and_lands_loose():
    from football.actions import Action, Kind
    env = FootballEnv(load_config(overrides={"mechanics.clear.spread": 0.0}))
    env.reset(seed=0, start_team=0)
    k = env.holder[1]
    env.pos[0][k] = np.array([15.0, 30.0])
    env.pos[1][:] = [[20.0 + 2 * j, 30.0] for j in range(5)]       # red players standing right on the ball's path
    env.controllers[1].act = lambda t, obs: {}                     # ... and not moving
    _, _, _, info = env.step({k: Action(Kind.CLEAR)})
    assert [e["type"] for e in info["events"]] == ["clear"] and env.stats["clearances"] == [1, 0]
    assert env.ball_state == "flight" and env.holder is None
    while env.ball_state == "flight":
        env.step({})
    assert np.allclose(env.ball_pos, [55.0, 30.0]) and env.holder is None and env.ball_state == "loose"


def test_a_pressed_defender_with_nobody_free_clears_the_ball():
    from football.actions import Kind
    from football.observation import build_observation
    env = FootballEnv(load_config(), scripted_ours=True)
    env.reset(seed=0, start_team=0)
    env.holder = (0, 0)                                            # blue defender on the ball, deep in his half
    env.pos[0][0] = np.array([20.0, 30.0])
    env.ball_pos = env.pos[0][0].copy()
    env.pos[1][0] = np.array([22.0, 30.0])                         # a red player on him ...
    for j in range(1, 5):                                          # ... and one on every teammate
        env.pos[1][j] = env.pos[0][j] + np.array([-1.5, 0.0])
    ctrl = env.controllers[0]
    a = ctrl._attack_with_ball(build_observation(env, 0, 0))
    assert a.kind == Kind.CLEAR and a.arg[0] > 20.0
    env.pos[1][0] = np.array([60.0, 30.0])                         # nobody on him: he does not clear
    assert ctrl._attack_with_ball(build_observation(env, 0, 0)).kind != Kind.CLEAR
