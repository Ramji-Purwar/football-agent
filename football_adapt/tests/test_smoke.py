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


TERMS = ("w_pass_done", "w_pass_accuracy", "w_receive", "w_free", "w_turnover", "w_progress", "w_chance", "w_shot",
         "w_big_chance", "w_poor_shot", "w_missed_shot", "w_chance_against", "w_shot_against", "w_tackle",
         "w_interception", "w_open_lanes", "w_formation")


def only(**on):
    """Reward overrides for a test of a few terms: every small term off but those given (in points), no cap, and
    everything to the team unless per_player is given."""
    ov = {f"reward.{t}": 0.0 for t in TERMS}
    ov.update({"reward.small_cap": 0.0, "reward.per_player": False})
    ov.update({f"reward.{k}": v for k, v in on.items()})
    return ov


def own_part(info):
    """Each player's small terms this step (reward units): his reward less the team's result and goals."""
    return info["reward_players"] - info["reward_parts"]["team"] - info["reward_parts"]["terminal"]


ONE_TERM = only()                         # every small term off


def _our_kickoff(overrides=None):
    """Our kick-off against a red team that stands still. Returns the env and our player on the ball."""
    env = FootballEnv(load_config(overrides=overrides))
    env.reset(seed=0, start_team=0)
    env.controllers[1].act = lambda t, obs: {}
    return env, env.holder[1]


def test_progress_is_paid_per_metre_and_going_back_takes_it_away():
    from football.actions import Action, Kind
    env, k = _our_kickoff(only(w_progress=0.05))
    w, v = env.cfg.reward.w_progress * env.cfg.reward.scale, env.cfg.speeds.dribble   # points -> reward
    fwd, back = (env.space.index_of(Action(Kind.MOVE_DIR, d)) for d in (0, 4))
    env.step({k: fwd})                                             # the first step with the ball sets the mark
    paid = [env.step({k: a})[3]["reward_parts"]["shaping"] for a in (fwd, fwd, back, back)]
    assert np.allclose(paid, [w * v, w * v, -w * v, -w * v]) and abs(sum(paid)) < 1e-12


def _parked_ball(overrides=None):
    """Blue has the ball in its own corner and does not move. Returns the env and the blue player on it."""
    env = FootballEnv(load_config(overrides=overrides))
    env.reset(seed=0, start_team=0)
    k = env.holder[1]
    env.pos[0][k] = np.array([3.0, 3.0])
    env.ball_pos = env.pos[0][k].copy()
    return env, k


def test_the_scripted_team_goes_and_gets_a_ball_that_is_parked_deep():
    env, k = _parked_ball()
    red, after = env.controllers[1], env.cfg.opponent.hunt.after
    lost_at = None
    for t in range(400):
        env.step({})
        if env.holder != (0, k):
            lost_at = t
            break
        assert red.hunting == (red.unpressed > after)
    assert lost_at is not None and lost_at > after                 # left alone at first, then hunted down
    assert env.holder is not None and env.holder[0] == 1           # ... and red has the ball


def test_pure_zonal_defence_leaves_a_parked_ball_alone():
    env, k = _parked_ball({"opponent.hunt.after": 0})
    for t in range(400):
        env.step({})
    assert env.holder == (0, k) and not env.controllers[1].hunting


def _pass_setup(env, k):
    """Blue's man on the ball at the centre spot, a teammate 20 m behind him, every red player far away.
    Returns the teammate and the index of the pass to him."""
    from football.actions import Action, Kind
    mate = [j for j in range(5) if j != k][0]
    env.pos[0][k] = np.array([50.0, 30.0])
    env.ball_pos = env.pos[0][k].copy()
    env.pos[0][mate] = np.array([30.0, 30.0])
    env.pos[1][:] = [[90.0, 10.0 + 10 * j] for j in range(5)]
    return mate, env.space.index_of(Action(Kind.PASS, 0))


def test_a_learning_team_is_only_offered_the_passes_it_can_see_are_open():
    env, k = _our_kickoff()
    mate, pass_to_mate = _pass_setup(env, k)
    assert env._obs_out()["mask"][k][pass_to_mate]                 # nobody near the lane
    env.pos[1][0] = np.array([40.0, 30.0])                         # a red player standing on it
    assert not env._obs_out()["mask"][k][pass_to_mate]
    env.pos[1][0] = np.array([40.0, 58.0])                         # ... and well away from it again
    assert env._obs_out()["mask"][k][pass_to_mate]
    off = FootballEnv(load_config(overrides={"actions.open_pass.enabled": False}))
    off.reset(seed=0, start_team=0)
    mate, pass_to_mate = _pass_setup(off, off.holder[1])
    off.pos[1][0] = np.array([40.0, 30.0])
    assert off._obs_out()["mask"][off.holder[1]][pass_to_mate]     # switched off: every pass is offered


def test_the_scripted_team_keeps_every_pass_in_its_mask():
    from football.observation import build_observation
    env = FootballEnv(load_config(), scripted_ours=True)
    env.reset(seed=0, start_team=0)
    k = env.holder[1]
    mate, pass_to_mate = _pass_setup(env, k)
    env.pos[1][0] = np.array([40.0, 30.0])
    assert build_observation(env, 0, k).mask[pass_to_mate]         # it does its own check before passing


def test_a_learning_team_meets_a_pass_without_being_asked():
    from football.actions import Action, Kind
    env, k = _our_kickoff()
    mate, pass_to_mate = _pass_setup(env, k)
    away = env.space.index_of(Action(Kind.MOVE_DIR, 2))            # the receiver wants to walk off sideways ...
    stay = env.space.index_of(Action(Kind.STAY))
    obs, _, _, _ = env.step({k: pass_to_mate, mate: away})
    assert env.ball_state == "flight" and env.flight["designated"] == (0, mate)
    while env.ball_state == "flight":
        assert list(np.flatnonzero(obs["mask"][mate])) == [stay]   # ... but while the ball is coming he has no choice
        obs, _, _, info = env.step({mate: away})
    assert env.holder == (0, mate) and env.stats["passes_completed"][0] == 1
    assert env.stats["invalid_actions"][0] == 0


def test_one_long_pass_is_paid_for_no_more_than_the_cap():
    env, k = _our_kickoff(only(w_progress=0.05))
    mate, pass_to_mate = _pass_setup(env, k)
    env.pos[0][k] = np.array([30.0, 30.0])                         # now the pass goes 50 m forward
    env.ball_pos = env.pos[0][k].copy()
    env.pos[0][mate] = np.array([80.0, 30.0])
    env.pos[1][:] = [[5.0, 2.0 + j] for j in range(5)]             # red is behind the passer, too far to cut it out
    env.step({})                                                   # a step with the ball sets the mark
    paid = 0.0
    env.step({k: pass_to_mate})
    while env.ball_state == "flight":
        paid += env.step({})[3]["reward_parts"]["shaping"]
    r = env.cfg.reward
    assert env.holder == (0, mate) and np.isclose(paid, r.w_progress * r.progress_pass_cap * r.scale) and r.progress_pass_cap < 50


def test_everyone_sees_the_ball_but_not_every_opponent():
    from football.observation import build_observation
    env = FootballEnv(load_config(overrides={"vision.radius": 5.0}))
    env.reset(seed=0)
    for team in (0, 1):
        for i in range(5):
            o = build_observation(env, team, i)
            assert o.ball_visible and np.allclose(o.ball_pos, to_frame(team, env.ball_pos, env.L, env.W))
            assert o.opp_visible.sum() < 5 or team == env.holder[0]     # with a 5 m radius hardly any opponent is seen
    hidden = FootballEnv(load_config(overrides={"vision.radius": 5.0, "vision.ball_always": False}))
    hidden.reset(seed=0)
    assert not all(build_observation(hidden, 0, i).ball_visible for i in range(5))


def test_a_defender_far_from_the_passer_cuts_out_a_long_ball():
    env, k = _our_kickoff()
    env.controllers[1].act = type(env.controllers[1]).act.__get__(env.controllers[1])     # red plays normally here
    mate, pass_to_mate = _pass_setup(env, k)
    env.pos[0][k] = np.array([20.0, 30.0])                         # a 60 m ball up the middle
    env.ball_pos = env.pos[0][k].copy()
    env.pos[0][mate] = np.array([80.0, 30.0])
    env.pos[1][:] = [[5.0, 2.0 + j] for j in range(5)]
    env.pos[1][0] = np.array([70.0, 40.0])                         # one red player 10 m off its path, 50 m from the passer
    assert dist_far(env, 0, k) > env.cfg.vision.radius             # ... so the passer does not see him, and may pass
    assert env._obs_out()["mask"][k][pass_to_mate]
    env.step({k: pass_to_mate})
    assert env.flight["designated"] == (1, 0) and env.flight["predicted"] == "intercepted"
    while env.ball_state == "flight":
        env.step({})
    assert env.holder == (1, 0)                                    # he walked to its path (the scripted first rule) and took it


def dist_far(env, j, k):
    return float(np.linalg.norm(env.pos[1][j] - env.pos[0][k]))


def test_the_scripted_team_closes_the_receiver_down_while_the_pass_is_in_the_air():
    env, k = _our_kickoff()
    env.controllers[1].act = type(env.controllers[1]).act.__get__(env.controllers[1])     # red plays normally here
    mate, pass_to_mate = _pass_setup(env, k)
    env.pos[0][mate] = np.array([30.0, 30.0])
    env.pos[1][:] = [[90.0, 2.0 + j] for j in range(5)]
    env.pos[1][2] = np.array([30.0, 52.0])                         # one red player 22 m to the side of the receiver:
    start = float(np.linalg.norm(env.pos[1][2] - env.pos[0][mate]))   # too far to cut the pass out, near enough to see him
    env.step({k: pass_to_mate})
    assert env.flight["designated"] == (0, mate)
    while env.ball_state == "flight":
        env.step({})
    assert float(np.linalg.norm(env.pos[1][2] - env.pos[0][mate])) < start - 3.0   # he came toward him while it flew


def test_a_shot_is_paid_by_its_chance_of_scoring():
    from football.actions import Action, Kind
    paid = {}
    for name, x in (("far", 68.0), ("close", 92.0)):
        env, k = _our_kickoff(only(w_shot=5.0))
        env.pos[1][:] = [[5.0, 2.0 + j] for j in range(5)]         # nobody in the way
        env.pos[0][k] = np.array([x, 30.0])
        env.ball_pos = env.pos[0][k].copy()
        _, _, _, info = env.step({k: env.space.index_of(Action(Kind.SHOOT))})
        shot = [e for e in info["events"] if e["type"] == "shot"][0]
        paid[name] = info["reward_parts"]["shaping"]
        assert shot["p"] > 0 and np.isclose(paid[name], env.cfg.reward.w_shot * shot["p"] * env.cfg.reward.scale, atol=1e-9)
    assert paid["far"] < paid["close"]
    # a big chance earns w_big_chance on top, a shot from far out does not
    for x, big in ((68.0, False), (92.0, True)):
        env, k = _our_kickoff(only(w_shot=5.0, w_big_chance=20.0))
        env.pos[1][:] = [[5.0, 2.0 + j] for j in range(5)]
        env.pos[0][k] = np.array([x, 30.0])
        env.ball_pos = env.pos[0][k].copy()
        _, _, _, info = env.step({k: env.space.index_of(Action(Kind.SHOOT))})
        p = [e for e in info["events"] if e["type"] == "shot"][0]["p"]
        assert (p >= env.cfg.reward.big_chance) == big
        r = env.cfg.reward
        assert np.isclose(info["reward_parts"]["shaping"], (r.w_shot * p + (20.0 if big else 0.0)) * r.scale, atol=1e-9)


def test_hold_shape_takes_a_learning_player_to_where_a_scripted_player_would_stand():
    from football.actions import Kind
    from football.geometry import move_toward
    from football.observation import build_observation
    env = FootballEnv(load_config())
    env.reset(seed=0, start_team=1)
    hold = {i: env.space.index_of_kind(Kind.HOLD_SHAPE) for i in range(5)}
    v, g = env.cfg.speeds.player, env.shape_guide
    before = env.pos[0].copy()
    env.step(hold)
    spots = np.array([g.shape_spot(i) for i in range(5)])
    assert np.allclose(env.pos[0], [move_toward(before[i], spots[i], v) for i in range(5)])
    assert not np.allclose(spots, env.anchors[0])                  # not the bare anchors: already leaning toward the ball
    # red keeps the ball out on one wing: the spots slide across to that side, as the scripted team's shape does
    env.holder, env.ball_state = (1, 0), "held"
    env.pos[1][0] = np.array([25.0, 50.0])
    env.ball_pos = env.pos[1][0].copy()
    for _ in range(60):
        g.watch({i: build_observation(env, 0, i) for i in range(5)})
    wing = np.array([g.shape_spot(i) for i in range(5)])
    assert wing[:, 1].mean() > env.anchors[0][:, 1].mean() + 1.0
    # switched off: the bare anchor, and the scripted red team never has a guide
    off = FootballEnv(load_config(overrides={"actions.shape_guide": False}))
    off.reset(seed=0, start_team=1)
    before = off.pos[0].copy()
    off.step(hold)
    assert off.shape_guide is None
    assert np.allclose(off.pos[0], [move_toward(before[i], off.anchors[0][i], v) for i in range(5)])
    both = FootballEnv(load_config(), scripted_ours=True)
    both.reset(seed=0)
    assert both.shape_guide is None


def test_the_team_is_paid_for_standing_on_its_hold_shape_spots():
    from football.geometry import to_frame
    def pay(move, overrides=None):
        env, k = _our_kickoff(overrides)
        g = env.shape_guide
        if g is not None:
            env.step({})                                           # one step, so that the guide has seen the team
        for j in range(5):
            if j != k and g is not None:
                env.pos[0][j] = to_frame(0, g.shape_spot(j), env.L, env.W) + move(j)
        return env.step({})[3]["reward_parts"]["shaping"], env.cfg.reward
    pay_ = pay
    pay = lambda move, overrides=only(w_formation=0.02): pay_(move, overrides)
    on_spot, r = pay(lambda j: np.zeros(2))
    assert 0.9 * r.w_formation * r.scale < on_spot <= r.w_formation * r.scale   # all four on their spots: nearly all
    off_spot, _ = pay(lambda j: np.array([-r.formation_dist, 0.0]))
    assert off_spot < 0.5 * on_spot                                # all four formation_dist m away: about a third
    one_off, _ = pay(lambda j: np.array([-40.0, 0.0]) if j == [i for i in range(5) if i != 0][0] else np.zeros(2))
    assert off_spot < one_off < on_spot
    assert pay(lambda j: np.zeros(2), ONE_TERM)[0] == 0.0          # switched off
    assert pay(lambda j: np.zeros(2), {"actions.shape_guide": False})[0] == 0.0   # no spots to stand on


def test_context_mask_offers_a_learning_player_only_the_choices_of_his_situation():
    from football.actions import Kind
    from football.observation import build_observation
    def kinds(mask):
        return {env.space.decode(i).kind for i in np.flatnonzero(mask)}
    env, k = _our_kickoff({"actions.context_mask": True})
    mate = [j for j in range(5) if j != k][0]
    mask = env._obs_out()["mask"]
    assert {Kind.MOVE_DIR, Kind.DRIBBLE_GOAL} <= kinds(mask[k]) <= {Kind.MOVE_DIR, Kind.STAY, Kind.PASS, Kind.DRIBBLE_GOAL, Kind.SHOOT}
    assert {Kind.STAY, Kind.HOLD_SHAPE, Kind.GO_TO_BALL} <= kinds(mask[mate]) and Kind.MOVE_DIR not in kinds(mask[mate])
    env, k = _our_kickoff()                                        # switched off: every valid action
    mask = env._obs_out()["mask"]
    assert {Kind.HOLD_SHAPE, Kind.GO_TO_BALL} <= kinds(mask[k]) and Kind.MOVE_DIR in kinds(mask[mate])
    env = FootballEnv(load_config(overrides={"actions.context_mask": True}), scripted_ours=True)
    env.reset(seed=0, start_team=0)                                # ... and a scripted team keeps all of them too
    assert Kind.MOVE_DIR in kinds(build_observation(env, 0, mate).mask)


def test_attack_value_is_the_carriers_chance():
    env = FootballEnv(load_config(overrides={"reward.w_chance": 1.0}))
    env.reset(seed=0, start_team=0)
    i = env.holder[1]
    env.pos[0][i] = np.array([50.0, 30.0])
    assert env.attack_value() == 0.0                            # out of shooting range
    env.pos[0][i] = np.array([90.0, 30.0])                      # close to goal: a chance
    assert 0.0 < env.attack_value() <= 1.0


def test_attack_reward_is_off_by_default_and_paid_on_the_change_of_the_chance_when_on():
    from football.actions import Kind
    env = FootballEnv(load_config(overrides=only()))
    env.reset(seed=0, start_team=0)
    env.step({})
    assert env.attack_v is None
    env = FootballEnv(load_config(overrides=only(w_chance=1.0)))
    env.reset(seed=0, start_team=0)
    stay = {j: env.space.index_of_kind(Kind.STAY) for j in range(5)}
    env.step(stay)
    v = env.attack_v
    env.pos[0][env.holder[1]] = np.array([92.0, 30.0])           # in front of goal at once
    _, _, _, info = env.step(stay)
    if env.holder is not None and env.holder[0] == 0:
        assert np.isclose(info["reward_parts"]["shaping"], (env.attack_v - v) * env.cfg.reward.scale) and env.attack_v > v


def test_per_player_reward_pays_each_player_for_his_own_spot():
    from football.actions import Kind
    for pp in (False, True):
        env = FootballEnv(load_config(overrides=only(per_player=pp, w_formation=0.01)))
        env.reset(seed=0, start_team=0)
        stay = {j: env.space.index_of_kind(Kind.STAY) for j in range(5)}
        for _ in range(3):
            _, r, _, info = env.step(stay)
        rp = info["reward_players"]
        assert rp.shape == (5,)
        if not pp:
            assert np.allclose(rp, r)                           # the team reward for everyone, as before
        else:
            own = own_part(info)
            assert (own >= 0).all() and (own > 0).sum() == 4    # the player nearest the ball is not paid
            assert not np.allclose(own[own > 0], own[own > 0][0])   # each player by his own spot


def test_gae_gives_each_player_his_own_advantage():
    from training.train import gae
    rew = np.zeros((3, 1, 5), np.float32)
    rew[1, 0, 2] = 1.0                                          # only player 2 is paid
    adv, ret = gae(rew, np.zeros((4, 1, 5), np.float32), np.zeros((3, 1), np.float32), 0.9, 1.0)
    assert adv[0, 0, 2] > 0 and np.allclose(np.delete(adv, 2, axis=2), 0.0)


def test_receiver_call_hides_a_pass_to_a_teammate_marked_out_of_the_passers_sight():
    from football.actions import Kind
    from football.observation import build_observation
    for call in (False, True):
        env = FootballEnv(load_config(overrides={"actions.open_pass.call": call}))
        env.reset(seed=0, start_team=0)
        i = env.holder[1]
        mate = [k for k in range(5) if k != i][0]
        env.pos[0] = np.array([[5.0, 5.0], [5.0, 55.0], [8.0, 8.0], [8.0, 52.0], [8.0, 30.0]])
        env.pos[0][i], env.pos[0][mate] = np.array([20.0, 30.0]), np.array([55.0, 30.0])     # 35 m apart
        env.pos[1] = np.array([[53.0, 30.0], [95.0, 5.0], [95.0, 55.0], [90.0, 30.0], [85.0, 10.0]])
        o = build_observation(env, 0, i)                        # the man marking him is 33 m from the passer
        assert not o.opp_visible[0]
        assert o.mask[env.space.index_of_kind(Kind.PASS)] == (not call)


def test_a_poor_shot_is_one_he_could_have_bettered():
    env = FootballEnv(load_config())
    env.reset(seed=0)
    far = [np.array([5.0, 5.0])] * 5
    assert env.poor_shot(np.array([75.0, 30.0]), far)                               # 25 m out, nobody near: go closer
    assert not env.poor_shot(np.array([85.0, 30.0]), far[:4] + [np.array([81.0, 33.0])])    # pressed, a 0.38 chance
    assert not env.poor_shot(np.array([85.0, 30.0]), far[:4] + [np.array([90.0, 26.0])])    # blocked, a 0.31 chance
    assert not env.poor_shot(np.array([66.0, 30.0]), far[:4] + [np.array([63.0, 33.0])])    # closing in, 0.17: shoot now
    assert env.poor_shot(np.array([66.0, 30.0]), far[:4] + [np.array([67.5, 30.0])])        # on him, 34 m: too poor
    assert env.poor_shot(np.array([75.0, 30.0]), far[:4] + [np.array([68.0, 30.0])])        # 7 m behind him: free
    assert not env.poor_shot(np.array([88.0, 30.0]), far)                           # 12 m out, open: a good shot
    assert env.poor_shot(np.array([88.0, 30.0]), far[:4] + [np.array([89.5, 30.0])])        # 12 m out, blocked: no


def test_a_poor_shot_costs_the_shooter_himself():
    from football.actions import Action, Kind
    env, k = _our_kickoff(only(per_player=True, w_poor_shot=-10.0))
    env.pos[1][:] = [[5.0, 2.0 + j] for j in range(5)]           # nobody near: from 25 m he should carry it on
    env.pos[0][k] = np.array([75.0, 30.0])
    env.ball_pos = env.pos[0][k].copy()
    _, r, _, info = env.step({k: env.space.index_of(Action(Kind.SHOOT))})
    own = own_part(info)
    assert np.isclose(own[k], -10.0 * env.cfg.reward.scale) and np.allclose(np.delete(own, k), 0.0)
    assert env.stats["poor_shots"][0] == 1


def test_a_carrier_tackled_soon_after_he_had_the_shot_on_pays_for_the_missed_shot(monkeypatch):
    import football.env as fe
    from football.actions import Action, Kind
    monkeypatch.setattr(fe, "tackle_probability", lambda a, b: 1.0)
    env, k = _our_kickoff(only(per_player=True, w_missed_shot=-10.0))
    env.pos[1][:] = [[5.0, 2.0 + j] for j in range(5)]
    env.pos[0][k] = np.array([74.0, 30.0])                       # 26 m out
    env.ball_pos = env.pos[0][k].copy()
    env.pos[1][0] = np.array([74.0, 25.0])                       # a defender closing in, 5 m away: the shot is on
    assert env.shot_on_now(k)
    stay = env.space.index_of_kind(Kind.STAY)
    env.step({k: stay})                                          # he does not shoot
    env.pos[1][0] = env.pos[0][k] + np.array([0.0, -1.5])        # now the defender is on him and takes it
    env.controllers[1].act = lambda t, obs: {0: Action(Kind.TACKLE)}
    _, r, _, info = env.step({k: stay})
    assert env.holder == (1, 0) and env.stats["missed_shots"][0] == 1
    assert np.isclose(own_part(info)[k], -10.0 * env.cfg.reward.scale)


def test_winning_the_ball_higher_up_pays_more():
    env = FootballEnv(load_config(overrides={"reward.win_zone": 1.0}))
    env.reset(seed=0)
    assert env._zone(np.array([0.0, 30.0])) == 1.0 and env._zone(np.array([100.0, 30.0])) == 2.0


def test_open_lanes_cost_counts_the_passes_their_carrier_has_open():
    from football.actions import Kind
    env = FootballEnv(load_config(overrides=only(w_open_lanes=-1.0)))
    env.reset(seed=0, start_team=1)
    h = env.holder[1]
    mates = [k for k in range(5) if k != h]
    env.pos[1][h] = np.array([60.0, 30.0])                       # their carrier, his four men 15-20 m in front
    for k, p in zip(mates, ([45.0, 15.0], [45.0, 45.0], [40.0, 25.0], [40.0, 35.0])):
        env.pos[1][k] = np.array(p)
    env.ball_pos = env.pos[1][h].copy()
    env.pos[0] = np.array([[95.0, 5.0], [95.0, 55.0], [90.0, 30.0], [92.0, 10.0], [92.0, 50.0]])   # all of ours far away
    assert env.open_lanes() == 4
    env.pos[0][0] = np.array([52.0, 22.0]); env.pos[0][1] = np.array([52.0, 38.0])                 # two on the lanes
    assert env.open_lanes() < 4
    n = env.open_lanes()
    _, _, _, info = env.step({j: env.space.index_of_kind(Kind.STAY) for j in range(5)})
    if env.holder is not None and env.holder[0] == 1:
        assert info["reward_parts"]["shaping"] < 0
        blame = env.open_lane_blame()                            # each open lane charged to one of ours near it,
        near = min(range(5), key=lambda j: np.linalg.norm(env.pos[0][j] - env.ball_pos))
        assert len(blame) == env.open_lanes() and near not in [j for j, _ in blame]   # never the one going for it
        assert all(0.0 <= d <= 1.0 for _, d in blame)            # weighted by how dangerous each pass is


def test_the_result_counts_for_more_than_all_small_terms_together():
    from football.actions import Kind
    env = FootballEnv(load_config(overrides=only(w_formation=100.0, small_cap=50.0)))   # huge, but capped
    env.reset(seed=0, start_team=0)
    stay = {j: env.space.index_of_kind(Kind.STAY) for j in range(5)}
    for _ in range(20):
        env.step(stay)
    assert np.allclose(env.small_total, env.cfg.reward.small_cap)      # stopped at the cap
    env.t, env.score = env.T - 1, [1, 0]                                 # last step, 1-0 up
    _, _, done, info = env.step(stay)
    r = env.cfg.reward
    assert done and np.isclose(info["reward_parts"]["terminal"], r.eta * r.scale)
    assert r.eta + r.goal > r.small_cap                                  # the result and the goals outweigh the small terms


def test_a_loose_pass_picked_up_by_them_costs_the_passer():
    from football.actions import Kind
    env = FootballEnv(load_config(overrides=only(per_player=True, w_turnover=-2.0)))
    env.reset(seed=0, start_team=1)
    env.holder, env.ball_state, env.flight = None, "loose", None
    env.ball_pos = env.pos[1][0].copy()                                  # lying at a red player's feet
    env.lost_from = 3                                                    # our player 3 played it there
    _, _, _, info = env.step({j: env.space.index_of_kind(Kind.STAY) for j in range(5)})
    assert env.holder == (1, 0) and env.stats["turnovers"][0] == 1
    assert np.isclose(own_part(info)[3], -2.0 * env.cfg.reward.scale)


def test_their_chance_rising_costs_us_the_mirror_of_ours():
    from football.actions import Kind
    env = FootballEnv(load_config(overrides=only(w_chance_against=-10.0)))
    env.reset(seed=0, start_team=1)
    env.controllers[1].act = lambda t, obs: {}                          # red stands still
    stay = {j: env.space.index_of_kind(Kind.STAY) for j in range(5)}
    env.step(stay)
    v = env.their_v
    env.pos[1][env.holder[1]] = np.array([10.0, 30.0])                   # their carrier in front of our goal
    env.pos[0][:] = [[95.0, 5.0 + 12 * j] for j in range(5)]             # ours far away
    _, _, _, info = env.step(stay)
    if env.holder is not None and env.holder[0] == 1:
        assert env.their_v > v
        assert np.isclose(info["reward_parts"]["shaping"], -10.0 * (env.their_v - v) * env.cfg.reward.scale)


def test_usefulness_is_zero_for_a_pass_back_and_full_for_one_far_forward():
    env = FootballEnv(load_config())
    env.reset(seed=0, start_team=0)
    env.pos[0][0], env.pos[0][1], env.pos[0][2] = [50.0, 30.0], [30.0, 30.0], [75.0, 30.0]
    env.pos[1][:] = [[5.0, 2.0 + j] for j in range(5)]
    assert env.usefulness(0, 0, 1) == 0.0 and env.usefulness(0, 0, 2) == 1.0


def test_a_learning_team_can_play_a_through_ball_into_space_ahead_of_a_teammate():
    from football.actions import Action, Kind
    env, k = _our_kickoff(only())
    mate = [j for j in range(5) if j != k][0]
    env.pos[0][k] = np.array([40.0, 30.0])
    env.ball_pos = env.pos[0][k].copy()
    env.pos[0][mate] = np.array([55.0, 30.0])                            # a teammate 15 m ahead, open grass in front
    env.pos[1][:] = [[95.0, 5.0 + 12 * j] for j in range(5)]
    through = env.space.index_of(Action(Kind.THROUGH, 0))
    assert through is not None and env._obs_out()["mask"][k][through]
    pt = env.through_point(0, k, 0, env.called_defenders(0, k, 0))
    assert pt[0] > env.pos[0][mate][0] + 3.0                             # ahead of him, not to his feet
    _, _, _, info = env.step({k: through})
    ev = [e for e in info["events"] if e["type"] == "pass"][0]
    assert ev["through"] and env.stats["through_balls"][0] == 1
    env.pos[1][0] = env.pos[0][mate] + np.array([6.0, 0.0])              # a defender in that space: not offered
    env.holder, env.ball_state, env.flight = (0, k), "held", None
    env.ball_pos = env.pos[0][k].copy()
    assert not env._obs_out()["mask"][k][through]

