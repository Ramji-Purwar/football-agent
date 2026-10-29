"""Fingerprint of the GAME only (events, positions, score), not of observations or rewards, to check that a change
to the reward or the learner's action set left play as it was. Run from inside a code tree (PYTHONPATH=.).
usage: hash_events.py [KEY=VALUE ...]   overrides for every match, e.g. actions.through.enabled=false"""
import hashlib, json, sys
from multiprocessing import Pool


def one(a):
    seat, ours, opp, seed, extra = a
    import numpy as np
    from football import FootballEnv, load_config
    from agents import RandomAgent
    ov = {"styles.ours": ours, "styles.opponent": opp}
    ov.update(extra)
    env = FootballEnv(load_config(overrides=ov), scripted_ours=(seat == "scripted"))
    ag = RandomAgent(seed) if seat == "random" else None
    obs, _ = env.reset(seed=seed)
    h = hashlib.sha256()
    done = False
    while not done:
        obs, r, done, info = env.step(ag.act(obs) if ag else None)
        h.update(np.round(env.pos, 6).tobytes()); h.update(np.round(env.ball_pos, 6).tobytes())
    h.update(json.dumps(env.event_log, sort_keys=True, default=str).encode())
    return f"{seat:<8} {ours:>10} v {opp:<10} seed {seed}  score {env.score[0]}-{env.score[1]}  {h.hexdigest()[:16]}"


if __name__ == "__main__":
    import yaml
    extra = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in sys.argv[1:])}
    jobs = [(s, a, b, seed, extra) for s in ("scripted", "random") for a, b in (("normal", "normal"), ("aggressive", "defensive"), ("defensive", "aggressive"))
            for seed in range(3)]
    with Pool(18) as pool:
        print("\n".join(pool.map(one, jobs)))
