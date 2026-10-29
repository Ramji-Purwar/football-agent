"""Phase 0 of idea.pdf (Section 4.1) on the final game, with enough matches to read the numbers: (i) a random team loses,
(ii) vision matters and the opponent's formation is readable from where it stands, (iii) a team with the playmaker
beats the same team without one, (iv) the formation win matrix M with best responses and the maximin formation.
Every match is the scripted team against the scripted team unless said otherwise. Run from football_adapt/ (server):
    PYTHONPATH=. python runs/_tools/phase0_full.py --matches 300 --out runs/phase0/phase0.txt"""
import argparse
import json
from multiprocessing import Pool
import numpy as np

MIDS = {"2-2-1": (3, 4), "2-1-2": (3,), "1-3-1": (2, 3, 4), "3-1-1": (4,)}


def one(job):
    kind, kw, seed = job
    from football import FootballEnv, load_config
    from agents import RandomAgent
    env = FootballEnv(load_config(), scripted_ours=(kind != "random"))
    obs, _ = env.reset(seed=seed, **kw)
    ag = RandomAgent(seed) if kind == "random" else None
    done = False
    while not done:
        obs, _, done, _ = env.step(ag.act(obs) if ag else None)
    out = dict(kind=kind, kw=kw, seed=seed, gf=env.score[0], ga=env.score[1])
    if kind == "random":
        out["exc"] = float(np.mean(env.opp_anchor_dist))
    return out


def W(rows):
    d = np.array([r["gf"] - r["ga"] for r in rows])
    s = (d > 0) + 0.5 * (d == 0)
    return s.mean(), 1.96 * s.std() / np.sqrt(len(s)), d.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=80_000)
    ap.add_argument("--procs", type=int, default=56)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from football import FootballEnv, load_config
    names = FootballEnv(load_config()).book.names
    big = load_config().vision.radius
    specs = [("random", {}), ("base", {})]
    specs += [("vision", dict(radius=r, radius_opp=big)) for r in (12.0, 120.0)]
    # (iii): our scripted team without a playmaker against theirs with one (each midfielder) and without
    specs += [("pm", dict(our_formation=f, opp_formation=f, sigma=s)) for f, ss in MIDS.items() for s in (0,) + ss]
    specs += [("M", dict(our_formation=f, opp_formation=g)) for f in names for g in names]
    jobs = [(k, kw, a.seed0 + i) for i in range(a.matches) for k, kw in specs]
    with Pool(a.procs) as p:
        rows = p.map(one, jobs, chunksize=4)
    json.dump(rows, open(a.out.replace(".txt", ".json"), "w"))
    get = lambda k, **kw: [r for r in rows if r["kind"] == k and all(r["kw"].get(x) == v for x, v in kw.items())]
    L = [f"Phase 0 on the final game, {a.matches} matches per line, seeds {a.seed0}+ (W of OUR side, +- 95%)", ""]
    w, c, d = W(get("random"))
    L.append(f"(i)   random team vs scripted 2-2-1: W {w:.2f} +-{c:.2f}, goal difference {d:+.2f}")
    w, c, d = W(get("base"))
    L.append(f"      scripted vs scripted (2-2-1 both): W {w:.2f} +-{c:.2f} (balance)")
    for r in (12.0, 120.0):
        w, c, d = W(get("vision", radius=r))
        L.append(f"(ii)  scripted with radius {r:g} vs scripted with {big:g}: W {w:.2f} +-{c:.2f}, GD {d:+.2f}")
    exc = np.mean([r["exc"] for r in get("random")])
    book = FootballEnv(load_config()).book
    seps = [np.mean(np.linalg.norm(book.anchors_team_frame(x) - book.anchors_team_frame(y), axis=1))
            for i, x in enumerate(names) for y in names[i + 1:]]
    L.append(f"      readability: opponents stand {exc:.1f} m from their home spot on average; the home spots of two "
             f"formations are {np.mean(seps):.1f} m apart (min {np.min(seps):.1f})")
    L.append("(iii) scripted team without a playmaker vs the same formation with one (W of the side WITHOUT him):")
    for f, ss in MIDS.items():
        w0, c0, d0 = W(get("pm", our_formation=f, sigma=0))
        parts = [f"none {w0:.2f}"] + [f"player {s} {W(get('pm', our_formation=f, sigma=s))[0]:.2f}" for s in ss]
        L.append(f"      {f}: " + ", ".join(parts))
    M = np.array([[W(get("M", our_formation=f, opp_formation=g))[0] for g in names] for f in names])
    L += ["", "(iv)  formation win matrix M(f, g): W of our scripted team in formation f (row) against g (column)",
          "        " + " ".join(f"{g:>6s}" for g in names) + "   worst   mean"]
    for f, row in zip(names, M):
        L.append(f"  {f:6s}" + " ".join(f"{x:6.2f}" for x in row) + f"  {row.min():6.2f} {row.mean():6.2f}")
    L.append("      best response to each g: " + ", ".join(f"{g} -> {names[int(np.argmax(M[:, j]))]}"
                                                          for j, g in enumerate(names)))
    L.append(f"      maximin formation: {names[int(np.argmax(M.min(1)))]} (worst case {M.min(1).max():.2f}); "
             f"best on average: {names[int(np.argmax(M.mean(1)))]} ({M.mean(1).max():.2f})")
    four = [names.index(g) for g in ("2-2-1", "2-1-2", "1-3-1", "3-1-1")]
    L.append(f"      against Q1's four training formations: best on average "
             f"{names[int(np.argmax(M[:, four].mean(1)))]} ({M[:, four].mean(1).max():.2f}), maximin "
             f"{names[int(np.argmax(M[:, four].min(1)))]} ({M[:, four].min(1).max():.2f})")
    L.append(f"      +- for one cell about {1.96 * 0.5 / np.sqrt(a.matches):.2f}")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
