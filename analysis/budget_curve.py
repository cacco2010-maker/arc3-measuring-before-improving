"""budget_curve.py — how much score a compute budget buys, and what the two memory designs got for theirs.

    python analysis/budget_curve.py            # prints the tables of paper section 6.5, writes analysis/budget_curve.json

Replays D' on the dp14 trajectories with the deadline scaled by f (the scheduler knows the new deadline).
The replay is chaotic in f, so each point averages five deadlines within 2% and three stitching seeds.
Check without the simulator: the three evaluations without memory, cut at a matched token or action count.
"""
import json, os, pickle, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sim'))
from sim import T_DEADLINE, game_score, load_traces, make_dprime, simulate

FS = [.55, .6, .625, .65, .675, .7, .725, .75, .775, .8, .85, .9, .95, 1, 1.05, 1.1, 1.2, 1.3]
ACTUAL = {'appended': ('arc3-dp14pysum-v1', 13.24, 14415), 'replacing': ('sumtrimB-v3', 19.70, 14432)}
DP14_TOKENS = 18.84


def per_game(name):
    out = {}
    for line in open(os.path.join(ROOT, 'data/summaries', name + '.txt')):
        m = re.match(r'\s+(\S+-\w+): score=([\d.]+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def replay(traces, f):
    games, tok, act = {}, [], []
    for j in (-.02, -.01, 0, .01, .02):
        for seed in range(3):
            _, runs = simulate(traces, make_dprime(), T=int(T_DEADLINE * f * (1 + j)), seed=seed)
            for r in runs:
                games.setdefault(r.tr.game, []).append(r.score())
            tok.append(sum(r.cum_tokens for r in runs) / 1e6)
            act.append(sum(r.src.act[min(r.i, r.src.n - 1)] if r.src is r.tr else r.tr.act[-1] for r in runs))
    return {g: float(np.mean(v)) for g, v in games.items()}, float(np.mean(tok)), float(np.mean(act))


def interp(curve, f):
    fs = sorted(curve)
    lo = max(x for x in fs if x <= f); hi = min(x for x in fs if x >= f)
    w = 0 if hi == lo else (f - lo) / (hi - lo)
    return {g: (1 - w) * curve[lo][g] + w * curve[hi][g] for g in curve[lo]}


def gap(actual, predicted, rng):
    x = np.array([actual[g] - predicted[g] for g in sorted(actual)])
    bs = [rng.choice(x, len(x)).mean() for _ in range(20000)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def cut(runs, key, target):
    """Score of an evaluation cut at the wall-clock time where its total tokens or actions reach target."""
    def at(t):
        s = tok = act = 0
        for r in runs:
            n = int((np.array(r['wall']) <= t).sum())
            s += game_score(r['N'], sum(c <= n for c in r['clears']), r['apl'], r['base'])
            tok += r['tok'][n - 1] / 1e6 if n else 0
            act += n
        return s / len(runs), tok, act
    lo, hi = 0, T_DEADLINE * 1.05
    for _ in range(30):
        mid = (lo + hi) / 2
        if at(mid)[key] < target: lo = mid
        else: hi = mid
    return at(mid)[0]


def cut_time(runs, f):
    s = 0
    for r in runs:
        n = int((np.array(r['wall']) <= f * T_DEADLINE).sum())
        s += game_score(r['N'], sum(c <= n for c in r['clears']), r['apl'], r['base'])
    return s / len(runs)


def main():
    traces = load_traces()
    curve, res = {}, []
    for f in FS:
        curve[f], tok, act = replay(traces, f)
        s = float(np.mean(list(curve[f].values())))
        res.append(dict(f=f, score=s, tokens=tok, actions=act))
        print(f'f={f:.3f}  score {s:6.2f}  tokens {tok:5.2f}M  actions {act:6.0f}')
    fa = max(r['f'] for r in res if r['actions'] <= 14420)  # stitching makes the action count approximate above f = 0.9
    print(f'replay reaches 14,420 actions at f = {fa:.3f}')
    rng = np.random.default_rng(0)
    out = dict(curve=res, f_actions=fa, gaps={})
    for name, (src, tokens, _) in ACTUAL.items():
        actual = per_game(src)
        for label, f in (('equal tokens', tokens / DP14_TOKENS), ('equal actions', fa)):
            pred = interp(curve, f)
            d, a, b = gap(actual, pred, rng)
            print(f'{name:9} {label:13} f={f:.3f} replay {np.mean(list(pred.values())):.2f}  actual {np.mean(list(actual.values())):.2f}'
                  f'  gap {d:+.2f} [{a:+.1f}, {b:+.1f}]')
            out['gaps'][f'{name} {label}'] = dict(f=f, replay=float(np.mean(list(pred.values()))), gap=d, lo=a, hi=b)
        s = [r['score'] for r in res]; f_eq = float(np.interp(np.mean(list(actual.values())), s, [r['f'] for r in res]))
        print(f'{name:9} scores like dp14 with f = {f_eq:.2f}')
        out['gaps'][f'{name} equivalent budget'] = f_eq
    data = pickle.load(open(os.path.join(ROOT, 'sim/dataset.pkl'), 'rb'))
    out['real_cuts'] = [dict(f=f, score=cut_time(data['arc3-dp14-v1'], f)) for f in (.6, .7, .8, .9)]
    print('dp14 really cut at f:', out['real_cuts'])
    for name in ('t06-v2', 'dprime-v1', 'arc3-dp14-v1'):
        print(f'{name:13} cut at 13.24M tokens {cut(data[name], 1, 13.24):.2f}, at 14,420 actions {cut(data[name], 2, 14420):.2f}')
    for name in ('t06-v2', 'dprime-v1', 'arc3-dp14-v1'):
        print(f'{name:13} real loss when cut at 90% / 70% of the time:', *(f'{cut_time(data[name], f) - cut_time(data[name], 1.01):+.2f}' for f in (.9, .7)))
    # residuals from the curve at each evaluation's own token count; the three evaluations without memory set the offset
    fs, ss = [r['f'] for r in res], [r['score'] for r in res]
    ev = {'t06': (17.49, 44.77), 'dprime': (17.40, 47.84), 'dp14': (18.84, 50.46), 'appended': (13.24, 43.96), 'replacing': (19.70, 44.54)}
    resid = {k: a - float(np.interp(t / DP14_TOKENS, fs, ss)) for k, (t, a) in ev.items()}
    off = float(np.mean([resid[k] for k in ('t06', 'dprime', 'dp14')]))
    print('residuals', {k: round(v, 2) for k, v in resid.items()}, 'offset', round(off, 2))
    out['residuals'] = resid; out['offset'] = off
    json.dump(out, open(os.path.join(ROOT, 'analysis/budget_curve.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
