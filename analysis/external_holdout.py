"""external_holdout.py — do the scheduler settings tuned on dp14 survive on another team's trajectories? Paper section 6.1.

    python analysis/external_holdout.py TRACES.json

TRACES.json is data/traces.json of the Kaggle dataset tantan0327/arc3-paper-artefacts (CC0): per action, generated
tokens, wall clock and level clears. Its requests are approximated as in sim/build_dataset.py (no prefill data).
The deadline is set so that D' can spend the stated share of all recorded tokens.
"""
import json, os, random, sys
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sim'))
import sim
from sim import Run, Trace, dfranzen_original, game_score, make_dprime, simulate

# These runs were cut after 25 to 60 minutes. A stalled run would burn the budget on tokens that were never recorded,
# so a run ends when its recorded trajectories end.
_exhausted = Run._exhausted
def _end(self):
    _exhausted(self)
    if self.stalled:
        self.stalled = False; self.done = True
Run._exhausted = _end

SY = json.load(open(os.path.join(ROOT, 'sim/synth.json')))
BEST = [x for x in json.load(open(os.path.join(ROOT, 'sim/eval_all.json'))) if x[0] == '探索#1'][0][3]


def trace(game, pas, g):
    steps = g['steps']
    apl, clears, last = [], [], 0
    for i, s in enumerate(steps, start=1):
        if s[2]:
            apl.append(i - last); clears.append(i); last = i
    N, base = g['N'], g['base_apl']
    req, since = [], 0.0
    for i, s in enumerate(steps):
        if s[0] <= 0:
            continue
        pt = 120000
        if since >= SY['gen_between_trims']:
            pt, since = 60000, 0.0
        req.append((i, pt, pt - (SY['unc_trim'] if pt == 60000 else SY['unc_normal']), s[0]))
        since += s[0]
    if not req:
        return None
    r = dict(game=game, pas=pas, N=N, apl=apl + [0] * (N - len(apl)), base=base, L=len(clears),
             state='won' if len(clears) >= N else 'gave_up', req=req, clears=clears,
             wall=[s[1] for s in steps], final_wall=steps[-1][1])
    return Trace(r)


data = json.load(open(sys.argv[1]))['runs']
per = defaultdict(int); traces = []
for run, games in data.items():
    for game, g in games.items():
        if g.get('steps'):
            t = trace(game, per[game], g)
            if t:
                traces.append(t); per[game] += 1
print(len(traces), 'trajectories,', len(per), 'games; recorded mean', round(float(np.mean([game_score(t.N, t.L_real, t.apl, t.base) for t in traces])), 2))
total = sum(sum(t.cost) for t in traces)
rng = random.Random(0)
games = sorted(per)
for share, slots in ((.5, 14), (.7, 14), (.9, 14)):
    T = int(total * share / slots)
    run = lambda tr, P, seed: simulate(tr, P, T=T, slots=slots, seed=seed)[0]
    d = np.mean([run(traces, make_dprime(), s) for s in range(4)])
    b = np.mean([run(traces, make_dprime(BEST), s) for s in range(4)])
    o = np.mean([run(traces, dfranzen_original, s) for s in range(4)])
    wins = 0
    for k in range(20):
        pick = [rng.choice(games) for _ in games]
        idx = [t for g in pick for t in traces if t.game == g]
        wins += run(idx, make_dprime(BEST), k) > run(idx, make_dprime(), k)
    print(f'budget {share:.0%} of all compute: D\' {d:.2f}, tuned on dp14 {b:.2f} ({b - d:+.2f}, better in {wins}/20 game resamples), original {o:.2f}')
