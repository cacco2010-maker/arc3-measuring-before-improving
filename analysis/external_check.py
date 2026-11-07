"""external_check.py — our noise model and budget loss on another team's runs. Paper section 5.3.

    python analysis/external_check.py DIR

DIR holds data/arms_2026-10-24/ of the Kaggle dataset tantan0327/arc3-paper-artefacts (CC0): 17 runs of the same
base harness and variants, 10 public games, one pass, 25 minutes each. Different games, budget and team from ours.
"""
import collections, glob, json, os, sys
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'sim'))
from sim import game_score

T = 1500
runs = {f.split('/')[-2]: json.load(open(f))['game_runs'] for f in sorted(glob.glob(sys.argv[1] + '/*/benchmark.json'))}
games = collections.defaultdict(list)
for rs in runs.values():
    for r in rs:
        games[r['game_id']].append(r['final_score'])
rms = float(np.sqrt(np.mean([np.var(v, ddof=1) for v in games.values()])))
means = np.array([np.mean([r['final_score'] for r in rs]) for rs in runs.values()])
se = rms / np.sqrt(len(games))
chi = ((means - means.mean()) ** 2).sum() / se ** 2
print(f'{len(runs)} runs, {len(games)} games: per-game RMS SD {rms:.1f}, predicted SE of a run mean {se:.2f}, '
      f'observed SD {means.std(ddof=1):.2f}, range {means.min():.1f}-{means.max():.1f}, chi2 {chi:.2f}, p {stats.chi2.sf(chi, len(means) - 1):.2f}')


def cut(rs, f):
    out = []
    for r in rs:
        n = sum(h['wallclock_seconds'] <= f * T for h in r['history'])
        clears = np.cumsum(r['actions_per_level'][:r['levels_completed']])
        tr_L = int((clears <= n).sum())
        out.append(game_score(r['number_of_levels'], tr_L, r['actions_per_level'], r['base_actions_per_level']))
    return np.mean(out)


end = np.mean([cut(rs, 1.05) for rs in runs.values()])
for f in (.9, .8, .7, .6):
    print(f'cut at {f:.0%} of the time: {np.mean([cut(rs, f) for rs in runs.values()]) - end:+.2f}')
