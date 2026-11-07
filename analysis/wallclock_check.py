"""wallclock_check.py — tests the replay's key assumption: a level's cost does not depend on when it is played.

    python analysis/wallclock_check.py

For every cleared level: log(tokens spent on it) against the hour it started, with game x level fixed effects
(the same level played at different times by different passes). Cleared levels only. Paper section 6.1.
"""
import os, pickle
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pickle.load(open(os.path.join(ROOT, 'sim/dataset.pkl'), 'rb'))
for name, runs in data.items():
    groups = defaultdict(list)
    for r in runs:
        prev = 0
        for k, c in enumerate(r['clears']):
            if c > len(r['tok']) or c < 1:
                continue
            t0 = r['wall'][prev - 1] if prev else 0.0
            cost = r['tok'][c - 1] - (r['tok'][prev - 1] if prev else 0)
            if cost > 0:
                groups[(r['game'], k)].append((t0 / 3600, np.log(cost)))
            prev = c
    keep = [g for g in groups.values() if len(g) >= 2]
    x = np.concatenate([np.array(g)[:, 0] - np.mean(g, axis=0)[0] for g in keep])
    y = np.concatenate([np.array(g)[:, 1] - np.mean(g, axis=0)[1] for g in keep])
    rng = np.random.default_rng(0)
    slope = lambda i: (x[i] * y[i]).sum() / (x[i] ** 2).sum()
    # bootstrap whole groups
    idx, s = [], 0
    for g in keep:
        idx.append(np.arange(s, s + len(g))); s += len(g)
    bs = [slope(np.concatenate([idx[j] for j in rng.integers(0, len(idx), len(idx))])) for _ in range(4000)]
    b = slope(np.arange(len(x)))
    print(f'{name:13} {len(x)} clears in {len(keep)} game-levels: {100 * (np.exp(b) - 1):+.1f}% tokens per hour later '
          f'(95% CI {100 * (np.exp(np.percentile(bs, 2.5)) - 1):+.1f} to {100 * (np.exp(np.percentile(bs, 97.5)) - 1):+.1f}), start-time SD {x.std():.2f} h')
