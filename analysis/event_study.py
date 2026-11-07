"""event_study.py — what happens around a context drop, by how much verbatim history survives it. Paper section 6.4.

    python analysis/event_study.py

Reads sim/dataset.pkl (dp14) and data/memory_trajectories.pkl (numbers only: action counter, prompt, cached and
generated tokens per request; only runs with real request logs). A drop is a fall of more than 20,000 prompt tokens.
A request is "no-action" if the action counter did not move, so its share is 1 - 1 / (requests per decision). Before = 10 requests before the drop,
after = 10 requests after it; windows that reach the run's ends or another drop are shortened. Bootstrap over runs.
"""
import os, pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DROP, W = 20000, 10


def events(req):
    """(no-action share before, after, tokens kept) for each drop of one run."""
    noact = [i > 0 and req[i][0] == req[i - 1][0] for i in range(len(req))]
    drops = [i for i in range(1, len(req)) if req[i - 1][1] - req[i][1] > DROP]
    out = []
    for j, d in enumerate(drops):
        lo = max(d - W, drops[j - 1] if j else 1)
        hi = min(d + W, drops[j + 1] if j + 1 < len(drops) else len(req))
        if d - lo >= 3 and hi - d >= 3:
            out.append((np.mean(noact[lo:d]), np.mean(noact[d:hi]), req[d][1], np.mean([r[3] for r in req[lo:d]]), np.mean([r[3] for r in req[d:hi]])))
    return out


def report(name, runs, rng):
    per_run = [e for e in (events(r['req']) for r in runs if r['req']) if e]
    flat = np.array([x for e in per_run for x in e])
    means = [np.array(e).mean(axis=0) for e in per_run]
    m = np.array(means); d = m[:, 1] - m[:, 0]
    bs = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    print(f'{name:10} {len(flat):4} drops in {len(per_run):3} runs, median kept {np.median(flat[:, 2]) / 1000:5.1f}k: '
          f'no-action {flat[:, 0].mean() * 100:4.1f}% -> {flat[:, 1].mean() * 100:4.1f}%, per-run change {d.mean() * 100:+.1f} points '
          f'(95% CI {np.percentile(bs, 2.5) * 100:+.1f} to {np.percentile(bs, 97.5) * 100:+.1f}); '
          f'tokens per request {flat[:, 3].mean():.0f} -> {flat[:, 4].mean():.0f}')
    return flat


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    data = pickle.load(open(os.path.join(ROOT, 'sim/dataset.pkl'), 'rb'))
    report('dp14', data['arc3-dp14-v1'], rng)
    memory = pickle.load(open(os.path.join(ROOT, 'data/memory_trajectories.pkl'), 'rb'))
    report('appended', memory['appended'], rng)
    f = report('replacing', memory['replacing'], rng)
    # dose within the replacing design
    q = np.percentile(f[:, 2], [33, 67])
    for lab, sel in (('least kept', f[:, 2] <= q[0]), ('middle', (f[:, 2] > q[0]) & (f[:, 2] <= q[1])), ('most kept', f[:, 2] > q[1])):
        print(f'  replacing, {lab:10} median {np.median(f[sel, 2]) / 1000:5.1f}k: no-action {f[sel, 0].mean() * 100:4.1f}% -> {f[sel, 1].mean() * 100:4.1f}%  (n={sel.sum()})')
