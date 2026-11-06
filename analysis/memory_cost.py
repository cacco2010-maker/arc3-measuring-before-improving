"""memory_cost.py — reproduces the numbers of paper section 5.2 and 6.4.

    python analysis/memory_cost.py --replacing DIR --appended DIR

DIR holds an evaluation's output files (<game>_p<pass>_requests.jsonl, benchmark.json).
`--replacing` is the public notebook kazuhiro3381/arc3-sumtrimb, version 3.
dp14 is read from sim/dataset.pkl, because its raw logs no longer exist.
Without the two options only the score comparison runs (data/summaries is enough).
"""
import argparse, collections, glob, json, os, pickle, re
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROP = 20000  # a fall in prompt size larger than this is a trim or a replacement
REGROUND = re.compile(r"re-?(ground|orient|read|derive|establish|sync|build|construct|inspect|examine)|summary says"
                      r"|per (the )?summary|according to (the|my) summary|let me (first )?(verify|confirm|check)", re.I)


def responses(path):
    return [x for x in map(json.loads, open(path)) if x.get('event') == 'response']


def per_game(path):
    out = {}
    for line in open(path):
        m = re.match(r'\s+(\S+-\w+): score=([\d.]+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def scores():
    data = pickle.load(open(os.path.join(ROOT, 'sim/dataset.pkl'), 'rb'))
    games = {}
    for r in data['arc3-dp14-v1']:
        games.setdefault(r['game'], []).append(r['score'])
    ev = {'dp14': {g: float(np.mean(v)) for g, v in games.items()}}
    for name, f in (('dprime', 'dprime-v1'), ('t06', 't06-v2'), ('appended', 'arc3-dp14pysum-v1'), ('replacing', 'sumtrimB-v3')):
        ev[name] = per_game(os.path.join(ROOT, 'data/summaries', f + '.txt'))
    means = np.array([46.49] + [np.mean(list(v.values())) for v in ev.values()])  # 46.49: base, skarin's audit notebook
    chi = ((means - means.mean()) ** 2).sum() / 2.2 ** 2
    print(f'six evaluations: mean {means.mean():.2f} SD {means.std(ddof=1):.2f} chi2 {chi:.2f} p {stats.chi2.sf(chi, 5):.3f}')
    rng = np.random.default_rng(0)
    for a, b in (('appended', 'dp14'), ('replacing', 'dp14'), ('dp14', 'dprime')):
        x = np.array([ev[a][g] - ev[b][g] for g in sorted(ev[a])])
        bs = [rng.choice(x, len(x)).mean() for _ in range(20000)]
        print(f'{a} - {b}: {x.mean():+.2f}, 95% CI {np.percentile(bs, 2.5):+.1f} to {np.percentile(bs, 97.5):+.1f}')
    return data


def first_drop(req):
    return next((i for i in range(1, len(req)) if req[i - 1][1] - req[i][1] > DROP), None)


def window(req, cap=50000):
    i = first_drop(req)
    return [x for x in (req[:i] if i else req) if x[1] < cap]


def totals(segments):
    tok = req = act = turns = 0
    for s in segments:
        if s:
            a = [x[0] for x in s]
            tok += sum(x[3] for x in s); req += len(s); act += max(a) - min(a) + 1; turns += len(set(a))
    return f'tokens/action {tok / act:.0f}  requests/decision {req / turns:.2f}  tokens/request {tok / req:.0f}'


def matched(data, replacing):
    dp14 = {(r['game'], r['pas']): r['req'] for r in data['arc3-dp14-v1'] if r['req']}
    rep = {}
    for f in glob.glob(replacing + '/*_requests.jsonl'):
        m = re.match(r'(.+)_p(\d+)_requests', os.path.basename(f))
        rep[(m.group(1), int(m.group(2)))] = [(int(x['action']), x['usage']['prompt_tokens'], 0, x['usage']['completion_tokens']) for x in responses(f)]
    keys = sorted(set(rep) & set(dp14))
    print('matched runs', len(keys))
    for name, runs in (('dp14', dp14), ('replacing', rep)):
        print(f' {name:10s} before: {totals(window(runs[k]) for k in keys)}')
        print(f' {name:10s} after:  {totals(runs[k][len(window(runs[k])):] for k in keys)}')


def behaviour(path):
    c = collections.defaultdict(lambda: [0, 0])
    idle = collections.Counter()
    for f in glob.glob(path + '/*_requests.jsonl'):
        r = responses(f)
        seen, since = False, None
        for i in range(len(r) - 1):
            if i and r[i - 1]['usage']['prompt_tokens'] - r[i]['usage']['prompt_tokens'] > DROP:
                seen, since = True, 0
            idle['after' if seen else 'before'] += 1
            idle['after_idle' if seen else 'before_idle'] += 'executed NO actions' in str(r[i]['messages'][-1].get('content'))[:400]
            nxt = [m for m in r[i + 1]['messages'] if m['role'] == 'assistant']
            if nxt and r[i]['usage']['prompt_tokens'] - r[i + 1]['usage']['prompt_tokens'] <= DROP:
                k = 'before first drop' if not seen else '1st after' if since == 0 else '2-4 after' if since < 4 else 'later'
                c[k][0] += 1; c[k][1] += bool(REGROUND.search((nxt[-1].get('reasoning_content') or '')[:400]))
            if since is not None:
                since += 1
    print(os.path.basename(path.rstrip('/')), {k: f'{v[1] / v[0]:.1%}' for k, v in c.items()})
    print('  no-action requests', f"before {idle['before_idle'] / idle['before']:.1%}  after {idle['after_idle'] / idle['after']:.1%}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--replacing'); ap.add_argument('--appended')
    a = ap.parse_args()
    data = scores()
    if a.replacing:
        matched(data, a.replacing); behaviour(a.replacing)
    if a.appended:
        behaviour(a.appended)
