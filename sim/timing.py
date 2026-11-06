"""1 リクエストの所要時間の見積もり。

events.jsonl の transcript 冒頭にある時刻（HH:MM:SS）で分析ステップごとの壁時計を取り、
requests.jsonl の同じステップのリクエスト（生成トークン、未キャッシュの入力）と突き合わせて
  所要秒 ≈ a0·件数 + 生成/g + 未キャッシュ入力/p
を最小二乗で当てる。trim の直後のステップ（枠待ちを含む）は外す。
"""
import json, glob, os, re, pickle
from collections import defaultdict
import numpy as np

RUN = os.path.expanduser('~/arc-agi-3/runs/arc3-dp14-v1')
S = os.path.dirname(os.path.abspath(__file__))
data = pickle.load(open(f'{S}/dataset.pkl', 'rb'))['arc3-dp14-v1']
TS = re.compile(r'\| (\d\d):(\d\d):(\d\d) \|')

rows = []  # (dur, n_req, gen, uncached, after_trim, has_length_cut)
trim_gaps = []; normal_gaps = []
for r in data:
    g, p = r['game'], r['pas']
    ev = [json.loads(l) for l in open(f'{RUN}/artifacts/{g}_p{p}_events.jsonl')]
    step_t = {}
    for e in ev:
        m = TS.search(e.get('transcript', '') or '')
        if m and e.get('analysis_step'):
            t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
            step_t.setdefault(int(e['analysis_step']), t)
    steps = sorted(step_t)
    # 日跨ぎ補正（01:51 開始、10:02 終了なので跨がないが念のため）
    per_step = defaultdict(lambda: [0, 0, 0, False])
    trims_by_step = set()
    prev_pt = None
    for line in open(f'{RUN}/{g}_p{p}_requests.jsonl'):
        x = json.loads(line)
        if x.get('event') != 'response': continue
        u = x['usage']; st = int(x['analysis_step'])
        d = per_step[st]; d[0] += 1; d[1] += u['completion_tokens']
        d[2] += u['prompt_tokens'] - u['prompt_tokens_details'].get('cached_tokens', 0)
        if x.get('finish_reason') == 'length': d[3] = True
        if prev_pt is not None and prev_pt - u['prompt_tokens'] > 30000: trims_by_step.add(st)
        prev_pt = u['prompt_tokens']
    for i in range(len(steps) - 1):
        st, nxt = steps[i], steps[i + 1]
        dur = step_t[nxt] - step_t[st]
        if dur < 0: dur += 86400
        n, gen, unc, cut = per_step[st]
        after_trim = (st in trims_by_step) or (st - 1 in trims_by_step)
        rows.append((dur, n, gen, unc, after_trim, cut))
    # trim 時の行動間の壁時計の差
    w = r['wall']
    for a in r['trims']:
        if 1 <= a < len(w): trim_gaps.append(w[a] - w[a - 1])
    for i in range(1, len(w)):
        if (i + 1) not in r['trims']: normal_gaps.append(w[i] - w[i - 1])

A = np.array([x for x in rows if not x[4] and x[1] > 0], dtype=float)
print(f'steps total {len(rows)}, used {len(A)}')
X = np.c_[A[:, 1], A[:, 2], A[:, 3]]; y = A[:, 0]
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
print(f'fit: sec = {coef[0]:.2f}*n_req + gen/{1/coef[1]:.1f} + uncached/{1/coef[2]:.0f}   (R2 {1-((y-pred)**2).sum()/((y-y.mean())**2).sum():.3f})')
print(f'per-request mean dur {y.sum()/A[:,1].sum():.1f}s, gen tokens/sec per slot (naive) {A[:,2].sum()/y.sum():.1f}')
# 単純モデル: 生成トークンに比例
k = (A[:, 2] * y).sum() / (A[:, 2] ** 2).sum()
print(f'single-rate: sec = gen/{1/k:.1f}  (R2 {1-((y-A[:,2]*k)**2).sum()/((y-y.mean())**2).sum():.3f})')
print(f'trim gaps: n={len(trim_gaps)} median {np.median(trim_gaps):.0f}s  p25 {np.percentile(trim_gaps,25):.0f}  p75 {np.percentile(trim_gaps,75):.0f}  max {max(trim_gaps):.0f}')
print(f'normal gaps: median {np.median(normal_gaps):.0f}s  p90 {np.percentile(normal_gaps,90):.0f}  p99 {np.percentile(normal_gaps,99):.0f}')
json.dump(dict(a0=coef[0], gen_rate=1 / coef[1], prefill_rate=1 / coef[2], single_rate=1 / k), open(f'{S}/timing.json', 'w'))
