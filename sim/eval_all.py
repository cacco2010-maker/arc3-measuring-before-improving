"""配分の式の候補を、3 つの評価の軌跡（dp14 / dprime / t06）で突き合わせる。
dp14 で探索し、dprime・t06 は手を触れていない検証用。"""
import random, statistics, sys, json
from sim import *

SETS = {
    'arc3-dp14-v1': dict(slots=14, rate=59.2),
    'dprime-v1': dict(slots=10, rate=59.2 * 1.27),
    't06-v2': dict(slots=10, rate=59.2 * 1.27),
}
TR = {}
for name, cfg in SETS.items():
    TIMING['gen_rate'] = cfg['rate']; TR[name] = load_traces(name)
TIMING['gen_rate'] = 59.2


def ev(P, name, seeds=range(4), idx=None):
    trs = TR[name] if idx is None else [TR[name][i] for i in idx]
    return simulate_mean(trs, make_dprime(P), seeds=seeds, slots=SETS[name]['slots'])


def variant(**kw):
    P = dict(D_PRIME); P.update(kw); return P


cands = {'D′（今）': dict(D_PRIME)}
cands['token_scale=450k'] = variant(token_scale=450_000)
cands['lin_floor=0.3'] = variant(lin_floor=0.3)
cands['b0=16'] = variant(b0=16.0)
cands['B×0.25'] = variant(b1=2.5, b2=3.5, b3=4.0)
cands['pace_pow=1'] = variant(pace_pow=1.0)
cands['fade=1'] = variant(fade=1.0)
cands['b0=16+lin_floor=0.3'] = variant(b0=16.0, lin_floor=0.3)
cands['b0=16+token_scale=450k'] = variant(b0=16.0, token_scale=450_000)
cands['b0=16+lin_floor=0.3+ts=450k'] = variant(b0=16.0, lin_floor=0.3, token_scale=450_000)

# dp14 でランダム探索
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
results = []
for trial in range(N_TRIALS):
    P = dict(D_PRIME)
    P['level_pow'] = rng.choice([0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    P['eff_const'] = rng.choice([100.0, 300.0, 1000.0, 1e9])
    P['eff_pow'] = rng.choice([1.0, 2.5, 4.0])
    P['pace_pow'] = rng.choice([0.0, 0.2, 0.4, 0.8])
    P['token_scale'] = rng.choice([75_000, 150_000, 225_000, 450_000, 1e9])
    P['c_token'] = rng.choice([0.5, 0.9, 1.0]); P['c_action'] = 1.0 - P['c_token']
    P['lin_floor'] = rng.choice([0.0, 0.1, 0.3])
    bs = rng.choice([0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    P['b1'], P['b2'], P['b3'] = 10 * bs, 14 * bs, 16 * bs
    P['fade'] = rng.choice([0.0, 0.2, 0.4, 0.6, 1.0])
    P['b0'] = rng.choice([0.0, 0.0, 5.0, 10.0, 16.0])
    results.append((ev(P, 'arc3-dp14-v1'), P))
results.sort(key=lambda x: -x[0])
for i, (sc, P) in enumerate(results[:8]):
    cands[f'探索#{i+1}'] = P

base = {n: ev(D_PRIME, n) for n in SETS}
print('基準 D′: ' + '  '.join(f'{n} {v:.2f}' for n, v in base.items()))
# dp14 のゲーム再抽出（20 組）
by_game = defaultdict(list)
for i, t in enumerate(TR['arc3-dp14-v1']): by_game[t.game].append(i)
games = sorted(by_game); brng = random.Random(0)
sets = [[i for g in [brng.choice(games) for _ in games] for i in by_game[g]] for _ in range(20)]
base_boot = [ev(D_PRIME, 'arc3-dp14-v1', seeds=range(2), idx=idx) for idx in sets]

rows = []
for name, P in cands.items():
    d = {n: ev(P, n) - base[n] for n in SETS}
    boot = [ev(P, 'arc3-dp14-v1', seeds=range(2), idx=idx) for idx in sets]
    wins = sum(1 for a, b in zip(base_boot, boot) if b > a)
    rows.append((name, d, wins, P))
rows.sort(key=lambda r: -(r[1]['dprime-v1'] + r[1]['t06-v2']))
keys = ['level_pow', 'eff_const', 'eff_pow', 'pace_pow', 'token_scale', 'c_token', 'lin_floor', 'b0', 'b3', 'fade']
print(f'{"候補":34} {"dp14":>7} {"dprime":>7} {"t06":>7} {"検証計":>7} {"再抽出勝ち":>6}')
for name, d, wins, P in rows:
    print(f'{name:34} {d["arc3-dp14-v1"]:+7.2f} {d["dprime-v1"]:+7.2f} {d["t06-v2"]:+7.2f} {d["dprime-v1"]+d["t06-v2"]:+7.2f} {wins:>5}/20   '
          + ' '.join(f'{k}={P.get(k, 0.0):g}' for k in keys if P.get(k, 0.0) != D_PRIME.get(k, 0.0)))
json.dump([(n, d, w, P) for n, d, w, P in rows], open(f'{S}/eval_all.json', 'w'), indent=1)
