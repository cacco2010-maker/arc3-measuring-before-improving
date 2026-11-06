"""配分の式を再現器で比べる: D′ の変種、理想（未来を知る貪欲）、D′ の係数のランダム探索。"""
import random, statistics, sys, json, os
from sim import *

traces = load_traces()
by_game = defaultdict(list)
for i, t in enumerate(traces): by_game[t.game].append(i)
games = sorted(by_game)


def oracle(run, s, now, T):
    """未来を知る貪欲: 次の突破までの枠時間あたりの得点増。突破が無ければ最低。"""
    tr = run.tr
    src = run.src
    if run.stalled: return -1.0
    cost = 0.0
    for j in range(run.i, src.n):
        cost += src.cost[j]
        k = src.clear_after.get(j)
        if k is not None and k > run.L:
            gain = game_score(tr.N, k, run.apl, tr.base) - game_score(tr.N, run.L, run.apl, tr.base)
            return gain / max(1.0, cost)
    return -1.0


def evaluate(prio, idx=None, T=T_DEADLINE, slots=SLOTS, seeds=range(4)):
    trs = traces if idx is None else [traces[i] for i in idx]
    return simulate_mean(trs, prio, seeds=seeds, T=T, slots=slots)


def bootstrap_sets(n=30, seed=0):
    rng = random.Random(seed)
    sets = []
    for _ in range(n):
        gs = [rng.choice(games) for _ in games]
        idx = [i for g in gs for i in by_game[g]]
        sets.append(idx)
    return sets


if __name__ == '__main__':
    base = evaluate(make_dprime())
    print(f"D′ 基準 {base:.2f}")
    print(f"理想（未来を知る貪欲） {evaluate(oracle):.2f}   ← 配分で届き得る上限の目安")
    print(f"dfranzen 元の式 {evaluate(dfranzen_original):.2f}")
    variants = {}
    for bscale in (0.0, 0.25, 0.5, 2.0):
        P = dict(D_PRIME); P.update(b1=D_PRIME['b1'] * bscale, b2=D_PRIME['b2'] * bscale, b3=D_PRIME['b3'] * bscale)
        variants[f'B×{bscale}'] = P
    for fade in (0.0, 0.2, 0.6, 1.0):
        P = dict(D_PRIME); P['fade'] = fade; variants[f'fade={fade}'] = P
    for ts in (112_500, 450_000, 1e9):
        P = dict(D_PRIME); P['token_scale'] = ts; variants[f'token_scale={ts:g}'] = P
    for lp in (0.0, 1.0, 2.0):
        P = dict(D_PRIME); P['level_pow'] = lp; variants[f'level_pow={lp}'] = P
    for ec in (100.0, 1e9):
        P = dict(D_PRIME); P['eff_const'] = ec; variants[f'eff_const={ec:g}'] = P
    for pp in (0.0, 1.0):
        P = dict(D_PRIME); P['pace_pow'] = pp; variants[f'pace_pow={pp}'] = P
    for b0 in (5.0, 10.0, 16.0):
        P = dict(D_PRIME); P['b0'] = b0; variants[f'b0={b0}（最終レベルにも伸びしろ）'] = P
    for fl in (0.0, 0.3):
        P = dict(D_PRIME); P['lin_floor'] = fl; variants[f'lin_floor={fl}'] = P
    for name, P in variants.items():
        print(f'  {name:22} {evaluate(make_dprime(P)):.2f}  ({evaluate(make_dprime(P)) - base:+.2f})')

    # ランダム探索
    rng = random.Random(1)
    results = []
    N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
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
        sc = evaluate(make_dprime(P))
        results.append((sc, P))
    results.sort(key=lambda x: -x[0])
    print(f'\nランダム探索 {N_TRIALS} 件: 上位')
    sets = bootstrap_sets(20)
    base_boot = [evaluate(make_dprime(), idx) for idx in sets]
    for sc, P in results[:8]:
        boot = [evaluate(make_dprime(P), idx) for idx in sets]
        d = [b - a for a, b in zip(base_boot, boot)]
        wins = sum(1 for x in d if x > 0)
        keys = ['level_pow', 'eff_const', 'eff_pow', 'pace_pow', 'token_scale', 'c_token', 'lin_floor', 'b0', 'b3', 'fade']
        print(f'  {sc:.2f} ({sc-base:+.2f})  bootstrap Δ 中央値 {statistics.median(d):+.2f}  勝ち {wins}/20   ' + ' '.join(f'{k}={P[k]:g}' for k in keys))
    json.dump([(sc, P) for sc, P in results[:20]], open(f'{S}/search_top.json', 'w'), indent=1)
