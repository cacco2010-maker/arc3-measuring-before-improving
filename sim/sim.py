"""dp14 の 100 本の軌跡を使い、枠（14 本）の配分だけを差し替えて点数を再計算する再現器。

軌跡 = リクエスト列（生成トークン、未キャッシュ入力、文脈の削り＝枠の受け渡し点）とレベル突破の位置。
ゲームの進み方は「それまでに使ったトークン・手数」だけで決まり、壁時計には依らないと仮定する。

軌跡が尽きて勝っていない本（締め切りで切られた本）に時間を回した場合の続きは、
**同じゲームの別の周**で「同じレベルから先」の軌跡を継ぎ足す（splice=True）。
そのレベルを越えた周が無ければ、以後は平均の歩調でトークンを使い続けるが突破しない。
splice=False なら継ぎ足し無し（実際に起きたことの範囲での再現。得点は実際を超えない）。
"""
import heapq, json, math, os, pickle, random, statistics
from collections import defaultdict

S = os.path.dirname(os.path.abspath(__file__))
TIMING = json.load(open(f'{S}/timing.json'))
T_DEADLINE = 532 * 60 * 100 // 110  # 29018 秒（評価）。本番は 532*60
SLOTS = 14


def game_score(N, L, apl, base):
    tot = 0.0; wsum = 0; wmax = 0
    for k in range(1, N + 1):
        wsum += k
        if k <= L:
            a = apl[k - 1]
            s = min(115.0, (base[k - 1] / a) ** 2 * 100) if a > 0 else 0.0
            tot += s * k
            if s > 0: wmax += k
    return min(tot / wsum, wmax / wsum * 100)


class Trace:
    """1 本ぶんのリクエスト列。"""
    def __init__(self, r):
        self.game, self.pas, self.N = r['game'], r['pas'], r['N']
        self.apl, self.base = r['apl'], r['base']
        self.won = r['state'] == 'won'
        self.L_real = r['L']
        req = r['req']
        self.cost = []; self.trim_before = []; self.gen = []; self.act = []
        prev = None
        for (act, pt, cached, comp) in req:
            self.cost.append(TIMING['a0'] + comp / TIMING['gen_rate'] + max(0, pt - cached) / TIMING['prefill_rate'])
            self.trim_before.append(prev is not None and prev - pt > 30000)
            self.gen.append(comp); self.act.append(act)
            prev = pt
        self.n = len(self.cost)
        # レベル k の突破は「action < clears[k] の最後のリクエスト」の完了時
        self.clear_after = {}  # request index -> level cleared at its completion
        self.level_start_req = {1: 0}
        for k, c in enumerate(r['clears'], start=1):
            idx = max(i for i, a in enumerate(self.act) if a < c) if any(a < c for a in self.act) else 0
            self.clear_after[idx] = k
            self.level_start_req[k + 1] = min(idx + 1, self.n - 1)
        self.mean_cost = sum(self.cost) / self.n
        self.mean_gen = sum(self.gen) / self.n
        ntrim = sum(self.trim_before)
        self.trim_every = max(5, self.n // (ntrim + 1))
        self.real_clear_t = [r['wall'][c - 1] for c in r['clears']]
        self.real_end = r['final_wall']


class Run:
    def __init__(self, tr: Trace, dispatch_index, siblings, rng, splice):
        self.tr = tr; self.dispatch = dispatch_index
        self.src = tr; self.i = 0
        self.siblings = siblings; self.rng = rng; self.splice = splice
        self.used = {tr.pas}
        self.level = 1; self.L = 0
        self.apl = list(tr.apl)
        self.actions_at_level = 0; self.tokens_at_level = 0.0
        self.cum_tokens = 0.0
        self.pace_completed = 0; self.pace_tokens = 0.0
        self.done = False; self.stalled = False
        self.clear_t = []; self.synthetic = 0; self.spliced = 0

    @property
    def pace(self):
        return self.pace_tokens / self.pace_completed if self.pace_completed else 0.0

    def _act(self):
        if self.stalled: return self.src.act[-1] + self.synthetic
        return self.src.act[min(self.i, self.src.n - 1)]

    def snapshot(self):
        return dict(level=self.level, actions=max(0, self._act() - self.actions_at_level),
                    tokens=max(0.0, self.cum_tokens - self.tokens_at_level), pace=self.pace, N=self.tr.N)

    def next_cost(self):
        if self.stalled: return self.src.mean_cost
        return self.src.cost[self.i]

    def trim_now(self):
        if self.stalled: return self.synthetic > 0 and self.synthetic % self.src.trim_every == 0
        return self.src.trim_before[self.i]

    def _exhausted(self):
        """自分の軌跡が尽きた。勝ちなら終了、そうでなければ別の周を継ぎ足す。"""
        if self.src.won:
            self.done = True; return
        if self.splice and self.spliced == 0:
            # そのレベルに到達した別の周（越えられなかった周も含む。越えた割合がそのまま見込みになる）
            cands = [s for s in self.siblings if s.pas not in self.used and s.L_real >= self.L
                     and (self.L + 1) in s.level_start_req]
            if cands:
                s = self.rng.choice(cands)
                self.used.add(s.pas); self.spliced += 1
                self.src = s; self.i = s.level_start_req[self.L + 1]
                self.actions_at_level = s.act[self.i]
                self.tokens_at_level = self.cum_tokens
                for k in range(self.L + 1, s.N + 1):
                    if k - 1 < len(s.apl): self.apl[k - 1] = s.apl[k - 1]
                return
        self.stalled = True

    def complete(self, now):
        """今のリクエストが終わった。突破・レベル更新をして、終了なら True。"""
        if self.stalled:
            self.cum_tokens += self.src.mean_gen; self.synthetic += 1
            return False
        self.cum_tokens += self.src.gen[self.i]
        k = self.src.clear_after.get(self.i)
        if k is not None and k > self.L:
            used = self.cum_tokens - self.tokens_at_level
            self.pace_completed += 1; self.pace_tokens += used
            self.L = k; self.level = k + 1
            self.tokens_at_level = self.cum_tokens
            self.actions_at_level = self.src.act[self.i + 1] if self.i + 1 < self.src.n else self.src.act[-1]
            self.clear_t.append(now)
            if self.L >= self.tr.N:
                self.done = True; return True
        self.i += 1
        if self.i >= self.src.n:
            self._exhausted()
        return self.done

    def score(self):
        return game_score(self.tr.N, self.L, self.apl, self.tr.base)


# ---------------- 優先度の式 ----------------
D_PRIME = {"level_pow": 0.5, "eff_const": 300.0, "eff_pow": 2.5, "pace_pow": 0.4,
           "pace_ref": 30_000.0, "c_action": 0.1, "action_scale": 115.0, "c_token": 0.9,
           "token_scale": 225_000.0, "lin_floor": 0.1, "rel_pow": 0.5,
           "b1": 10.0, "b2": 14.0, "b3": 16.0, "fade": 0.4}


def d_prime_value(s, phi, P=D_PRIME):
    level = max(1, s['level']); n = min(10, max(6, s['N']))
    norm = 55 / (n * (n + 1) / 2)
    a, t, pace = s['actions'], s['tokens'], s['pace']
    A = (1.0 + P["level_pow"] * (level - 1)) * norm
    A *= (P["eff_const"] / (P["eff_const"] + a)) ** P["eff_pow"]
    if pace > 0:
        A *= min(4.0, max(0.25, (P["pace_ref"] / max(1.0, pace)) ** P["pace_pow"]))
    scale = P["token_scale"]
    rel = pace if pace > 0 else P["pace_ref"]
    scale *= min(2.0, max(0.5, rel / P["pace_ref"])) ** P["rel_pow"]
    fl = P["lin_floor"]
    C = P["c_action"] * max(fl, 1.0 - a / P["action_scale"]) + P["c_token"] * max(fl, 1.0 - t / scale)
    remaining = max(0, s['N'] - level)
    B = (P.get("b0", 0.0), P["b1"], P["b2"], P["b3"])[min(remaining, 3)]
    return A * C + B * phi


def make_dprime(P=D_PRIME):
    fade = P.get('fade', 0.4)
    def prio(run, s, now, T):
        phi = 1.0
        if fade and fade > 0:
            window = fade * T
            phi = min(1.0, max(0.0, (T - now) / window))
        return d_prime_value(s, phi, P)
    return prio


def dfranzen_original(run, s, now, T):
    """彼の priority_value（normalize + tail 'remaining' + fade 0.2）。未開始は先頭の帯。"""
    if run.i == 0 and run.src is run.tr:
        return 2_000_000 - run.dispatch
    level = max(1, s['level']); n = min(10, max(6, s['N']))
    a, t = s['actions'], s['tokens']
    pace = 1.0
    eff = (25.0 / (25.0 + a)) ** 2
    cur = level * eff * 55.0 / (n * (n + 1) / 2)
    tail = ((8.0,) * (n - 3) + (7.0, 5.0, 0.0))[min(n, level) - 1]
    window = 0.2 * T
    tail *= min(1.0, max(0.0, (T - now) / window))
    chance = 0.25 * 0.5 ** ((a / 115.0) ** 2) + 0.75 * 0.5 ** ((t / (62000.0 * pace)) ** 2)
    return max(1, int((cur + tail) * chance / pace * 100))


# ---------------- 再現 ----------------
def simulate(traces, prio_fn, T=T_DEADLINE, slots=SLOTS, splice=True, seed=0):
    rng = random.Random(seed)
    by_game = defaultdict(list)
    for tr in traces: by_game[tr.game].append(tr)
    runs = [Run(tr, i, [s for s in by_game[tr.game] if s is not tr], rng, splice) for i, tr in enumerate(traces)]
    free = slots
    waiting = []
    events = []
    seq = 0
    now = 0.0

    def pump():
        nonlocal free, seq
        if free <= 0 or not waiting: return
        scored = sorted(((-prio_fn(run, run.snapshot(), now, T), run.dispatch, run) for run in waiting),
                        key=lambda x: (x[0], x[1]))
        while free > 0 and scored:
            _, _, run = scored.pop(0)
            waiting.remove(run); free -= 1
            seq += 1
            heapq.heappush(events, (now + run.next_cost(), seq, run))

    for run in runs: waiting.append(run)
    pump()
    while events:
        now, _, run = heapq.heappop(events)
        if run.complete(now):
            free += 1; pump(); continue
        if now >= T:                      # 持ち時間切れ
            run.done = True; free += 1; pump(); continue
        if run.trim_now():                # 文脈の削り → 枠の受け渡し
            free += 1; waiting.append(run); pump()
        else:
            seq += 1
            heapq.heappush(events, (now + run.next_cost(), seq, run))
    return statistics.mean(r.score() for r in runs), runs


def simulate_mean(traces, prio_fn, seeds=range(6), **kw):
    return statistics.mean(simulate(traces, prio_fn, seed=s, **kw)[0] for s in seeds)


def load_traces(name='arc3-dp14-v1'):
    data = pickle.load(open(f'{S}/dataset.pkl', 'rb'))[name]
    return [Trace(r) for r in sorted(data, key=lambda r: (r['game'], r['pas']))]


if __name__ == '__main__':
    traces = load_traces()
    real = statistics.mean(game_score(t.N, t.L_real, t.apl, t.base) for t in traces)
    print(f'実際の平均 {real:.2f}')
    sc, runs = simulate(traces, make_dprime(), splice=False)
    print(f"再現 D′（継ぎ足し無し）: {sc:.2f}   レベル数一致 {sum(1 for r in runs if r.L == r.tr.L_real)}/100")
    for seed in range(3):
        sc, runs = simulate(traces, make_dprime(), splice=True, seed=seed)
        n_sp = sum(1 for r in runs if r.spliced)
        gained = sum(1 for r in runs if r.L > r.tr.L_real)
        print(f"再現 D′（継ぎ足し、seed {seed}）: {sc:.2f}   継ぎ足した本 {n_sp}、実際より進んだ本 {gained}")
    print(f"再現 dfranzen 元の式（継ぎ足し）: {simulate_mean(traces, dfranzen_original):.2f}")
