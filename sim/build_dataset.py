"""評価 Notebook の出力（benchmark.json / requests.jsonl）から、1 本ごとの軌跡を取り出す。

軌跡 = 行動ごとの (累積壁時計, 累積生成トークン)、レベル突破の行動番号、
文脈の削り（trim; 枠の受け渡し点）の行動番号。採点は arc_agi 0.9.8 scorecard.py と同じ式。
"""
import json, glob, os, pickle, sys
from collections import defaultdict

RUNS = os.path.expanduser('~/arc-agi-3/runs')
OUT = os.path.join(os.path.dirname(__file__), 'dataset.pkl')


def game_score(N, L, actions_per_level, base, won):
    """arc_agi scorecard: 重み k、各レベル min(115,(base/a)^2*100)、上限は突破した重みの割合。"""
    tot = 0.0; wsum = 0; wmax = 0
    for k in range(1, N + 1):
        w = k; wsum += w
        if k <= L:
            a = actions_per_level[k - 1]
            s = min(115.0, (base[k - 1] / a) ** 2 * 100) if a > 0 else 0.0
            tot += s * w
            if s > 0: wmax += w
    return min(tot / wsum, wmax / wsum * 100)


def load_run_dir(d):
    b = json.load(open(os.path.join(d, 'benchmark.json')))
    runs = []
    per_game_pass = defaultdict(int)
    for r in b['game_runs']:
        g = r['game_id']; p = per_game_pass[g]; per_game_pass[g] += 1
        hist = r['history']
        wall = [h['wallclock_seconds'] for h in hist]
        tok = []; c = 0
        for h in hist:
            c += h['generated_tokens']; tok.append(c)
        apl = r['actions_per_level']; L = r['levels_completed']; N = r['number_of_levels']
        clears = []; cum = 0
        for k in range(L):
            cum += apl[k]; clears.append(cum)  # 行動 cum 回目でレベル k+1 突破（1 始まり）
        sc = game_score(N, L, apl, r['base_actions_per_level'], r['state'] == 'won')
        assert abs(sc - r['final_score']) < 0.05, (g, p, sc, r['final_score'])
        # trim: requests.jsonl で prompt_tokens が 3 万以上落ちた所
        trims = []
        f = glob.glob(os.path.join(d, f'{g}_p{p}_requests.jsonl'))
        req = []
        if f:
            prev = None
            for line in open(f[0]):
                x = json.loads(line)
                if x.get('event') != 'response': continue
                u = x['usage']; pt = u['prompt_tokens']
                act = int(x['action'])
                req.append((act, pt, u['prompt_tokens_details'].get('cached_tokens', 0), u['completion_tokens']))
                if prev is not None and prev - pt > 30000:
                    trims.append(act)
                prev = pt
        if not req:
            # requests.jsonl が無い評価: 行動ごとの生成トークンから近似のリクエスト列を作る
            # （trim は生成 46.8k トークンごと、未キャッシュ入力は trim 時 51k・通常 4.3k）
            SY = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synth.json')))
            since = 0.0
            for i, h in enumerate(hist):
                comp = h['generated_tokens']
                if comp <= 0: continue
                if since >= SY['gen_between_trims']:
                    pt = 60000; unc = SY['unc_trim']; since = 0.0
                else:
                    pt = 120000; unc = SY['unc_normal']
                req.append((i, pt, pt - unc, comp))
                since += comp
            if not req: req.append((0, 120000, 120000 - SY['unc_normal'], 1))
            trims = [a for (a, pt, c, comp) in req if pt == 60000]
        runs.append(dict(game=g, pas=p, N=N, L=L, apl=apl, base=r['base_actions_per_level'],
                         state=r['state'], score=r['final_score'], wall=wall, tok=tok,
                         clears=clears, trims=trims, req=req, final_wall=r['final_wallclock_seconds'],
                         n_actions=len(hist)))
    return runs, b


if __name__ == '__main__':
    data = {}
    for name in ['arc3-dp14-v1', 'dprime-v1', 't06-v2']:
        runs, b = load_run_dir(os.path.join(RUNS, name))
        data[name] = runs
        print(f'== {name}: {len(runs)} runs, mean {sum(r["score"] for r in runs)/len(runs):.2f}')
    pickle.dump(data, open(OUT, 'wb'))

    runs = data['arc3-dp14-v1']
    T = 29018  # 持ち時間（秒）
    print('\n--- dp14: 1 本ごと（突破時刻は秒、waste = 最後の突破のあとに使ったトークン）')
    print(f'{"game":5}{"p":>2}{"N":>3}{"L":>3}{"score":>7}{"tok(k)":>8}{"waste(k)":>9}{"lastclear":>10}{"end":>7}{"acts":>5}{"trims":>6}  clears(s)')
    agg = defaultdict(lambda: dict(score=0, tok=0, waste=0, n=0, late=0))
    for r in sorted(runs, key=lambda r: (r['game'], r['pas'])):
        last_clear_i = r['clears'][-1] - 1 if r['clears'] else None
        last_clear_t = r['wall'][last_clear_i] if last_clear_i is not None else 0
        waste = r['tok'][-1] - (r['tok'][last_clear_i] if last_clear_i is not None else 0)
        clear_ts = [int(r['wall'][i - 1]) for i in r['clears']]
        a = agg[r['game']]; a['score'] += r['score']; a['tok'] += r['tok'][-1]; a['waste'] += waste; a['n'] += 1
        if r['state'] != 'won' and last_clear_t > 0.8 * T: a['late'] += 1
        print(f'{r["game"][:4]:5}{r["pas"]:>2}{r["N"]:>3}{r["L"]:>3}{r["score"]:>7.1f}{r["tok"][-1]/1000:>8.0f}{waste/1000:>9.0f}{last_clear_t:>10.0f}{r["final_wall"]:>7.0f}{r["n_actions"]:>5}{len(r["trims"]):>6}  {clear_ts}')
    print('\n--- ゲームごと（4 周平均）。late = 勝たずに持ち時間の最後 2 割でまだ突破していた本数')
    tot_tok = sum(a['tok'] for a in agg.values())
    for g, a in sorted(agg.items(), key=lambda kv: kv[1]['score'] / kv[1]['n']):
        print(f'{g[:4]:5} score {a["score"]/a["n"]:6.1f}  tok {a["tok"]/1000/a["n"]:6.0f}k ({a["tok"]/tot_tok*100:4.1f}%)  waste {a["waste"]/1000/a["n"]:6.0f}k ({a["waste"]/a["tok"]*100:3.0f}%)  late {a["late"]}')
    tot_waste = sum(a['waste'] for a in agg.values())
    print(f'\n合計トークン {tot_tok/1e6:.2f}M、最後の突破より後 {tot_waste/1e6:.2f}M ({tot_waste/tot_tok*100:.0f}%)')
