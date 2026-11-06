"""_make_nb.py — m2-base（dfranzen そのまま）から、推論設定を変えた Notebook を作る。

    python3 _make_nb.py <名前> [--base 元] [--bench MIN] [--cfg KEY=VAL ...] [--env KEY=VAL ...] [--args "追加の sglang 引数"]

例:
    python3 _make_nb.py b-base   --bench 60
    python3 _make_nb.py b-s14hc  --bench 60 --cfg MAXREQ=14 CUDAGRAPH_MAXBS=14 --env ARC3_MAX_ACTIVE_STREAMS=14 \
                                 --args "--enable-hierarchical-cache --hicache-size 32"

- --base は元にする nb/<元>/（既定 m2-base）。dprime のようにセルの数が違ってもよい（セルは中身で探す）
- --cfg はセル 12 の CFG 辞書の値を上書きする（推論サーバーの設定）
- --env はセル 4 の setup_env の値を上書きする（ハーネスの設定）
- --args は sglang serve の引数の末尾に足す
- --bench MIN: 試走（非提出時）をデモ25本×4周・MIN 分の短い計測にする。指定しなければ本番と同じ持ち時間の評価（約8時間）
- --keepalive: 起動待ちの延命（10-23）。HTTP の最初の猶予を 900 秒→5400 秒にし、「起動前 RESET 役のゲームが
  サーバーが上がるまで 9 分ごとに RESET を打ち直す」セル（_keepalive_cell.py）をベンチマーク実行セルの直前に挿む。
  本番でも効く（Kaggle のモデル読み込みが 40 分級に遅い日に 0.00 で切られないため）
- --no-trial: 試走を飛ばす（10-25）。保存のための実行（Save & Run）では先頭のセルが dummy の submission.parquet を書き、
  以降のセルをすべて no-op にして数十秒で終える。本番の再実行（KAGGLE_IS_COMPETITION_RERUN）では何もせず全セルがそのまま走る。
  GPU 機でデータセットが載らず試走が通らない間に、提出に使える版を保存するための措置。Kaggle 上の名前は arcb-<名前>（評価に数えない）
- 本番（提出時）の分岐は --cfg/--env/--args/--keepalive だけが効き、それ以外は dfranzen のまま

出力は nb/<名前>/。Kaggle 上の名前は、計測（--bench）なら kazuhiro3381/arcb-<名前>、評価なら kazuhiro3381/arc3-<名前>。
計測を arcb- にするのは、_daily_submit.py の「週 1 回の評価」の数え方（arc3- だけ数える）に入れないため。
"""
import argparse, json, os, re

ROOT = os.path.expanduser('~/arc-agi-3')
p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('--base', default='m2-base')
p.add_argument('--bench', type=int)
p.add_argument('--cfg', nargs='*', default=[])
p.add_argument('--env', nargs='*', default=[])
p.add_argument('--args', default='')
p.add_argument('--keepalive', action='store_true')
p.add_argument('--no-trial', action='store_true')
a = p.parse_args()

nb = json.load(open(f'{ROOT}/nb/{a.base}/{a.base}.ipynb'))
src = lambda i: ''.join(nb['cells'][i]['source'])
def put(i, s): nb['cells'][i]['source'] = s.splitlines(True)
def find(key):
    hit = [i for i in range(len(nb['cells'])) if key in src(i)]
    assert len(hit) == 1, f'{key!r} のセルが {len(hit)} 個'
    return hit[0]
C4, C12, C16 = find('os.environ.update({k: str(v) for k,v in setup_env.items()})'), find('graph_bs = sorted('), find('bm.solver.max_runtime_s_per_game =')

# セル 12: CFG の上書きと追加引数
c12 = src(C12)
for kv in a.cfg:
    k, v = kv.split('=', 1)
    c12, n = re.subn(rf'(\n    {k}=)[^,\n]+,', rf'\g<1>{v},', c12)
    assert n == 1, f'CFG に {k} が見つからない'
graph_old = 'graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})'
graph_new = 'graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, 12, 14, 16, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})'
assert graph_old in c12 or graph_new in c12  # _make_nb.py で作った元（dp14 など）はすでに置き換え済み
c12 = c12.replace(graph_old, graph_new)
if a.args:
    anchor = 'if CFG["PREFETCH_CHECKPOINTS"]:'
    assert anchor in c12
    c12 = c12.replace(anchor, f'args += {a.args.split()!r}  # _make_nb.py で追加\n' + anchor, 1)
put(C12, c12)

# セル 4: setup_env の上書き（priority 側の update も含めて、最後に効かせる）
c4 = src(C4)
if a.env:
    extra = {k: v for k, v in (kv.split('=', 1) for kv in a.env)}
    anchor = "os.environ.update({k: str(v) for k,v in setup_env.items()})"
    assert anchor in c4
    c4 = c4.replace(anchor, f'setup_env.update({extra!r})  # _make_nb.py で上書き\n' + anchor)
put(C4, c4)

# セル 16: 試走の持ち時間
c16 = src(C16)
old = next(l.strip() for l in c16.splitlines() if l.strip().startswith('bm.solver.max_runtime_s_per_game =') and ('25*60' in l or 'concurrency // 110' in l))
if a.bench:
    new = f"bm.solver.max_runtime_s_per_game = {a.bench}*60  # 計測: デモ25本×4周を {a.bench} 分だけ回す"
else:
    new = "bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110  # 本番と同じ持ち時間でデモ25本×4周を評価"
put(C16, c16.replace(old, new))

# --keepalive: 起動待ちの延命（セル 4 の猶予を伸ばし、ベンチマーク実行セルの直前に keepalive のセルを挿む）
if a.keepalive:
    c4 = src(C4)
    old = "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '900'"
    assert c4.count(old) == 1, 'ARC3_HTTP_RETRY_INITIAL_SECONDS の行が 1 つでない'
    c4 = c4.replace(old, "os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '5400'  # _make_nb.py --keepalive: 起動待ちを 90 分まで")
    # 入力の取り付け待ち（10-24: GPU 機で cp が「cannot stat .../taaf-kaggle-source-bundle-copy」で 4 回連続失敗。CPU 機では載っていた）
    anchor = '# Apply harness patch\n'
    assert c4.count(anchor) == 1, '「# Apply harness patch」が 1 つでない'
    wait_src = '''# _make_nb.py --keepalive: 入力が載るまで待つ（GPU 機ではデータセットが遅れて載る・載らないことがある。10-24 に 4 回連続で cp が失敗）
def _wait_inputs(paths, timeout_s=20 * 60):
    t0 = time.time()
    missing = [p for p in paths if not os.path.exists(p)]
    while missing and time.time() - t0 < timeout_s:
        print(f"  waiting for inputs ({int(time.time() - t0)}s): {missing}", flush=True)
        time.sleep(10)
        missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"inputs STILL MISSING after {int(time.time() - t0)}s: {missing}", flush=True)
    else:
        print(f"inputs ready after {int(time.time() - t0)}s", flush=True)

_wait_inputs([ORIG_BUNDLE_DIR, WHEELHOUSE_DIR, MODEL_DIR, DRAFT_MODEL_DIR])

'''
    put(C4, c4.replace(anchor, wait_src + anchor))
    assert not any('#OURS_KEEPALIVE' in src(i) for i in range(len(nb['cells']))), '元の Notebook にすでに keepalive がある'
    CB = find("print('Starting benchmark...')")
    ka_src = open(f'{ROOT}/_keepalive_cell.py').read()
    nb['cells'].insert(CB, {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                            'source': ka_src.splitlines(True)})

# --no-trial: 先頭（%%writefile の直後、環境セルの前）に「試走を飛ばす」セルを挿む
if a.no_trial:
    assert not a.bench, '--no-trial と --bench は同時に使えない'
    assert not any('#OURS_NO_TRIAL' in src(i) for i in range(len(nb['cells']))), '元の Notebook にすでに no-trial がある'
    nt_src = open(f'{ROOT}/_no_trial_cell.py').read()
    nb['cells'].insert(C4, {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                            'source': nt_src.splitlines(True)})

desc = [f'## {a.name}', '', f'dfranzen Milestone 2 解法の写し（{a.base}）からの変更:']
desc += [f'- CFG {kv}' for kv in a.cfg] + [f'- env {kv}' for kv in a.env]
if a.args: desc.append(f'- sglang 引数 {a.args}')
if a.no_trial: desc.append('- no-trial: 保存のための実行では試走を飛ばす（dummy の parquet だけ書く）。本番の再実行では全セルが走る')
if a.keepalive: desc.append('- keepalive: 起動前 RESET 役のゲームがサーバーが上がるまで 9 分ごとに RESET を打ち直す。HTTP の最初の猶予 900→5400 秒（本番でも効く）')
desc.append(f'- 試走: デモ25本×4周を {a.bench} 分（推論の速さの計測）' if a.bench else '- 試走: デモ25本×4周・本番と同じ持ち時間（基準 46.49）')
put(0, '\n'.join(desc) + '\n\n' + src(0))

kid = f"kazuhiro3381/{'arcb' if (a.bench or a.no_trial) else 'arc3'}-{a.name}"
out = f'{ROOT}/nb/{a.name}'
os.makedirs(out, exist_ok=True)
json.dump(nb, open(f'{out}/{a.name}.ipynb', 'w'), ensure_ascii=False, indent=1)
m = json.load(open(f'{ROOT}/nb/{a.base}/kernel-metadata.json'))
m.pop('id_no', None)  # 基の kernel の id_no を引き継ぐと、その kernel が改名されて版が上書きされる（2026-11-01 sumtrimA/B で発生）
m.update(id=kid, title=kid.split('/')[1], code_file=f'{a.name}.ipynb')
json.dump(m, open(f'{out}/kernel-metadata.json', 'w'), indent=2, ensure_ascii=False)
print(out, '→', kid)
