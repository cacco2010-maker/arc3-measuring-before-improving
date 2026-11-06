# --- keepalive: 推論サーバーが上がるまで、起動前 RESET 役のゲームが 9 分ごとに RESET を打ち直す ---
# 背景（2026-10-22）: Kaggle のモデル読み込みが 40 分級に遅くなり、ゲートウェイは 15 分間
# 行動が無いとその時点の点（0.00）で走行を打ち切る（1 位 Jeroen Cottaar の説明）。
# dfranzen の仕組み（ARC3_WARMUP_ACTION_GAMES=1）は最初の 1 ゲームが RESET を 1 回打つだけ
# なので、読み込みが長引くと 15 分で切られる。ここでは、その RESET 役のゲームに、サーバーの
# /health が 200 を返すまで 9 分ごとに RESET を繰り返させる。
# 通常日（読み込み 11 分）はゲームが始まる前にサーバーが上がっているので、何も変わらない。
# RESET は dfranzen の起動前 RESET と同じ経路（_execute_auto_reset）で打つ。スコアカードは
# ハーネスが開いた 1 枚を共有し、別に開かない（本番のゲートウェイは 1 枚しか許さない）。
import threading as _ka_th, urllib.request as _ka_url, logging as _ka_log
import inference.framework.solver as _ka_solver

_KA_INTERVAL = float(os.environ.get('ARC3_KEEPALIVE_INTERVAL_SECONDS', 9 * 60))
_KA_HEALTH = f'http://{SERVED_MODEL_HOST}:{SERVED_MODEL_PORT}/health'
_ka_logger = _ka_log.getLogger('ours.keepalive')
KEEPALIVE = {'resets': 0, 'owner': None}


def _ka_healthy() -> bool:
    try:
        with _ka_url.urlopen(_KA_HEALTH, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _ka_loop(session, stop):
    expected = session.action_count  # 起動前 RESET を打った直後の手数
    while not stop.wait(_KA_INTERVAL):
        run = session.game.game_run
        if _ka_healthy():
            _ka_logger.warning('%s: keepalive stops, server is up', run.game_id)
            return
        if run is None or run.state != 'playing' or session.action_count != expected:
            return  # 本物の行動が始まった／ゲームが終わった
        try:
            session._execute_auto_reset()
            KEEPALIVE['resets'] += 1
            expected = session.action_count
            _ka_logger.warning('%s: keepalive RESET #%d (server still loading)', run.game_id, KEEPALIVE['resets'])
        except Exception as exc:
            _ka_logger.warning('keepalive RESET failed: %r', exc)
            return


_ka_orig_play_inner = _ka_solver._HarnessGameSession._play_inner


def _ka_play_inner(self):
    # 元の _play_inner より先に起動前 RESET の枠を取る。取れたゲームが keepalive 役。
    # 元の関数もこの後に枠を取ろうとするが、残りが 0 なので何もしない。
    if not self.solver._claim_warmup_slot():
        return _ka_orig_play_inner(self)
    run = self.game.game_run
    _ka_logger.warning('%s: warmup RESET issued before the analyzer is reachable (keepalive owner)', run.game_id)
    self._execute_auto_reset()
    KEEPALIVE['owner'] = run.game_id
    stop = _ka_th.Event()
    if not _ka_healthy():
        _ka_th.Thread(target=_ka_loop, args=(self, stop), name='ours-keepalive', daemon=True).start()
    try:
        return _ka_orig_play_inner(self)
    finally:
        stop.set()


_ka_solver._HarnessGameSession._play_inner = _ka_play_inner
print(f'#OURS_KEEPALIVE installed interval={_KA_INTERVAL:.0f}s health={_KA_HEALTH}', flush=True)
