# --- no-trial: 保存のための実行では試走を飛ばす（_make_nb.py --no-trial、2026-10-25） ---
# GPU 機でデータセット（ソース束・wheelhouse）が載らず、Save & Run の試走が通らない間の措置。
# 本番の再実行（KAGGLE_IS_COMPETITION_RERUN=true）では何もせず、以降のセルがそのまま走る。
# それ以外（Save & Run）では dummy の submission.parquet を書き、以降のセルをすべて no-op に置き換える
# （IPython の入力変換で、各セルの中身を print 1 行に差し替える。%%writefile などのセルマジックも含めて飛ぶ）。
import os as _nt_os
_NT_RERUN = _nt_os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
if _NT_RERUN:
    print("#OURS_NO_TRIAL competition rerun: running the full notebook", flush=True)
else:
    import pandas as _nt_pd
    from pathlib import Path as _nt_Path
    _nt_Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    _nt_pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
        "/kaggle/working/submission.parquet", index=False)
    get_ipython().input_transformers_cleanup.append(
        lambda lines: ['print("skipped: no-trial version, saved for submission only")\n'])
    print("#OURS_NO_TRIAL wrote dummy submission.parquet; the remaining cells are skipped", flush=True)
