# Measuring Before Improving

Tools and data behind the ARC Prize 2026 paper-track write-up *Measuring Before Improving: variance,
compute allocation and cache costs in an LLM harness for ARC-AGI-3*.

- Paper: [paper/paper.pdf](paper/paper.pdf) (long version); [paper/writeup.md](paper/writeup.md) (1,500-word version)
- Submitted notebook: [arc3-dp14, version 1](https://www.kaggle.com/code/kazuhiro3381/arc3-dp14?scriptVersionId=359191772) (versions 2 and 3 are a different agent)
- Replacing-summary evaluation: [arc3-sumtrimB, version 3](https://www.kaggle.com/code/kazuhiro3381/arc3-sumtrimb)

## Contents

| Path | What it is |
|---|---|
| `sim/` | Offline replay simulator for the harness's scheduler (paper section 6.1) |
| `sim/dataset.pkl` | Trajectories of three evaluations: plain lists, dicts, numbers and game names. Load pickles only from sources you trust |
| `analysis/memory_cost.py` | Sections 5.2 and 6.4: noise across evaluations, cost of memory designs, re-grounding counts |
| `analysis/budget_curve.py` | Section 6.5: score against compute budget, and the memory designs at matched tokens and actions |
| `analysis/event_study.py` | Section 6.4: requests around a context drop, by how much history survives. Uses `data/memory_trajectories.pkl` (numbers only) |
| `analysis/wallclock_check.py` | Section 6.1: tests the replay's assumption that a level's cost does not depend on when it is played |
| `data/summaries/` | Per-game scores of four evaluations |
| `notebook_tools/` | Variant generator and keep-alive cells. They edit a copy of the base notebook, which is not included. Comments are in Japanese |
| `paper/` | Paper sources and figures |

## Reproduce

Python 3.12 with numpy and scipy.

```bash
python sim/sim.py
python sim/eval_all.py
python analysis/memory_cost.py
python analysis/budget_curve.py
python analysis/wallclock_check.py
python analysis/event_study.py
```

`sim/sim.py` replays the scheduler (D' gives 50.43; the real run scored 50.46). `sim/search.py` includes the future-knowing oracle.
`analysis/memory_cost.py` with no options gives the score comparison. For the token and re-grounding tables, download
the output of arc3-sumtrimB and pass it as `--replacing DIR`, and pass the appended-summary evaluation as `--appended DIR`.

## Known limits

- The raw logs of `dp14` no longer exist. Its request sequence survives in `sim/dataset.pkl`, its text does not.
- `dprime-v1` and `t06-v2` kept no request logs, so their per-request timing in `sim/dataset.pkl` is approximated.
- `sim/build_dataset.py` and `sim/timing.py` need the raw evaluation outputs and will not run from this repository alone.

## License

CC0 1.0, except the third-party formulas listed in [NOTICE](NOTICE).
