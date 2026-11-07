# Measuring Before Improving

### Variance, compute allocation and cache costs in an LLM harness for ARC-AGI-3

<!-- Kaggle Writeup body (limit 1,500 words; count with `python3 paper/wordcount.py`). The long version with all
tables is paper.md, attached as a PDF via the Project Link. 
Cover image (Media Gallery, required): paper/fig/cover.png, 1600x800, made by paper/fig/cover.py. -->

**Submission:** ARC-AGI-3 submission 57374366 ([notebook `arc3-dp14`, version 1](https://www.kaggle.com/code/kazuhiro3381/arc3-dp14?scriptVersionId=359191772)), public LB 33.96 (best of six scored runs; mean 24.63). **Long version and tools:** [repository](https://github.com/cacco2010-maker/arc3-measuring-before-improving).

## Summary

Most ARC-AGI-3 entries fork one public harness, and the leaderboard cannot separate them: our unchanged copy scored 21.8 to 29.9 over 15 identical runs. We measured how large an improvement must be to be visible, then built tools for smaller effects. What is new:

1. **Detectable effect:** ≈5 points between two leaderboard runs or two 8-hour local evaluations. Another team's 17 runs fit the same noise model (§1).
2. **A replay simulator** reproduces a real run to within 0.03 points (0.15 with stitching). Scheduler tunings worth +3 to +4 on their own trajectories are worth 0 or less on held-out ones (§2).
3. **What memory costs:** appended summaries doubled prefill. A prefix-stable design removed that cost, yet the score did not recover: our prediction failed. Memory has two prices, prefill and re-checking, and re-checking follows lost evidence (§4).
4. **A price for compute:** ≈0.2 points per 1% on public games, which turns mechanism measurements into predicted scores. Top entries confirm the direction (§4).
5. **A platform failure** scoring 0.00 on four of ten submissions (§5).

## Setting

The base is dfranzen's Milestone-2 solution: Tufa Labs' Duck harness (the agent loop) with Qwen3.8-Flash-Next (4-bit) on one RTX Pro 6000. An LLM inspects frames through a Python tool and chooses actions. A priority scheduler decides which of ≈110 concurrent games gets the next LLM call. A full context is trimmed from ≈115k to ≈56k tokens.

Our submission (`dp14`) changes two things: shiiin9's D′ priority formula, and 14 concurrent LLM requests, or streams (was 10), with SGLang's KV cache spilling into 32 GB of host memory.

**Local evaluation** ("demo 25×4") plays the 25 public games 4 times each with the real time budget and official scorer. One costs 8 GPU-hours.

## 1. How big must an improvement be?

**Leaderboard.** Same-code reruns have an SD of 2.4 (our 15 runs; concurrent entries report 2.1–2.9), so two single runs must differ by ≈5 points for 90% one-sided confidence (1.28 × √2 × SD). Public copies spread more (3.93, Son Pham).

**Local.** Across the 4 passes of a game, the SD averages **16.5 points** (RMS 22). The standard error of a 25×4 mean is ±2.2, and of a difference between two evaluations **±3.1**. (5 points = 1.645 × 3.1, 95% one-sided).

**One evaluation resolves only about 5 points, and most published tweaks are worth 1–2.**

**All six evaluations (43.96–50.46; five ours, the base dfranzen's own) have an SD of 2.48, matching the ±2.2 noise (χ² p = 0.28): no change we tried is distinguishable from the base.**

**Independent check.** We applied this model to tantan0327's public runs (17 runs of the base and variants; 10 games, 25 minutes): RMS 20.0 predicts an SD of 6.3; observed 6.6 (χ² p = 0.35). Their variants are indistinguishable. yy5652 report RMS 21.4.

![Figure 1](fig/lb_dist.png)

*Figure 1. Public LB scores of identical code. dp14 averages 24.63 over six scored runs (27.85 without an unexplained 8.53), the base 26.18.*

## 2. Replaying the scheduler offline

**Simulator.** A trajectory is a run's LLM requests (generated tokens, uncached input tokens, trims) and its level clears. We assume a game's progress depends on the tokens and actions it consumes, not on wall-clock time; a level's token cost indeed does not vary with start time (+0.5% per hour, 95% CI −3.3 to +4.2). Request time fits t = 5.2 s + gen/59 + uncached/2100 (R² = 0.97). A run that the replay carries past its recorded end is **stitched**: it continues on another pass of the same game, from the same level. Passes that failed there are included.

**Validation.** Replaying D′ on the same run gives 50.43 against an actual 50.46; 99 of 100 runs reach the same level.

**Findings.**
- An oracle that sees each trajectory's future reaches 53.33 against D′'s 50.61: perfect allocation is worth about +2.7.
- On the trajectories they were fitted to, the best scheduler settings gain **+3 to +4**, in 18 of 20 bootstrap resamples over games.
- **On trajectories from two other evaluations (timing approximated), the best setting gains −2.6 and −1.3; on tantan0327's 158 trajectories, −1.8 to +0.9.**

With 100 trajectories, a scheduler fits their noise. We rejected every allocation change. tantan0327's simulator finds an oracle worth +2 over D′ and searched gains shrinking on held-out seeds; our test holds out real evaluations.

## 3. Throughput as a proxy with little noise

Generated tokens per second barely varies between runs and takes 60 minutes, so we screened servers on it: 14 streams with a host cache gave **+10.5%** (563 to 622 tok/s, 60-minute bench). 14 streams with D′ scored **50.46** (base 46.49), but that is the best of six, inside §1's noise and invisible on the leaderboard.

## 4. What summarization actually costs

Trims discard ≈59k tokens in 96 of 100 runs. We tested two memory designs: summaries appended every 24k tokens, and summaries replacing the history at about trim time ([`sumtrimB`](https://www.kaggle.com/code/kazuhiro3381/arc3-sumtrimb)).

| | dp14 | appended | replacing |
|---|---|---|---|
| Demo 25×4 | 50.46 | 43.96 | 44.54 |
| New prefill tokens | 60.6M | **122.1M** | 60.5M |
| Generated tok/s | 639 | 449 | **671** |
| Actions | 21,166 | 14,415 | 14,432 |
| Generated tokens/action | 890 | 918 | **1,365** |

Appended summaries mutate the shared prefix, so prefill (input tokens computed rather than served from cache) doubled and throughput fell 30%. We concluded that compute explained the loss, and predicted that a prefix-stable design would recover the score.

**The prediction failed.** Replacing summaries removed the whole compute cost, yet the score stayed at 44.54, as each action took 53% more tokens.

**Why.** Memory has two prices. Mutating a prefix is paid in prefill, whatever the summary's length. Compressing history is paid in re-checking. In 97 matched runs, tokens per action equal dp14's until the first replacement (1,049 vs 1,018), then diverge (1,469 vs 923). After a replacement, 85% of first responses re-ground (restate the summary, check the frame), against 7% in the appended design. **The price is lost evidence, not the rebuild:** around a context drop, requests advancing no action change by +13.0 points (95% CI 10.4 to 15.7) when ≈8k tokens survive, +1.8 (−1.4 to +5.0) at dp14's ≈56k, and −5.3 (−11.0 to −1.1) at appended's ≈94k.

**Pricing compute.** Actions = time × tok/s ÷ tokens per action, so both designs lost the same 32%, by different routes. Replaying dp14 under a scaled deadline prices that loss (Figure 2): −10% costs 1.7 points, −30% costs 9.2. Real runs cut early agree within 1.1. So compute is not saturated, even though allocation is: +10.5% throughput is worth ≈+2. Zatravkin's 6th-place entry took the same Franzen agent from ≈600 to ≈3,500 tok/s with 110 sessions and lossless changes, moving from 25.72 to 47.19; 1st place (75.56, another harness) reports ≈3,800 tok/s. That is ≈0.05 points per 1% on hidden games, near tantan0327's simulated 0.06, so our public-game price is an upper estimate.

![Figure 2](fig/budget_curve.png)

*Figure 2. Replay under scaled budgets (line), the real run cut early (circles), actual evaluations (±2.2).*

At 70% of the tokens the replay predicts 41.6 and appended scored 43.96 (+2.3, 95% CI −3.7 to +8.1): compute alone explains that loss. Replacing spent 105% of the tokens and scored what dp14 scores with 77%: −7.1 (−14.7 to +0.7) against equal tokens. This is post hoc, every interval includes zero, and dp14 is the best of six.

**Next experiment.** Keep ≈56k verbatim tokens and summarize only at the trim: predicted prefill ≈60M, ≈900 tokens per action, and a clean read of memory's value. Not run.

## 5. A platform failure that scores 0.00

From mid-October, model loading slowed from ≈5 to ≈35 minutes, while the game server ends a run after 15 idle minutes: **four of our ten `dp14` submissions scored exactly 0.00**. Our keep-alive passed a stub test but **was never verified in a scored run**.

## What this means for reaching 85%

- **Compute, not allocation, is the cheap lever.** Our 14 streams were a first step on the path the top entries followed to 110 sessions.
- **Then what the model learns per token**: Provorov finds 46.5% of the attainable score in levels never reached.
- **Evaluation is the bottleneck.** Progress needs more evaluations, held-out game collections, or mechanism-level proxies instead of single scores.

**Universality.** Any LLM agent running long shared contexts under a fixed GPU budget can (1) estimate its detectable effect from per-episode variance, (2) replay its scheduler from request logs and test tunings on held-out episodes, and (3) price memory by the prefill it forces and the extra requests it triggers.

## Code and license

Notebook Apache-2.0; tools CC0; third-party licenses in the long version.

**References.** dfranzen, ARC-AGI-3 Milestone 2 Solution; Tufa Labs, duck-harness; shiiin9, D′ scheduler; Son Pham, discussion/745062; F. Liang, discussion/750885; skarin, 100-Trajectory Audit; A. Provorov, The Book of Ideas; tantan0327, The agent starts like an agent; yy5652, Luck, Budget and Search; G. Zatravkin, 6th-place write-up; Y.-C. Chen, discussion/755795.
