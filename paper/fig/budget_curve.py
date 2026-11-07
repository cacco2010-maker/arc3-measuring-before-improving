"""Figure 2: what a compute budget buys (replay of dp14, data from release/analysis/budget_curve.json)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("../../release/analysis/budget_curve.json"))
f = [r["f"] for r in d["curve"]]
s = [r["score"] for r in d["curve"]]
BLUE, ORANGE, GRAY, INK = "#4c72b0", "#dd8452", "#8c8c8c", "#333333"

fig, ax = plt.subplots(figsize=(6.4, 3.1))
ax.plot([x for x in f if x <= 1], s[:len([x for x in f if x <= 1])], color=BLUE, lw=2, zorder=2)
ax.plot([x for x in f if x >= 1], s[-len([x for x in f if x >= 1]):], color=BLUE, lw=2, ls=(0, (4, 2)), zorder=2)
c = d["real_cuts"]
ax.scatter([r["f"] for r in c], [r["score"] for r in c], s=30, facecolor="white", edgecolor=BLUE, lw=1.3, zorder=3)
for x, y, lab, dx, dy, ha in ((1.0, 50.46, "dp14", -0.015, 2.6, "right"),
                              (0.703, 43.96, "appended summaries\n(70% of dp14's tokens)", -0.145, 5.2, "left"),
                              (1.046, 44.54, "replacing summaries\n(105% of dp14's tokens)", 0.02, -3.2, "left")):
    ax.errorbar([x], [y], yerr=2.2, fmt="o", ms=6, color=ORANGE, mec="white", mew=1, elinewidth=1.2, zorder=4)
    ax.text(x + dx, y + dy, lab, fontsize=7.5, color=INK, ha=ha, va="center")
ax.text(1.2, 50.6, "replay of dp14,\nbudget scaled", fontsize=7.5, color=INK, ha="left", va="top")
ax.text(0.62, 32.3, "○ real dp14 run, cut early", fontsize=7.5, color=INK, va="center")
ax.annotate("", xy=(0.79, 44.54), xytext=(1.03, 44.54), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1))
ax.set_xlim(0.53, 1.33); ax.set_ylim(29, 59)
ax.set_xticks([.6, .7, .8, .9, 1, 1.1, 1.2, 1.3]); ax.set_xticklabels(["60%", "70%", "80%", "90%", "100%", "110%", "120%", "130%"])
ax.tick_params(labelsize=8, length=0)
ax.set_xlabel("generated tokens, relative to dp14 (dashed: beyond the run, stitching only)", fontsize=8)
ax.set_ylabel("demo 25×4 score", fontsize=8)
ax.grid(axis="y", alpha=0.3)
for k in ("top", "right"):
    ax.spines[k].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"budget_curve.{ext}", dpi=200)
