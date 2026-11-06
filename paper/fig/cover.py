"""Cover image for the Kaggle Writeup (2:1, 1600x800).

Message: identical code spreads over ~8 points, most published tweaks are worth 1-2, so measure first.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = [26.10, 28.84, 29.86, 23.98, 25.18, 24.80, 23.47, 21.80, 27.54, 27.26,
        25.96, 26.98, 29.39, 28.44, 23.11]
mu, sd = 25.77, 3.93                                    # public copies (Son Pham)

BG, FG, MUTED, ACC, BLUE = "#101418", "#f2f2f2", "#8a939c", "#f0a04b", "#5b8fd6"
fig = plt.figure(figsize=(8, 4), dpi=200, facecolor=BG)

fig.text(0.05, 0.86, "Measuring Before Improving", color=FG, fontsize=25, weight="bold")
fig.text(0.05, 0.775, "Variance, compute allocation and cache costs in an LLM harness for ARC-AGI-3",
         color=MUTED, fontsize=10.5)

ax = fig.add_axes([0.05, 0.17, 0.9, 0.5], facecolor=BG)
xs = np.linspace(12, 40, 400)
ax.fill_between(xs, 0, np.exp(-0.5 * ((xs - mu) / sd) ** 2), color=BLUE, alpha=0.28, lw=0)
ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2), color=BLUE, lw=1.2)
rng = np.random.default_rng(1)
ax.scatter(base, 0.06 + rng.uniform(-0.03, 0.03, len(base)), s=22, color=FG, zorder=3)
ax.text(mu, 1.06, "same code, 15 runs: 21.8 – 29.9", color=FG, fontsize=9.5, ha="center",
        bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))

# a typical published tweak vs the smallest effect one run can see
y1, y2 = 0.62, 0.40
ax.annotate("", xy=(mu + 1.5, y1), xytext=(mu, y1),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
ax.text(32.4, y1, "typical tweak  +1 to +2", color=MUTED, fontsize=9, va="center")
ax.plot([mu + 1.7, 32.1], [y1, y1], color=MUTED, lw=0.6, ls=":")
ax.annotate("", xy=(mu + 5, y2), xytext=(mu, y2),
            arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2.2))
ax.text(32.4, y2, "detectable in one run  ≈ +5", color=ACC, fontsize=9.5, va="center", weight="bold")
ax.axvline(mu, ymin=0, ymax=0.84, color=MUTED, lw=0.8, ls=":")

ax.set_xlim(12, 40)
ax.set_ylim(0, 1.18)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=8)
ax.set_xlabel("public leaderboard score", color=MUTED, fontsize=8.5)

fig.savefig("cover.png", facecolor=BG)
