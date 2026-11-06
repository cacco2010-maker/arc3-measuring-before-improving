"""Figure 1: public-LB scores of identical code (data from README.md submission log)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = [26.10, 28.84, 29.86, 23.98, 25.18, 24.80, 23.47, 21.80, 27.54, 27.26,
        25.96, 26.98, 29.39, 28.44, 23.11]
others = [31.47, 27.89, 27.80, 26.98, 21.01]          # other teams' unchanged copies
dp14 = [33.96, 26.12, 28.69, 28.77, 21.70, 8.53]
dp14_zero = 4                                           # platform-failure runs (0.00)
mu, sd = 25.77, 3.93                                    # public copies (Son Pham)

fig, ax = plt.subplots(figsize=(6.4, 2.9))
rng = np.random.default_rng(0)
rows = [("dfranzen base\n(ours, 15 runs)", base, "#4c72b0"),
        ("dfranzen copies\n(other teams)", others, "#8c8c8c"),
        ("dp14 (ours)", dp14, "#dd8452")]
for i, (lab, xs, c) in enumerate(rows):
    y = i + rng.uniform(-0.12, 0.12, len(xs))
    ax.scatter(xs, y, s=28, color=c, zorder=3)
    ax.plot([np.mean(xs)] * 2, [i - 0.28, i + 0.28], color=c, lw=2)
ax.text(11.0, 2, f"+{dp14_zero}× 0.00\n(platform failure)", fontsize=7, va="center")
xs = np.linspace(6, 38, 300)
ax.fill_between(xs, -0.75, -0.75 + 1.1 * np.exp(-0.5 * ((xs - mu) / sd) ** 2),
                color="#8c8c8c", alpha=0.25, lw=0)
ax.text(36.5, -0.55, f"copies\nN({mu}, {sd}²)", fontsize=7, ha="right")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylim(-0.8, 2.5)
ax.set_xlim(6, 38)
ax.set_xlabel("public LB score (same code, separate runs)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"lb_dist.{ext}", dpi=200)
