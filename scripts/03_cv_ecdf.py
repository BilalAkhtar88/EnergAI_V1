"""Paper Figure 2: empirical CDF of raw and min/max-trimmed energy CV."""
import numpy as np
import matplotlib.pyplot as plt
from _common import KEY, REP_ENERGY, repetitions, trimmed, cv, save_plot


def main():
    raw, kept = [], []
    for _, group in repetitions().groupby(KEY):
        x = group[REP_ENERGY].to_numpy()
        raw.append(cv(x))
        kept.append(cv(trimmed(x)))

    p95_raw = float(np.percentile(raw, 95))
    p95_trimmed = float(np.percentile(kept, 95))
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    # Match the paper's Figure 2: red raw ECDF, blue trimmed ECDF.
    for values, label, color in [
        (raw, "Before trimming (10 runs)", "tab:red"),
        (kept, "After trimming (8 runs)", "tab:blue"),
    ]:
        x = np.sort(values)
        y = 100 * np.arange(1, len(x) + 1) / len(x)
        ax.step(np.r_[0, x], np.r_[0, y], where="post", label=label,
                color=color, linewidth=2.2)

    # 95th-percentile guides, markers, and labels as shown in the paper.
    ax.vlines(p95_trimmed, 0, 95, color="tab:blue", linestyles="--", linewidth=1.6)
    ax.vlines(p95_raw, 0, 95, color="tab:red", linestyles="--", linewidth=1.6)
    ax.scatter([p95_trimmed, p95_raw], [95, 95],
               c=["tab:blue", "tab:red"], s=42, zorder=5)
    ax.text(0.36, 96.2, f"Trimmed P95 = {p95_trimmed:.2f}%",
            color="tab:blue", fontsize=11)
    ax.text(1.76, 88.5, f"Raw P95 = {p95_raw:.2f}%",
            color="tab:red", fontsize=11)
    ax.set_xlim(0, 3)  # Display zoom only; all 512 configurations are included.
    ax.set_ylim(0, 101.5)
    ax.set_xticks(np.arange(0, 3.01, 0.5))
    ax.set_yticks([0, 25, 50, 75, 95, 100])
    ax.set_xlabel("Coefficient of variation, CV (%)")
    ax.set_ylabel("Configurations with CV ≤ x (%)")
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", frameon=False)
    save_plot("fig2_cv_ecdf.png", vector=True)

    print("Figure 2 saved: results/fig2_cv_ecdf.png and .pdf")
    print(f"95th percentile CV: {p95_raw:.2f}% → {p95_trimmed:.2f}%")
    print(f"Maximum CV: {max(raw):.2f}% → {max(kept):.2f}%")


if __name__ == "__main__":
    main()
