"""Paper Figure 2: maximum trimmed CV across the 24 pilot configurations."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import KEY, REP_ENERGY, repetitions, trimmed, cv, save_plot

# Three CNNs and two CPUs used in the paper's repetition-count pilot.
PILOT_MODELS = [
    "cifar10_huyv_mobilenet_v2",
    "cifar10_huyv_resnet50",
    "cifar10_huyv_vgg13_bn",
]
PILOT_MACHINES = ["machine_001", "machine_004"]


def main():
    reps = repetitions()
    pilot = reps[reps.model_id.isin(PILOT_MODELS) & reps.machine_id.isin(PILOT_MACHINES)]
    groups = list(pilot.groupby(KEY))
    assert len(groups) == 24, f"Expected 24 pilot configurations, found {len(groups)}"

    rows = []
    for (machine, model, batch), group in groups:
        x = group.sort_values("repetition_id")[REP_ENERGY].to_numpy()
        for n in range(4, 11):
            kept = trimmed(x[:n])
            rows.append([machine, model, batch, n, cv(kept), kept.mean()])
    pilot_cv = pd.DataFrame(rows, columns=KEY + ["n", "cv_percent", "mean_j"])
    maxima = pilot_cv.groupby(["batch_size", "n"])["cv_percent"].max().unstack("batch_size")

    # Match the paper's Figure 2: four colored curves and legend inside bottom-left.
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {1: "tab:blue", 8: "tab:orange", 16: "tab:green", 32: "tab:red"}
    for batch in (1, 8, 16, 32):
        ax.plot(maxima.index, maxima[batch], "o-", color=colors[batch],
                linewidth=1.8, markersize=6, label=f"Batch {batch}")
    ax.set_xlabel("Number of repetitions, n")
    ax.set_ylabel("Maximum trimmed CV (%)")
    ax.set_xticks(range(4, 11))
    ax.set_ylim(0, 1.55)
    ax.set_yticks(np.arange(0, 1.41, 0.2))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", ncol=4, frameon=False)
    save_plot("fig2_pilot_cv.png", vector=True)

    means = pilot_cv[pilot_cv.n.isin([9, 10])].pivot(index=KEY, columns="n", values="mean_j")
    change = 100 * ((means[10] - means[9]).abs() / means[10]).max()
    print("Figure 2 saved: results/fig2_pilot_cv.png and .pdf")
    print(f"Maximum change in trimmed mean (9 to 10 repetitions): {change:.3f}%")
    print("Batch 16 max CV at n=8, 9, 10:",
          ", ".join(f"{maxima.loc[n, 16]:.2f}%" for n in [8, 9, 10]))


if __name__ == "__main__":
    main()
