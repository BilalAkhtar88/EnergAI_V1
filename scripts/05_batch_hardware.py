"""Paper Figure 6: batch-size and hardware effects on energy per image."""
import numpy as np
import matplotlib.pyplot as plt
from _common import ENERGY, main_data, save_plot


def main():
    data = main_data()
    medians = data.groupby(["machine_id", "batch_size"])[ENERGY].median()
    for machine, values in medians.groupby(level=0):
        series = values.droplevel("machine_id").sort_index()
        plt.plot(series.index, series.to_numpy() * 1000, marker="o",
                 label=machine.replace("machine_", "M"))
    plt.xticks([1, 8, 16, 32])
    plt.xlabel("Batch size")
    plt.ylabel("Median energy (mJ/image, across CNNs)")
    plt.legend()
    plt.grid(alpha=0.2)
    save_plot("fig6_batch_hardware.png")

    pairs = data.pivot(index=["machine_id", "model_id"], columns="batch_size", values=ENERGY)
    ratios = data.groupby(["model_id", "batch_size"])[ENERGY].agg(["min", "max"])
    by_machine = data.groupby(["model_id", "machine_id"])[ENERGY].mean().unstack()
    rank = by_machine.corr(method="spearman").to_numpy()
    rank = rank[np.triu_indices_from(rank, k=1)]

    print("Figure 6 saved: results/fig6_batch_hardware.png")
    print(f"Batch 1→8 energy reduction: {(pairs[8] < pairs[1]).sum()}/{len(pairs)} pairs")
    print(f"Median reduction: {100 * np.median(1 - pairs[8] / pairs[1]):.1f}%")
    print(f"Batch 32 > batch 8: {(pairs[32] > pairs[8]).sum()}/{len(pairs)} pairs")
    print(f"Median across-CPU maximum/minimum energy ratio: "
          f"{(ratios['max'] / ratios['min']).median():.2f}")
    print(f"Across-CPU Spearman correlations: {rank.min():.3f} to {rank.max():.3f}")


if __name__ == "__main__":
    main()
