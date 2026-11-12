"""Paper Figures 4–5: MAC/parameter relationships with energy at batch size 1."""
import matplotlib.pyplot as plt
from _common import ENERGY, main_data, save_plot


def main():
    data = main_data()
    for feature, name, output in [
        ("mac_count", "MAC count", "fig4_macs_energy.png"),
        ("total_parameter_count", "Parameter count", "fig5_params_energy.png"),
    ]:
        for machine, group in data[data.batch_size == 1].groupby("machine_id"):
            plt.scatter(group[feature] / 1e6, group[ENERGY] * 1000,
                        label=machine.replace("machine_", "M"), s=20, alpha=0.8)
        plt.xscale("log")  # Plot only; Pearson correlations use untransformed values.
        plt.xlabel(f"{name} (millions, log scale)")
        plt.ylabel("Net CPU-package energy (mJ/image)")
        plt.grid(alpha=0.2)
        plt.legend()
        save_plot(output)

        correlations = [g[feature].corr(g[ENERGY])
                        for _, g in data.groupby(["machine_id", "batch_size"])]
        print(f"{name} Pearson r across CPU/batch groups: "
              f"{min(correlations):.3f} to {max(correlations):.3f}")

    print(f"Net energy range: {data[ENERGY].min() * 1000:.2f} to "
          f"{data[ENERGY].max() * 1000:.2f} mJ/image")
    print("Figures 4–5 saved in results/")


if __name__ == "__main__":
    main()
