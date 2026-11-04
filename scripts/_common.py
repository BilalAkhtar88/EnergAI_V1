"""Shared paths and calculations for the EnergAI paper scripts."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
KEY = ["machine_id", "model_id", "batch_size"]
ENERGY = "energy_j_per_sample"
REP_ENERGY = "net_package_energy_j_per_sample"


def main_data():
    return pd.read_csv(ROOT / "energai_v1_main.csv")


def repetitions():
    return pd.read_csv(ROOT / "energai_v1_repetitions.csv")


def trimmed(values):
    """Drop one minimum and one maximum net-energy observation."""
    return np.sort(np.asarray(values, dtype=float))[1:-1]


def cv(values):
    """Sample coefficient of variation, expressed as a percentage."""
    x = np.asarray(values, dtype=float)
    return 100 * x.std(ddof=1) / x.mean()


def save_plot(filename, vector=False):
    from matplotlib import pyplot as plt
    RESULTS.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=300)
    if vector:
        plt.savefig((RESULTS / filename).with_suffix(".pdf"))
    plt.close()
