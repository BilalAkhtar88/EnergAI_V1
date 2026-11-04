"""Check dataset coverage and independently reconstruct every energy label."""
import numpy as np
from _common import ROOT, KEY, ENERGY, REP_ENERGY, main_data, repetitions, trimmed


def main():
    data, reps = main_data(), repetitions()
    assert not data.duplicated(KEY).any(), "Duplicate main configurations"
    assert not reps.duplicated(KEY + ["repetition_id"]).any(), "Duplicate repetition IDs"

    main_keys = set(map(tuple, data[KEY].to_numpy()))
    rep_keys = set(map(tuple, reps[KEY].drop_duplicates().to_numpy()))
    assert main_keys == rep_keys, "Main and repetition configuration keys differ"

    reference = data.set_index(KEY)[ENERGY]
    for key, group in reps.groupby(KEY):
        x = group[REP_ENERGY].to_numpy()
        flags = group["included_in_canonical_trim"].to_numpy().astype(bool)
        assert len(x) == 10 and flags.sum() == 8, f"Invalid repetitions: {key}"
        assert np.isfinite(x).all() and (x > 0).all(), f"Invalid energy: {key}"
        expected = trimmed(x)
        assert np.allclose(np.sort(x[flags]), expected, rtol=0, atol=1e-12), f"Trim flags: {key}"
        assert np.isclose(reference.loc[key], expected.mean(), rtol=0, atol=1e-10), f"Energy label: {key}"

    assert data[ENERGY].notna().all() and (data[ENERGY] > 0).all()
    print("Validation passed")
    print(f"  Configurations: {len(data)}")
    print(f"  Repetitions: {len(reps)}")
    print(f"  CNNs: {data.model_id.nunique()}")
    print(f"  CPU systems: {data.machine_id.nunique()}")
    print(f"  Batch sizes: {sorted(map(int, data.batch_size.unique()))}")
    print("  All energy labels match min/max trimming of ten readings")

    # Verify the per-CNN structure files when they are included in the repository.
    layer_dir = ROOT / "layers_by_model"
    if layer_dir.is_dir():
        files = sorted(layer_dir.glob("*.csv"))
        assert {p.stem for p in files} == set(data.model_id), "CNN layer files do not match main CSV"
        import pandas as pd
        n_layers = sum(len(pd.read_csv(p)) for p in files)
        print(f"  Per-CNN structure files: {len(files)}; operation records: {n_layers}")


if __name__ == "__main__":
    main()
