"""Paper Table II: additive and bilinear ridge regressors on held-out CNN families."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from _common import KEY, ENERGY, RESULTS, main_data

# Edit these lists to select numerical columns from energai_v1_main.csv.
MODEL_FEATURES = ["mac_count", "total_parameter_count"]
HARDWARE_FEATURES = ["hw_cpu_physical_cores", "hw_cpu_max_freq_mhz", "hw_l3_cache_kb"]
INFERENCE_FEATURES = ["batch_size"]

# Paper's held-out-family experiment. These may also be changed.
HELD_OUT_FAMILIES = ["RepVGG", "InceptionV3", "GoogLeNet"]
EXCLUDED_MODELS = ["cifar10_huyv_mobilenet_v2"]
RIDGE_ALPHA = 1.0  # Paper setting (scikit-learn default value is 1.0)


def score(actual, predicted):
    error = predicted - actual
    return {
        "MAPE_percent": 100 * np.mean(np.abs(error) / actual),
        "MAE_mJ": 1000 * np.mean(np.abs(error)),
        "RMSE_mJ": 1000 * np.sqrt(np.mean(error ** 2)),
    }


def main():
    data = main_data()
    columns = MODEL_FEATURES + HARDWARE_FEATURES + INFERENCE_FEATURES
    assert MODEL_FEATURES and HARDWARE_FEATURES, "Select CNN and hardware features"
    assert len(columns) == len(set(columns)), "Duplicate feature column"
    for col in columns:
        if col not in data:
            raise ValueError(f"Unknown column {col!r}. Check energai_v1_main.csv")
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"{col!r} is not numeric; select numeric feature columns")
        if col in [ENERGY, "top1_accuracy"]:
            raise ValueError(f"{col!r} is an outcome, not a design-time feature")

    available_families = set(data.architecture_family)
    if not set(HELD_OUT_FAMILIES).issubset(available_families):
        raise ValueError("Unknown held-out CNN family")
    if not set(EXCLUDED_MODELS).issubset(set(data.model_id)):
        raise ValueError("Unknown excluded CNN")
    data = data[~data.model_id.isin(EXCLUDED_MODELS)]
    train = data[~data.architecture_family.isin(HELD_OUT_FAMILIES)]
    test = data[data.architecture_family.isin(HELD_OUT_FAMILIES)]
    assert len(train) and len(test) and not (set(train.model_id) & set(test.model_id))
    if data[columns + [ENERGY]].isna().any().any():
        raise ValueError("Missing values in selected features or target")

    # Standardize from the training data only. Keep the target in joules.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[columns])
    x_test = scaler.transform(test[columns])
    y_train, y_test = train[ENERGY].to_numpy(), test[ENERGY].to_numpy()
    m, h = len(MODEL_FEATURES), len(HARDWARE_FEATURES)

    def bilinear(x):
        interactions = (x[:, :m, None] * x[:, None, m:m + h]).reshape(len(x), -1)
        return np.column_stack([x, interactions])

    rows = []
    for name, inputs, held_out in [
        ("Additive linear", x_train, x_test),
        ("Bilinear", bilinear(x_train), bilinear(x_test)),
    ]:
        predicted = Ridge(fit_intercept=True).fit(inputs, y_train).predict(held_out)
#        predicted = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(inputs, y_train).predict(held_out)
        rows.append({"predictor": name, **score(y_test, predicted)})

    results = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "table2_demo_predictors.csv", index=False)
    print(f"Train: {train.model_id.nunique()} CNNs, {len(train)} configurations")
    print(f"Test: {test.model_id.nunique()} CNNs, {len(test)} configurations")
    print("Ridge alpha:", RIDGE_ALPHA)
    print("Model features:", MODEL_FEATURES)
    print("Hardware features:", HARDWARE_FEATURES)
    print("Inference features:", INFERENCE_FEATURES)
    print(results.round(2).to_string(index=False))
    print("Table II saved: results/table2_demo_predictors.csv")


if __name__ == "__main__":
    main()
