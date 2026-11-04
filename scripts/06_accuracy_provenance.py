"""Paper examples: accuracy–energy contrast and MobileNetV2 provenance."""
import numpy as np
from _common import ENERGY, main_data


def main():
    data = main_data()
    print(f"Top-1 accuracy range: {data.top1_accuracy.min() * 100:.2f}% to "
          f"{data.top1_accuracy.max() * 100:.2f}%")

    for model in ["cifar10_chen_mobilenetv2_x0_75", "cifar10_huyv_densenet161"]:
        row = data[(data.model_id == model) & (data.machine_id == "machine_001") &
                   (data.batch_size == 1)].iloc[0]
        print(f"{model}: accuracy {row.top1_accuracy * 100:.2f}%, "
              f"energy {row[ENERGY] * 1000:.2f} mJ/image (M001, batch 1)")

    chen = data[data.model_id == "cifar10_chen_mobilenetv2_x1_0"]
    huyv = data[data.model_id == "cifar10_huyv_mobilenet_v2"]
    matched = chen.merge(huyv, on=["machine_id", "batch_size"],
                         suffixes=("_chen", "_huyv"), validate="one_to_one")
    assert len(matched) == 16
    difference = 100 * np.abs(matched[ENERGY + "_huyv"] - matched[ENERGY + "_chen"]) / matched[ENERGY + "_chen"]
    print("Chen/Huy MobileNetV2 matching parameter counts:",
          bool((matched.total_parameter_count_chen == matched.total_parameter_count_huyv).all()))
    print("Chen/Huy MobileNetV2 matching MAC counts:",
          bool((matched.mac_count_chen == matched.mac_count_huyv).all()))
    print(f"Median absolute relative energy difference: {difference.median():.2f}%")


if __name__ == "__main__":
    main()
