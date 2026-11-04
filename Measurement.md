# Energy Measurement Protocol
This document describes the energy measurement protocol used for the current version of EnergAI. The protocol may be adapted for different hardware or measurement instruments. The measurement conditions and procedures should be documented to support the interpretation and reproducibility of the energy values.

## 1. Controlled environment

The following settings were used for EnergAI v1.

| Setting | Value or control |
|---|---|
| Execution Device | CPU only |
| Inference Framework | PyTorch |
| Numerical Precision | FP32 |
| CPU threads | 1 intra-operation thread and 1 inter-operation thread |
| CPU frequency | Dynamic frequency with the `performance` governor |
| Turbo boosting | Disabled |
| Power supply | AC power connected |
| Wi-Fi and Bluetooth | Disabled |
| Background activity | Other applications closed, and background activity minimized |

## 2. Measurement procedure

1. **Prepare the inputs.** 
For each CNN, CPU, and batch size configuration, the pretrained CNN was loaded in evaluation mode. The complete CIFAR-10 test set of 10,000 images was normalized using the CNN's preprocessing settings, divided into batches, and stored in CPU memory before measurement. The same prepared batches were reused for every repetition. This excluded image loading, decoding, normalization, and batch preparation from the measured interval. It also reduced variability from repeated file access and data loading.

2. **Warm up the CNN.** 
Each CNN performed 50 forward passes before measurement. This initialized the execution path and reduced the influence of initial execution overhead on the energy readings. Warmup was excluded from the measured interval.

3. **Measure idle power.** CPU package energy was measured over a 120-second idle interval after warmup. Average idle power was calculated by dividing the measured energy by the elapsed time. One idle power value was obtained for each configuration and used for its ten repetitions.

4. **Measure inference.** Intel RAPL energy was read through the Linux powercap interface using the `package-0` domain. Each repetition processed all 10,000 test images once. Energy and elapsed time were measured around the inference loop. Ten repetitions were performed with 120 seconds of rest between consecutive repetitions.

5. **Calculate and aggregate energy.** Estimated idle energy was subtracted from each raw CPU package energy reading. The resulting net energy was divided by 10,000 to obtain the value in joules per image. The ten values were sorted, the lowest and highest were excluded, and the remaining eight were averaged. This value was recorded as `energy_j_per_sample`.

6. **Evaluate accuracy.** Top-1 accuracy was calculated separately and recorded as `top1_accuracy`.

Model loading, image decoding, normalization, batch preparation, warmup, and accuracy evaluation were outside the energy measurement interval. The measured interval included model forward passes and the CPU tensor operations within the inference loop.

## 3. Measurement considerations

For each configuration, mean idle power was calculated as

$$
P_{\mathrm{idle}} = \frac{E_{\mathrm{idle}}}{T_{\mathrm{idle}}}.
$$

For each repetition, net energy per image was calculated as

$$
E_{\mathrm{image}} = \frac{E_{\mathrm{raw}} - P_{\mathrm{idle}} T_{\mathrm{inference}}}{10{,}000}.
$$

The final energy label was the mean of the eight retained values of \(E_{\mathrm{image}}\).

**Energy scope.** RAPL measurements represented CPU package energy, not total system energy.

**Counter wraparound.** RAPL counters reset to zero when their maximum range is reached. The measurement code handled a single RAPL counter wraparound by using Linux's `max_energy_range_uj` when the ending counter value was lower than the starting counter value. One wraparound was confirmed in the recorded measurements and corrected.

**Frequency limitation.** The governor and turbo settings were controlled, but the actual CPU frequency was neither fixed nor continuously recorded. Frequency drift cannot be excluded.

## 4. Alternative measurement methods

Future EnergAI measurements may use different instruments or experimental settings. The measurement metadata should describe the procedure actually used. For example, the following table illustrates a hardware power meter with three repetitions. All three energy readings are retained and averaged without trimming.

| Main CSV column | EnergAI v1 | Illustrative external CPU package meter |
|---|---|---|
| `num_eval_samples` | `10000` | `10000` |
| `num_repetitions` | `10` | `3` |
| `warmup_iterations` | `50` | `50` |
| `robust_trim_count` | `2` | `0` |
| `robust_trim_rule` | `classic_sorted_min_max` | `none` |
| `idle_subtraction_applied` | `1` | `1` |
| `cpu_energy_measurement_method` | `rapl` | `external_power_meter` |

The alternative is illustrative. The actual instrument, its placement, and the measured energy scope should be documented, along with any changes to environment controls, idle measurement duration, or rest intervals.
