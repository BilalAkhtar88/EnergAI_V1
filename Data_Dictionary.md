# EnergAI Data Dictionary

The fields appear in the same order as the CSV columns. The README explains the dataset files and their relationships.

**Value conventions**

Flags generally use `1` for yes and `0` for no. Accuracy and probability values are fractions. `None` means that no physical unit applies.

## 1. `energai_v1_main.csv`

### Identification metadata

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `model_id` | Text | None | Unique identifier of the specific dataset, pretrianed-weight source and CNN implementation. Use for links to repetitions and layer descriptions. |
| `model_name` | Text | None | Name of the CNN model or variant. |
| `model_source` | Text | None | Repository that supplied the CNN implementation and weights. |
| `pretrained` | Flag | None | Indicates whether pretrained weights were used. `1` means pretrained weights were used; `0` means no pretrained weights. All current records are `1`. |
| `machine_id` | Text | None | Identifier for the CPU machine. Use to link repetition records. |
| `dataset_id` | Text | None | Identifier for the image dataset (`cifar10` in current version). |

### CNN characteristics

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `architecture_family` | Text | None | CNN architecture family such as ResNet, DenseNet, etc. |
| `total_parameter_count` | Integer | Parameters | Total number of CNN parameters. |
| `model_state_size_bytes` | Integer | Bytes | Size of the PyTorch model state, including parameters and buffers, obtained from its `state_dict`. |
| `conv_layer_count` | Integer | Modules | Number of convolution modules in the CNN. |
| `linear_layer_count` | Integer | Modules | Number of linear modules in the CNN. |
| `conv_and_linear_layer_count` | Integer | Modules | Combined number of convolution and linear modules in the CNN. |
| `mac_count` | Integer | MACs | Estimated multiply and accumulate operations for one CNN forward pass at the recorded input size. |
| `flop_count` | Integer | FLOPs | Estimated floating point operations for one CNN forward pass at the recorded input size. |
| `input_channels` | Integer | Channels | Number of input image channels in the CNN input tensor (3 for CIFAR-10 RGB images). |
| `input_height` | Integer | Pixels | Expected input image height (32 in current version). |
| `input_width` | Integer | Pixels | Expected input image width (32 in current version). |

### Image dataset characteristics

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `task` | Text | None | Task performed by the CNN (`image_classification` in current version). |
| `modality` | Text | None | Type of input data (`image` in current version). |
| `num_classes` | Integer | Classes | Number of classification categories in the evaluated dataset (`10` in current version). |

### Hardware characteristics

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `hw_cpu_model_name` | Text | None | CPU model name. |
| `hw_cpu_architecture` | Text | None | CPU instruction set architecture. |
| `hw_cpu_physical_cores` | Integer | Cores | Number of physical CPU cores. |
| `hw_cpu_logical_cores` | Integer | Threads | Number of logical CPU processors. |
| `hw_cpu_performance_core_count` | Integer | Cores | Number of performance cores. |
| `hw_cpu_efficiency_core_count` | Integer | Cores | Number of efficiency cores. |
| `hw_cpu_low_power_efficiency_core_count` | Integer | Cores | Number of low power efficiency cores. |
| `hw_cpu_max_freq_mhz` | Integer | MHz | Recorded maximum CPU frequency specification. Not an observed inference frequency. |
| `hw_cpu_min_freq_mhz` | Integer | MHz | Recorded minimum CPU frequency specification. Not an observed inference frequency. |
| `hw_l1d_cache_kb` | Integer | KB | Total level 1 data cache capacity. |
| `hw_l1i_cache_kb` | Integer | KB | Total level 1 instruction cache capacity. |
| `hw_l2_cache_kb` | Integer | KB | Total level 2 cache capacity. |
| `hw_l3_cache_kb` | Integer | KB | Total level 3 cache capacity. |
| `hw_cpu_cache_line_size_bytes` | Integer | Bytes | Size of one CPU cache line as reported for the system. |
| `hw_ram_total_gb` | Decimal | GB | Recorded machine RAM capacity. Not CNN memory use. |
| `hw_memory_type` | Text | None | RAM technology such as DDR4 or LPDDR5X. |
| `hw_memory_speed_mts` | Integer | MT/s | Rated memory transfers per second, in millions. |
| `hw_cpu_lithography` | Text | None | Reported CPU manufacturing process. |
| `hw_cpu_max_power_w` | Decimal | W | Recorded CPU power specification. Not measured inference power. |

### Inference configuration

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `batch_size` | Integer | Images | Number of images in one inference batch. |
| `num_threads` | Integer | Threads | Number of PyTorch CPU inference threads. |
| `precision` | Text | None | Numerical precision used for inference. |
| `inference_framework` | Text | None | Software framework used for inference. |
| `cpu_frequency_mode` | Text | None | CPU frequency control mode. |
| `cpu_governor` | Text | None | Linux CPU frequency governor. |
| `cpu_turbo_enabled` | Flag | None | Indicates whether CPU turbo boosting was enabled. |

### Measurement information

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `num_eval_samples` | Integer | Images | Number of images processed in each repetition. |
| `num_repetitions` | Integer | Repetitions | Number of times complete measurements for this CNN–CPU–batch-size configuration was repeated (`10` in current version ). |
| `warmup_iterations` | Integer | Iterations | Number of warmup iterations before measurement. |
| `robust_trim_count` | Integer | Repetitions | Number of excluded repetitions in total (`2` in current version). |
| `robust_trim_rule` | Text | None | Aggregation rule. `classic_sorted_min_max` used in current version that excludes `robust_trim_count / 2` lowest and `robust_trim_count / 2` highest net energy reading. |
| `idle_subtraction_applied` | Flag | None | `1` if the estimated idle energy was subtracted. `0` otherwise. |
| `cpu_energy_measurement_method` | Text | None | Method used to obtain CPU energy (`rapl`,Intel Running Average Power Limit, accessed via Linux powercap in current version). |

### Measured outcomes

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `energy_j_per_sample` | Decimal | J/image | Mean net CPU package energy per image after trimming. |
| `top1_accuracy` | Decimal | Fraction | Share of images classified correctly using the top prediction; e.g., `0.9406` means 94.06%, not 0.9406%. |

## 2. `energai_v1_repetitions.csv`

### Repetition fields

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `machine_id` | Text | None | Identifier for the CPU machine. |
| `model_id` | Text | None | Identifier for the exact CNN implementation. |
| `batch_size` | Integer | Images | Number of images in one inference batch. |
| `repetition_id` | Integer | None | Number assigned to this repetition, from 1 to `num_repetitions`. |
| `net_package_energy_j_per_sample` | Decimal | J/image | CPU package energy per image after subtracting estimated idle energy. |
| `raw_package_energy_j_per_sample` | Decimal | J/image | CPU package energy per image before subtracting estimated idle energy. |
| `measured_samples` | Integer | Images | Number of images processed during this repetition. |
| `duration_s` | Decimal | Seconds | Elapsed time of the measured inference interval for this repetition. |
| `included_in_canonical_trim` | Flag | None | `1` if the repetition was retained when calculating the main CSV energy label. `0` for the lowest or highest net-energy repetition excluded by `robust_trim_rule`. |
| `timestamp_utc` | Text | UTC date and time | Recorded date and time in UTC. |

## 3. `layers_by_model/` CSV files

### Operation identification

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `model_id` | Text | None | Identifier for the CNN described by this record. |
| `layer_index` | Integer | None | Position of the operation in the traced CNN sequence. Starts at 0. |
| `layer_type` | Text | None | Specific recorded operation name. |
| `layer_type_id` | Integer | Category | Broad operation category. See the [Layer type codes](#layer-type-codes) table. |

### Operation configuration

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `in_channels` | Integer | Channels | Recorded count of input channels or features associated with the operation. |
| `out_channels` | integer | channels/features | Recorded count of output channels or features associated with the operation. |
| `kernel_h` | integer | elements | Kernel height for applicable operations. `0` is used when a kernel attribute is not applicable. |
| `kernel_w` | integer | elements | Kernel width for applicable operations; `0` when not applicable. |
| `stride_h` | integer | elements | Vertical stride recorded for an applicable operation; `0` when not applicable. |
| `stride_w` | integer | elements | Horizontal stride recorded for an applicable operation; `0` when not applicable. |
| `padding_h` | integer | elements | Vertical padding amount recorded for an applicable operation; `0` can mean either no padding or not applicable. |
| `padding_w` | integer | elements | Horizontal padding amount; `0` can mean either no padding or not applicable. |
| `dilation_h` | integer | elements | Vertical dilation factor for applicable convolution operations; `0` when not applicable. |
| `dilation_w` | integer | elements | Horizontal dilation factor for applicable convolution operations; `0` when not applicable. |
| `groups` | integer | groups | Number of convolution groups for applicable convolution operations; `0` when not applicable. |
| `is_depthwise` | boolean (0/1) | — | `1` if the recorded convolution is depthwise, otherwise `0`. For non-convolution operations this is `0`. |

### Input and output dimensions

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `input_h` | Integer | Positions | Height of the input feature map when applicable. |
| `input_w` | Integer | Positions | Width of the input feature map when applicable. |
| `output_h` | Integer | Positions | Height of the output feature map when applicable. |
| `output_w` | Integer | Positions | Width of the output feature map when applicable. |

### Parameters and computation

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `parameter_count` | Integer | Parameters | Number of parameters attributed to the operation. |
| `mac_count` | Integer | MACs | Estimated multiply and accumulate operations attributed to the operation. |
| `has_bias` | Flag | None | Indicates whether a bias parameter is present. |

### Connections and merges

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `incoming_skip_count` | integer | connections | Number of incoming connections other than the main input connection represented by the extracted operation graph. |
| `outgoing_skip_count` | integer | connections | Number of additional outgoing connections from this operation that are used as non-main inputs by later operations in the extracted graph. |
| `minimum_skip_distance` | integer | operation-index steps | Smallest difference in `layer_index` between this operation and a source of an additional incoming connection. `0` if none. |
| `maximum_skip_distance` | integer | operation-index steps | Largest difference in `layer_index` between this operation and a source of an additional incoming connection. `0` if none. |
| `merge_type_id` | integer | category code | Type of merge recorded in the traced graph: `0` = not a merge, `1` = addition, `2` = concatenation, and `3` = multiplication. |
| `merge_input_count` | integer | inputs | Number of merge inputs recorded by the graph extraction process; `0` when no merge is recorded. |

### Inference properties

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `active_in_inference` | Flag | None | Indicates whether the operation is active during inference. |
| `dropout_probability` | Decimal | Fraction | Dropout probability. Zero where dropout does not apply. |

### Layer type codes

These IDs indicate broad categories, not unique operations. Use `layer_type` to distinguish specific operations.

| Code | Category | Recorded operations |
|---|---|---|
| 1 | Convolution | `Conv2d` |
| 2 | Normalization | `BatchNorm2d` |
| 3 | Activation | `ReLU`, `ReLU6` |
| 4 | Pooling | `AdaptiveAvgPool2d`, `adaptive_avg_pool2d`, `AvgPool2d`, `avg_pool2d`, `MaxPool2d`, `max_pool2d`, `GlobalMeanPool` |
| 5 | Linear | `Linear` |
| 6 | Dropout | `Dropout` |
| 7 | Addition | `Add` |
| 8 | Concatenation | `Concatenate` |
| 9 | Tensor shape and layout | `flatten`, `reshape`, `contiguous`, `transpose`, `view` |
| 10 | Splitting | `chunk` |
