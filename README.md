# EnergAI - A dataset for CNN inference-energy prediction
EnergAI is an extensible dataset of measured energy consumption by convolutional neural networks (CNNs) during image-classification inference. It links measured energy to CNN characteristics, hardware information, inference settings, and measurement details to support the development and evaluation of **AI/ML-based CNN inference energy predictors**. Researchers can reuse these measurements to investigate different prediction approaches, reducing the need to repeat energy experiments when different input features are required.

## Dataset Overview

The first release covers **32 pretrained CNNs** spanning **eight architecture families**, evaluated on the **10,000 CIFAR-10 test images** across **four Intel CPUs** and **four batch sizes** (1, 8, 16, and 32). This gives **512 measurement configurations**, each representing one CNN–CPU–batch-size combination. 

Each configuration was measured **ten times**. The highest and lowest net-energy values were excluded (*symmetric trimming*) to reduce outlier effects, and the remaining eight were averaged. The dataset retains both the individual repetitions and their aggregated energy value.

A separate layer-level description is also provided for each CNN, comprising **5,645 operation records** in total.

We selected CIFAR-10 to keep the classification task and input data consistent. Its small images also make repeated CPU measurements practical. The pretrained CNNs came from:

1. [PyTorch CIFAR-10 Models by Yaofo Chen](https://github.com/chenyaofo/pytorch-cifar-models) — 19 CNNs.
2. [PyTorch CIFAR-10 by Huy Phan](https://github.com/huyvnphan/PyTorch_CIFAR10) — 13 CNNs.

## Repository Structure

```text
EnergAI/
├── energai_v1_main.csv
├── energai_v1_repetitions.csv
├── layers_by_model/
│   └── ...                  # one CSV per CNN
├── Data_Dictionary.md
└── README.md
```

- **`energai_v1_main.csv`** — Contains one record per measurement configuration. Its columns cover identification metadata; CNN, image-dataset, hardware, and inference-configuration characteristics; measurement information; and measured outcomes (energy consumption and classification accuracy).

- **`energai_v1_repetitions.csv`** — Contains ten repetition records per configuration, identifying the CNN and CPU identifiers, batch size, and repetition number, followed by net and raw energy per image, number of evaluated images, measurement duration, and whether the repetition was retained for aggregation. This file is intended to support the assessment of measurement reliability and reproducibility under the experimental conditions of the current version of EnergAI, rather than to provide input features for energy predictors.

- **`layers_by_model/`** — Contains a separate CSV for each CNN, named using its model_id (e.g., cifar10_chen_mobilenetv2_x0_5.csv). Each file describes the CNN's layers, including their types, configurations, tensor dimensions, computational characteristics, and connections. These are structural descriptions, not individual layer-energy measurements.

The combination of `model_id`, `machine_id`, and `batch_size` connects each configuration in the main CSV to its ten repetitions. The `model_id` also identifies the corresponding CSV in layers_by_model/, where `layer_index` identifies each layer type.

See the [**Data Dictionary**](Data_Dictionary.md) for detailed field descriptions and units.
