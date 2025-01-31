# Docling-eval


[![arXiv](https://img.shields.io/badge/arXiv-2408.09869-b31b1b.svg)](https://arxiv.org/abs/2408.09869)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ds4sd.github.io/docling/)
[![PyPI version](https://img.shields.io/pypi/v/docling)](https://pypi.org/project/docling/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling)](https://pypi.org/project/docling/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/DS4SD/docling)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling/month)](https://pepy.tech/projects/docling)

Evaluate [Docling](https://github.com/DS4SD/docling) on various datasets.

## Features

Evaluate docling on various datasets. You can use the cli

```sh
docling-eval % poetry run evaluate --help
2024-12-20 10:51:57,593 - INFO - PyTorch version 2.5.1 available.

 Usage: evaluate [OPTIONS]

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --task        -t      [create|evaluate|visualize]                                                                Evaluation task [default: None] [required]                                                                              │
│ *  --modality    -m      [end-to-end|layout|tableformer|codeformer]                                                 Evaluation modality [default: None] [required]                                                                          │
│ *  --benchmark   -b      [DPBench|OmniDcoBench|WordScape|PubLayNet|DocLayNet|Pub1M|PubTabNet|FinTabNet|WikiTabNet]  Benchmark name [default: None] [required]                                                                               │
│ *  --input-dir   -i      PATH                                                                                       Input directory [default: None] [required]                                                                              │
│ *  --output-dir  -o      PATH                                                                                       Output directory [default: None] [required]                                                                             │
│    --help                                                                                                           Show this message and exit.                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## End to End examples

### FinTabNet

Using a single command (loading the dataset from Huggingface: [FinTabNet_OTSL](https://huggingface.co/datasets/ds4sd/FinTabNet_OTSL)),

```sh
poetry run python ./docs/examples/benchmark_fintabnet.py
```

<details>
<summary><b>Table evaluations for FinTabNet</b></summary>
<br>

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b FinTabNet -i ./benchmarks/fintabnet-dataset/tableformer -o ./benchmarks/fintabnet-dataset/tableformer
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b FinTabNet -i ./benchmarks/fintabnet-dataset/tableformer -o ./benchmarks/fintabnet-dataset/tableformer
```

The final result (struct only here) can be visualised as,

|   x0<=TEDS |   TEDS<=x1 |   prob [%] |   acc [%] |   1-acc [%] |   total |
|------------|------------|------------|-----------|-------------|---------|
|       0    |       0.05 |        0   |       0   |       100   |       0 |
|       0.05 |       0.1  |        0   |       0   |       100   |       0 |
|       0.1  |       0.15 |        0   |       0   |       100   |       0 |
|       0.15 |       0.2  |        0.2 |       0   |       100   |       2 |
|       0.2  |       0.25 |        0   |       0.2 |        99.8 |       0 |
|       0.25 |       0.3  |        0   |       0.2 |        99.8 |       0 |
|       0.3  |       0.35 |        0   |       0.2 |        99.8 |       0 |
|       0.35 |       0.4  |        0   |       0.2 |        99.8 |       0 |
|       0.4  |       0.45 |        0   |       0.2 |        99.8 |       0 |
|       0.45 |       0.5  |        0   |       0.2 |        99.8 |       0 |
|       0.5  |       0.55 |        0.3 |       0.2 |        99.8 |       3 |
|       0.55 |       0.6  |        0.5 |       0.5 |        99.5 |       5 |
|       0.6  |       0.65 |        0.7 |       1   |        99   |       7 |
|       0.65 |       0.7  |        0.6 |       1.7 |        98.3 |       6 |
|       0.7  |       0.75 |        1.5 |       2.3 |        97.7 |      15 |
|       0.75 |       0.8  |        3.3 |       3.8 |        96.2 |      33 |
|       0.8  |       0.85 |       15.3 |       7.1 |        92.9 |     153 |
|       0.85 |       0.9  |       19   |      22.4 |        77.6 |     190 |
|       0.9  |       0.95 |       30.7 |      41.4 |        58.6 |     307 |
|       0.95 |       1    |       27.9 |      72.1 |        27.9 |     279 |
</details>

### Pub1M

Using a single command (loading the dataset from Huggingface: [Pub1M_OTSL](https://huggingface.co/datasets/ds4sd/Pub1M_OTSL)),

```sh
poetry run python ./docs/examples/benchmark_p1m.py
```

<details>
<summary><b>Table evaluations for Pub1M</b></summary>
<br>

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b Pub1M -i ./benchmarks/Pub1M-dataset/tableformer -o ./benchmarks/Pub1M-dataset/tableformer
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b Pub1M -i ./benchmarks/Pub1M-dataset/tableformer -o ./benchmarks/Pub1M-dataset/tableformer
```

|   x0<=TEDS |   TEDS<=x1 |   prob [%] |   acc [%] |   1-acc [%] |   total |
|------------|------------|------------|-----------|-------------|---------|
|       0    |       0.05 |        1.3 |       0   |       100   |      13 |
|       0.05 |       0.1  |        0.8 |       1.3 |        98.7 |       8 |
|       0.1  |       0.15 |        0.2 |       2.1 |        97.9 |       2 |
|       0.15 |       0.2  |        0.2 |       2.3 |        97.7 |       2 |
|       0.2  |       0.25 |        0   |       2.5 |        97.5 |       0 |
|       0.25 |       0.3  |        0   |       2.5 |        97.5 |       0 |
|       0.3  |       0.35 |        0.3 |       2.5 |        97.5 |       3 |
|       0.35 |       0.4  |        0   |       2.8 |        97.2 |       0 |
|       0.4  |       0.45 |        0.1 |       2.8 |        97.2 |       1 |
|       0.45 |       0.5  |        0.3 |       2.9 |        97.1 |       3 |
|       0.5  |       0.55 |        0.8 |       3.2 |        96.8 |       8 |
|       0.55 |       0.6  |        1.6 |       4   |        96   |      16 |
|       0.6  |       0.65 |        1.6 |       5.6 |        94.4 |      16 |
|       0.65 |       0.7  |        2.3 |       7.2 |        92.8 |      23 |
|       0.7  |       0.75 |        4.6 |       9.5 |        90.5 |      46 |
|       0.75 |       0.8  |       10.8 |      14.1 |        85.9 |     108 |
|       0.8  |       0.85 |       15.3 |      24.9 |        75.1 |     153 |
|       0.85 |       0.9  |       21.6 |      40.2 |        59.8 |     216 |
|       0.9  |       0.95 |       22.9 |      61.8 |        38.2 |     229 |
|       0.95 |       1    |       15.3 |      84.7 |        15.3 |     153 |
</details>

### PubTabNet

Using a single command (loading the dataset from Huggingface: [Pubtabnet_OTSL](https://huggingface.co/datasets/ds4sd/Pubtabnet_OTSL)),

```sh
poetry run python ./docs/examples/benchmark_pubtabnet.py
```

<details>
<summary><b>Table evaluations for Pubtabnet</b></summary>
<br>

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b Pubtabnet -i ./benchmarks/pubtabnet-dataset/tableformer -o ./benchmarks/pubtabnet-dataset/tableformer
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b Pubtabnet -i ./benchmarks/pubtabnet-dataset/tableformer -o ./benchmarks/pubtabnet-dataset/tableformer
```

The final result (struct only here) can be visualised as,

|   x0<=TEDS |   TEDS<=x1 |   prob [%] |   acc [%] |   1-acc [%] |   total |
|------------|------------|------------|-----------|-------------|---------|
|       0    |       0.05 |       0    |      0    |      100    |       0 |
|       0.05 |       0.1  |       0.01 |      0    |      100    |       1 |
|       0.1  |       0.15 |       0.01 |      0.01 |       99.99 |       1 |
|       0.15 |       0.2  |       0.02 |      0.02 |       99.98 |       2 |
|       0.2  |       0.25 |       0    |      0.04 |       99.96 |       0 |
|       0.25 |       0.3  |       0    |      0.04 |       99.96 |       0 |
|       0.3  |       0.35 |       0    |      0.04 |       99.96 |       0 |
|       0.35 |       0.4  |       0    |      0.04 |       99.96 |       0 |
|       0.4  |       0.45 |       0.02 |      0.04 |       99.96 |       2 |
|       0.45 |       0.5  |       0.1  |      0.06 |       99.94 |      10 |
|       0.5  |       0.55 |       0.1  |      0.15 |       99.85 |      10 |
|       0.55 |       0.6  |       0.24 |      0.25 |       99.75 |      25 |
|       0.6  |       0.65 |       0.47 |      0.49 |       99.51 |      49 |
|       0.65 |       0.7  |       1.04 |      0.96 |       99.04 |     108 |
|       0.7  |       0.75 |       2.44 |      2    |       98    |     254 |
|       0.75 |       0.8  |       4.65 |      4.44 |       95.56 |     483 |
|       0.8  |       0.85 |      13.71 |      9.09 |       90.91 |    1425 |
|       0.85 |       0.9  |      21.2  |     22.8  |       77.2  |    2204 |
|       0.9  |       0.95 |      28.48 |     43.99 |       56.01 |    2961 |
|       0.95 |       1    |      27.53 |     72.47 |       27.53 |    2862 |
</details>


## DP-Bench

[See DP-Bench benchmarks](docs/DP-Bench_benchmarks.md)


## OmniDocBench

[See OmniDocBench benchmarks](docs/OmniDocBench_benchmarks.md)


## Contributing

Please read [Contributing to Docling](https://github.com/DS4SD/docling/blob/main/CONTRIBUTING.md) for details.

## License

The Docling codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.

## IBM ❤️ Open Source AI

Docling-eval has been brought to you by IBM.
