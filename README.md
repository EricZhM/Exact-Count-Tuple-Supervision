# NTMP: n-Tuple with m-Positive

This repository is the minimal implementation accompanying **Exact-Count Tuple
Supervision for Label-Efficient Classification: Unbiased Risk Estimation and
Guarantees**. NTMP trains a binary classifier from two weakly supervised
sources: an exact-count tuple set `D_T`, where every length-`n` tuple contains
exactly `m` positive instances, and a reference set `D_U` with known positive
class prior `pi`.

The implementation includes NTMP-URE and the NTMP-ABS/NTMP-ReLU corrections,
the original MLP and ResNet-20 architectures, deterministic weak-data
construction, and final Accuracy, Macro-F1, and AUROC evaluation. Instance
labels are used only by the offline constructor and official test evaluation;
the training dataset interfaces expose features only.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data preparation

The CLI uses official torchvision train/test splits for MNIST, Fashion-MNIST,
CIFAR-10, and SVHN. Following the original NTMP experiments, odd original
class IDs are positive and even IDs are negative. Pass the dataset root with
`--data-dir`. Existing torchvision data are reused; add `--download` only when
downloads are desired.

`D_T` and `D_U` are static, disjoint, sampled without replacement, and saved as
an index manifest below the output directory. The constructor requires
`reference_size * pi` to be an integer, so `D_U` has exactly the requested
prior.

## Training

NTMP-URE:

```bash
python train.py --dataset mnist --data-dir /path/to/data --n 5 --m 2 --pi 0.8 --correction ure --seed 41
```

NTMP-ABS:

```bash
python train.py --dataset mnist --data-dir /path/to/data --n 5 --m 2 --pi 0.8 --correction abs --seed 41
```

NTMP-ReLU:

```bash
python train.py --dataset cifar10 --data-dir /path/to/data --n 5 --m 2 --pi 0.8 --correction relu --seed 41
```

The same commands work on Windows, for example `--data-dir D:\data`.

Important arguments:

- `--dataset`: `mnist`, `fashionmnist`, `cifar10`, or `svhn`.
- `--n`, `--m`: tuple length and exact positive count; `alpha=m/n`.
- `--pi`: known positive prior of `D_U`; it must differ from `m/n`.
- `--correction`: `ure`, `abs`, or `relu`.
- `--tuple-count`, `--reference-size`: default to 1,000 each, matching the original core protocol.
- `--epochs`, `--batch-size`: default to 100 epochs and 256 complete tuples per update.
- `--seed`: controls source construction, model initialization, and epoch shuffling.
- `--device`: `auto`, `cpu`, or `cuda`.
- `--output-dir`: stores the manifest, configuration, history, final metrics, and last checkpoint.

The original optimizer protocol is retained: Adam with learning rates `1e-5`
(MNIST), `1e-4` (Fashion-MNIST and CIFAR-10), and `1e-3` (SVHN), with input
scaling to `[0,1]`, no augmentation, and last-epoch reporting. Test labels are
never used for gradient updates or model selection.

## Evaluation and outputs

Each epoch reports Accuracy, Macro-F1, and rank-based AUROC on the official test
split. Continuous sigmoid scores are used for AUROC and a zero-logit (`0.5`
probability) threshold is used for Accuracy and Macro-F1. A run directory
contains `config.json`, `history.csv`, `result.json`, and `checkpoint_last.pt`.
No pretrained weights or claimed paper results are included.

## Tests

The lightweight suite checks imports and CLI parsing, exact-count/disjoint data
construction, all three paper objectives, both model forward/backward passes,
label-free training views, and one synthetic training update:

```bash
pytest -q
```

It does not run a formal experiment or download data.

## Citation

Please cite the accompanying paper:

> *Exact-Count Tuple Supervision for Label-Efficient Classification: Unbiased
> Risk Estimation and Guarantees.*

Add the final author, venue, year, and DOI metadata from the published version
when it becomes available.

## License

The repository code is released under the MIT License. Runtime dependencies are
not vendored: PyTorch and torchvision use BSD-style/BSD-3-Clause licenses,
NumPy uses BSD-3-Clause for its main project code (binary distributions may
include separately licensed components), and pytest uses MIT. Dataset terms
are separate from torchvision's software license; users must check the terms
for each downloaded dataset.
