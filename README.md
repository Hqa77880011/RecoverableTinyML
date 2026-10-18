# RecoverableTinyML

Implementation of **Recoverability-Aware Checkpoint Scheduling under Energy and Storage Budgets for Intermittent TinyML**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data

The bundled integer-operator device model provides the power trace, capture times, stage costs, SRAM/NVM limits and lifetime. It is a trace simulation input.

## Run

```bash
recoverable-tinyml execute --input examples/finite_device.json --output runs/full --variant full
```

## Ablations

Use the same input profile for each variant: `periodic`, `no_lifetime`.

```bash
recoverable-tinyml execute --input examples/finite_device.json --output runs/periodic --variant periodic
```

## Outputs

`runs/full/execution.json` stores completion/expiry events, checkpoint traffic and energy accounting. Input and output paths are controlled by `--input` and `--output`.
