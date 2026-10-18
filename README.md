# Recoverability-Aware Checkpoint Scheduling under Energy and Storage Budgets for Intermittent TinyML

Implementation of **Recoverability-Aware Checkpoint Scheduling under Energy and Storage Budgets for Intermittent TinyML**.

The execution simulator maintains epoch, capture time, program position, activation, two persistent slots, and an atomic slot selector. A commit becomes authoritative only after its entire payload write completes; interrupted writes retain the old slot. Scheduling combines replay exposure, survival, preservation cost, and lifetime, with hard SRAM/NVM/energy/write guards. The included integer segment workload supplies a deterministic semantic oracle; replace its segment operations and measured costs together for an application graph.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

Cost-model training uses `x[N,D]`, `y[N,1]`. Execute uses a JSON workload of deterministic integer segments (factor,offset,exec_ms,energy,payload,live,scratch), a power trace of alternating on/off durations, offered capture times, lifetime, and resource budgets. Slot payloads retain every state needed by the declared segment workload; metadata and issued partial bytes are charged separately.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
recoverable-tinyml demo --output data/example.npz --count 96 --config config.json
recoverable-tinyml prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
recoverable-tinyml train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
recoverable-tinyml evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
recoverable-tinyml plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
recoverable-tinyml calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `periodic`, `no_lifetime`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
recoverable-tinyml train --input data/prepared --output runs/baseline --config config.json --variant baseline
recoverable-tinyml evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Execute the mechanism

```bash
recoverable-tinyml execute --input example.json --output runs/execution --variant full
```

The example is an explicit analytical/simulated input specification, not an experiment result. Replace it with the schema described above for application data. The execute path writes the actual selected actions, constraints, and mechanism metrics.

## Selected reproduction setting

Use the bundled three-stage integer operator graph and on/off trace, with 4096-byte SRAM, 512-byte writable NVM and a 15-ms input lifetime. Integer semantic equality and the two-slot commit protocol are the exact task; execution/write energy and timing are declared finite-device coefficients. MCU measurements are optional for a separate device setting.

```bash
python -m recoverable_tinyml.cli execute --input examples/finite_device.json --output runs/finite --variant full
```
