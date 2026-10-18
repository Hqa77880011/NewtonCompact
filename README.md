# Newton-Consistent Neural Compact Models for Reliable Cross-Technology MOSFET Simulation

Implementation of **Newton-Consistent Neural Compact Models for Reliable Cross-Technology MOSFET Simulation**.

A technology-conditioned neural core predicts independent terminal currents and charges; the last terminal is the exact negative sum. Automatic differentiation forms G+alpha*C. A reference-preconditioned directional stamp mismatch supplies the Newton-consistency penalty, with a stated regularization floor. The model exports currents, charges, stamps, and directional mismatch for device/circuit probe evaluation.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

`x[N,V]` contains independent terminal voltages, `technology[N,P]` known device descriptors, `y[N,V]` independent reference currents, `charge[N,V]` reference charges, and `stamp[N,V,V]` reference G+alpha*C. Use measured/TCAD/compact-model sweeps with technology-locked split groups. Current and charge units must be normalized consistently with alpha and stamp.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
newton-compact demo --output data/example.npz --count 96 --config config.json
newton-compact prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
newton-compact train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
newton-compact evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
newton-compact plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
newton-compact calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `no_preconditioning`, `no_charge`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
newton-compact train --input data/prepared --output runs/baseline --config config.json --variant baseline
newton-compact evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Selected reproduction setting

Use the existing cubic compact-model data generator and width-32 network. The added solver consumes a finite JSON nodal circuit with conductances, current sources, fixed node voltages and four-terminal learned devices. It assembles conservative full-terminal stamps, applies damped Newton steps and records convergence, iterations and the actual KCL residual. backward_euler_alpha=1/dt and per-device previous_charge enable a declared transient step; zero selects DC. Singular/nonconvergent solves are explicit outcomes.

```bash
python -m newton_compact.solver --model runs/full/model.pt --netlist examples/netlist.json --output runs/full/circuit.json
```

Each device uses nodes=[terminal1,terminal2,terminal3,reference] and technology=[t1,t2] (a length-two vector for the selected network). Fixed-voltage nodes impose the reference; conductances are positive, currents are directed first-to-second, and previous_charge contains four terminal charges from the preceding accepted transient step.
