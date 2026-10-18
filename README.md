# NewtonCompact

Implementation of **Newton-Consistent Neural Compact Models for Reliable Cross-Technology MOSFET Simulation**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Generate independent cubic device currents, linear charges and analytic stamps with two technology descriptors.

```bash
newton-compact demo --output data/generated.npz --count 96 --config config.json --seed 13
newton-compact prepare --input data/generated.npz --output data/prepared
```

`x[N,V]` contains independent terminal voltages, `technology[N,P]` known device descriptors, `y[N,V]` independent reference currents, `charge[N,V]` reference charges, and `stamp[N,V,V]` reference G+alpha*C. Use measured/TCAD/compact-model sweeps with technology-locked split groups. Current and charge units must be normalized consistently with alpha and stamp.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

## Training and inference

```bash
newton-compact train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

The bundled nodal circuit uses four terminal nodes per device and two technology descriptors:

```bash
python -m newton_compact.solver --model runs/full/model.pt --netlist examples/netlist.json --output runs/full/circuit.json
```

## Evaluation

```bash
newton-compact evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

## Ablations

Variants: `full`, `baseline`, `no_preconditioning`, `no_charge`. Reuse the prepared data and training seed:

```bash
newton-compact train --input data/prepared --output runs/baseline --config config.json --variant baseline --epochs 100 --batch-size 16 --seed 13
newton-compact evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
newton-compact plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.
`runs/full/circuit.json` records solved node voltages, convergence, iterations and KCL residual.
