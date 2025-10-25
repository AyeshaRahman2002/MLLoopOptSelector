# ML Loop Optimization Selector (MLLoopOptSelector)

This project automates **kernel-level optimization selection** for compute-intensive workloads such as `matmul`, `conv1d`, and `stencil2d`.  
It combines traditional ML models, meta-learners, and GNNs to predict the best tile, unroll, and optimization parameters for each kernel configuration — minimizing runtime (regret) while maintaining high accuracy and generality.

## Features

- **Data collection**: Runs tile configurations and gathers runtime metrics.  
- **Model training**: Uses scikit-learn (ExtraTrees, XGB, etc.) and GNNs to learn optimal scheduling patterns.  
- **Cross-validation & model comparison**: Auto-selects models by F1, regret, or speedup metrics.  
- **LOKO evaluation**: Per-kernel model quality and runtime speedup analysis.  
- **GNN integration**: Encodes kernel dependency graphs for meta-learning.  
- **Hybrid selector**: Combines ML and rule-based confidence for fallback.  
- **Visualization**: Generates accuracy/F1/regret and runtime 3D plots.  
- **Cross-architecture evaluation**: Test one model on data from another machine.

## Setup

### 1. Install Python environment

You can use `conda` or `venv`:

```bash
conda create -n mlloop python=3.11
conda activate mlloop
pip install -r requirements.txt
```

For macOS users, install OpenMP first:
```bash
brew install libomp
```

Then make sure the Makefile can find it:
```bash
export OMP_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp"
```

## Project Structure
```
MLLoopOptSelector/
│
├── src/                     # Core Python source files
│   ├── collect_data.py      # Collects runtime data
│   ├── train_model.py       # Trains baseline ML model
│   ├── predict.py           # Runs prediction for given kernel
│   ├── eval_loko.py         # Per-kernel LOKO evaluation
│   ├── train_gnn.py         # Graph neural network model
│   ├── gnn_predict_all.py   # Apply trained GNN to all kernels
│   └── ...                  # Other analysis scripts
│
├── artifacts/               # Generated datasets, models, and reports
│   ├── dataset.csv
│   ├── model.joblib
│   ├── loko.json
│   ├── gnn_model.pt
│   ├── gnn_regret_details.csv
│   └── ...
│
├── Makefile                 # One-command automation for experiments
└── README.md                # This file
```

## Usage (Makefile Targets)
Quick Start
```bash
make workflow
```

Equivalent to:
```bash
make data
make train
```

---

## 🚀 Usage (Makefile Targets)

### Quick Start

```bash
make all
```

Equivalent to:

```bash
make data
make train
```

### Key Commands

| Target           | Description                                              |
| ---------------- | -------------------------------------------------------- |
| `make data`      | Collects dataset (`artifacts/dataset.csv`)               |
| `make train`     | Trains ML model and saves report                         |
| `make predict`   | Runs prediction on a sample kernel                       |
| `make loko`      | Evaluates per-kernel model accuracy, F1, regret, speedup |
| `make gnn`       | Full GNN pipeline: graphify → train → predict → evaluate |
| `make vis3d`     | 3D runtime visualization                                 |
| `make explain`   | Feature importance and explanation plots                 |
| `make fewshot`   | Few-shot fine-tuning for new kernels                     |
| `make meta`      | Train meta-learners for transfer learning                |
| `make report`    | Generate Markdown performance report                     |
| `make clean`     | Remove temporary files                                   |
| `make deepclean` | Remove caches and pyc files                              |
| `make distclean` | Remove all artifacts entirely                            |

## Outputs

Each workflow generates structured results under `artifacts/`:

| File                      | Purpose                                    |
| ------------------------- | ------------------------------------------ |
| `dataset.csv`             | Full runtime dataset                       |
| `model.joblib`            | Trained ML model                           |
| `report.json`             | Model metrics (accuracy, F1, regret, etc.) |
| `loko.json`               | LOKO (Leave-One-Kernel-Out) evaluation     |
| `gnn_model.pt`            | Trained GNN                                |
| `gnn_regret_summary.json` | GNN summary metrics                        |
| `gnn_regret_details.csv`  | Per-sample GNN predictions and regrets     |
| `report.md`               | Human-readable summary                     |

## Example Results (Typical)

| Kernel    | Accuracy | F1 (macro) | Median Speedup | Median Regret |
| --------- | -------- | ---------- | -------------- | ------------- |
| matmul    | 0.989    | 0.665      | 6.595×         | 1.000         |
| conv1d    | 0.898    | 0.712      | 2.571×         | 1.000         |
| stencil2d | 1.000    | 1.000      | 1.019×         | 1.000         |

## Tips

* Always run from project root so paths resolve correctly.
* Use `make collect-fast` for multi-core data collection.
* If OpenMP builds fail on macOS, reinstall `libomp` and re-export `OMP_FLAGS`.
* Regret = `pred_time / best_time` — lower is better.
* Speedup = `baseline_time / pred_time` — higher is better.

## Citation

If you use this in research or benchmarking:

```text
@software{rahman2025_mlloopoptselector,
  title = {ML Loop Optimization Selector},
  author = {Ayesha Rahman},
  year = {2025},
  url = {https://github.com/AyeshaRahman2002/MLLoopOptSelector}
}
```
