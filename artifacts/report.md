# MLLoopOptSelector Report

## Training

```json
{
  "train_acc": 0.9455587392550143,
  "test_acc": 0.8425925925925926,
  "train_f1_macro": 0.919488061237454,
  "test_f1_macro": 0.3964898689464995,
  "val_med_regret": 1.0,
  "select_by": "f1",
  "classes": [
    "baseline",
    "omp",
    "tile_omp",
    "tile_unroll_omp",
    "unroll",
    "vec_pragma"
  ],
  "confusion_matrix": [
    [
      13,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      67,
      0,
      0,
      0,
      0,
      351
    ]
  ],
  "classification_report": {
    "baseline": {
      "precision": 0.16049382716049382,
      "recall": 1.0,
      "f1-score": 0.276595744680851,
      "support": 13.0
    },
    "unroll": {
      "precision": 0.0,
      "recall": 0.0,
      "f1-score": 0.0,
      "support": 1.0
    },
    "vec_pragma": {
      "precision": 1.0,
      "recall": 0.8397129186602871,
      "f1-score": 0.9128738621586476,
      "support": 418.0
    },
    "accuracy": 0.8425925925925926,
    "macro avg": {
      "precision": 0.3868312757201646,
      "recall": 0.613237639553429,
      "f1-score": 0.3964898689464995,
      "support": 432.0
    },
    "weighted avg": {
      "precision": 0.9724222679469593,
      "recall": 0.8425925925925926,
      "f1-score": 0.8916134700536245,
      "support": 432.0
    }
  },
  "label_counts": {
    "vec_pragma": 648,
    "baseline": 206,
    "omp": 134,
    "tile_unroll_omp": 100,
    "tile_omp": 23,
    "unroll": 19
  },
  "n_train": 698,
  "n_test": 432,
  "cost_weighting": true,
  "class_balance": "balanced",
  "model_picked": "ExtraTrees (sel_by=f1)",
  "kfold": 5,
  "cv_acc_mean": 0.7345958073806175,
  "cv_f1_macro_mean": 0.44326220755193163
}
```

## LOKO

```json
{
  "matmul": {
    "acc": 0.9893617021276596,
    "f1_macro": 0.6645702306079665,
    "model_family": "RandomForest",
    "n": 94,
    "n_covered": 94,
    "coverage": 1.0,
    "regret_mean": 1.0451255664913628,
    "regret_median": 1.0,
    "speedup_median": 6.595285685676883
  },
  "conv1d": {
    "acc": 0.8976063829787234,
    "f1_macro": 0.7123301985370952,
    "model_family": "RandomForest",
    "n": 752,
    "n_covered": 752,
    "coverage": 1.0,
    "regret_mean": 1.354922507280364,
    "regret_median": 1.0,
    "speedup_median": 2.5709797200799773
  },
  "stencil2d": {
    "acc": 1.0,
    "f1_macro": 1.0,
    "model_family": "RandomForest",
    "n": 176,
    "n_covered": 176,
    "coverage": 1.0,
    "regret_mean": 1.0,
    "regret_median": 1.0,
    "speedup_median": 1.0194190547572686
  }
}
```

## Cross-Validation

```json
{
  "k": 5,
  "acc_mean": 0.7420969794387515,
  "f1_macro_mean": 0.4588028043114849
}
```

## GNN (Training)

```json
{
  "acc": 0.6232394366197183,
  "f1_macro": 0.38929762064347706,
  "n_train": 850,
  "n_test": 284
}
```

## GNN (Regret Summary)

```json
{
  "n_configs": 1134,
  "regret_mean": 22.739827250404794,
  "regret_median": 1.0,
  "regret_90p": 1.4995004995004995,
  "speedup_vs_baseline_mean": 4.97608308948535,
  "speedup_vs_baseline_median": 2.998001998001998,
  "strict_regret_mean": 1.1391879799650468,
  "strict_speedup_vs_baseline_mean": 3.6657135847404714
}
```
