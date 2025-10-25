# MLLoopOptSelector/src/eval_cross_arch.py
"""
Cross-architecture transfer eval:
- Load a pre-trained model (e.g., artifacts/models_meta/<archA>.joblib)
- Evaluate on a *different* dataset CSV (collected on archB)
Reports accuracy/F1 on best_choice and ε-regularized runtime regret.
"""

import argparse, json, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from features import FEATURE_COLUMNS

EPS = 1e-9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_in", required=True, help="trained model .joblib (typically from artifacts/models_meta)")
    ap.add_argument("--data_csv", required=True, help="dataset.csv from another machine/arch")
    ap.add_argument("--out_json", default="artifacts/xarch.json")
    args = ap.parse_args()

    Path("artifacts").mkdir(exist_ok=True)

    # load
    bundle = joblib.load(args.model_in)
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
    arch = bundle.get("arch", "unknown")

    df = pd.read_csv(args.data_csv)
    cfg = df.drop_duplicates(subset=["group_id"]).copy()

    X = cfg[feature_columns]
    y = cfg["best_choice"]

    # label metrics
    yhat = model.predict(X)
    acc = accuracy_score(y, yhat) if len(y) else float("nan")
    f1m = f1_score(y, yhat, average="macro", zero_division=0) if len(y) else float("nan")

    # runtime regret metrics (ε-regularized)
    times = df[["group_id","choice","time_sec"]]
    best = df.loc[df.groupby("group_id")["time_sec"].idxmin(), ["group_id","time_sec"]].rename(columns={"time_sec":"best_time"})
    pred_df = cfg[["group_id"]].copy(); pred_df["pred_choice"] = yhat

    pred_time = times.merge(pred_df, left_on=["group_id","choice"], right_on=["group_id","pred_choice"], how="inner")
    pred_time = pred_time[["group_id","time_sec"]].rename(columns={"time_sec":"pred_time"})

    merged = cfg.merge(best, on="group_id", how="left").merge(pred_time, on="group_id", how="left")
    base_time = times[times["choice"]=="baseline"][["group_id","time_sec"]].rename(columns={"time_sec":"baseline_time"})
    merged = merged.merge(base_time, on="group_id", how="left")
    merged = merged.dropna(subset=["best_time","pred_time","baseline_time"])

    merged["regret"] = (merged["pred_time"] + EPS) / (merged["best_time"] + EPS)
    merged["speedup_vs_baseline"] = (merged["baseline_time"] + EPS) / (merged["pred_time"] + EPS)

    out = {
        "model_arch": arch,
        "data_csv": args.data_csv,
        "n_configs": int(len(merged)),
        "label_acc": float(acc),
        "label_f1_macro": float(f1m),
        "regret_mean": float(merged["regret"].mean()) if len(merged) else float("nan"),
        "regret_median": float(merged["regret"].median()) if len(merged) else float("nan"),
        "speedup_vs_baseline_median": float(merged["speedup_vs_baseline"].median()) if len(merged) else float("nan"),
    }

    with open(args.out_json, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
