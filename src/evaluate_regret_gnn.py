# MLLoopOptSelector/src/evaluate_regret_gnn.py
"""
Compute runtime regret for GNN predictions:
- Inputs: artifacts/dataset.csv, artifacts/gnn_pred.csv
- Outputs: artifacts/gnn_regret_summary.json, artifacts/gnn_regret_details.csv
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

EPS = 1e-9

def _safe_div(a, b):
    a = float(a); b = float(b)
    if b <= 0.0:
        return np.nan
    return a / b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="artifacts/dataset.csv")
    ap.add_argument("--pred_csv", default="artifacts/gnn_pred.csv")
    ap.add_argument("--summary_out", default="artifacts/gnn_regret_summary.json")
    ap.add_argument("--details_out", default="artifacts/gnn_regret_details.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    pred = pd.read_csv(args.pred_csv)

    # best time per config
    best = df.loc[df.groupby("group_id")["time_sec"].idxmin(), ["group_id","time_sec"]]\
             .rename(columns={"time_sec":"best_time"})
    times = df[["group_id","choice","time_sec"]]
    base  = times[times["choice"]=="baseline"][["group_id","time_sec"]]\
            .rename(columns={"time_sec":"baseline_time"})

    merged = pred.merge(best, on="group_id", how="left")
    # predicted time
    pred_time = times.merge(pred.rename(columns={"pred_choice":"choice"}),
                            on=["group_id","choice"], how="inner")[["group_id","time_sec"]]\
                     .rename(columns={"time_sec":"pred_time"})
    merged = merged.merge(pred_time, on="group_id", how="left").merge(base, on="group_id", how="left")
    merged = merged.dropna(subset=["pred_time","best_time","baseline_time"]).copy()

    merged["regret_strict"] = merged.apply(lambda r: _safe_div(r["pred_time"], r["best_time"]), axis=1)
    merged["speedup_vs_baseline_strict"] = merged.apply(lambda r: _safe_div(r["baseline_time"], r["pred_time"]), axis=1)
    merged["regret"] = (merged["pred_time"] + EPS) / (merged["best_time"] + EPS)
    merged["speedup_vs_baseline"] = (merged["baseline_time"] + EPS) / (merged["pred_time"] + EPS)

    summary = {
        "n_configs": int(len(merged)),
        "regret_mean": float(merged["regret"].mean()),
        "regret_median": float(merged["regret"].median()),
        "regret_90p": float(merged["regret"].quantile(0.90)),
        "speedup_vs_baseline_mean": float(merged["speedup_vs_baseline"].mean()),
        "speedup_vs_baseline_median": float(merged["speedup_vs_baseline"].median()),
        "strict_regret_mean": float(np.nanmean(merged["regret_strict"])),
        "strict_speedup_vs_baseline_mean": float(np.nanmean(merged["speedup_vs_baseline_strict"])),
    }
    Path(args.details_out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.details_out, index=False)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved details to {args.details_out}")

if __name__ == "__main__":
    main()
