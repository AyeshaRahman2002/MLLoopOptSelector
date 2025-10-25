# MLLoopOptSelector/src/evaluate_regret.py
"""
Evaluate the scheduler in *runtime* terms:
- regret = time(predicted) / time(best)
- speedup_vs_baseline = time(baseline) / time(predicted)
Handles zero/near-zero times robustly.
"""

import argparse, json, joblib, pandas as pd
import numpy as np
from pathlib import Path
from features import FEATURE_COLUMNS

EPS = 1e-9  # guard against divide-by-zero and timer underflow

def _safe_div(a, b):
    return float(a) / float(b) if float(b) != 0.0 else np.inf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, default="artifacts/dataset.csv")
    ap.add_argument("--model_in", type=str, default="artifacts/model.joblib")
    ap.add_argument("--summary_out", type=str, default="artifacts/regret_summary.json")
    ap.add_argument("--details_out", type=str, default="artifacts/regret_details.csv")
    args = ap.parse_args()

    Path("artifacts").mkdir(exist_ok=True)
    df = pd.read_csv(args.data_csv)

    # one row per config
    cfg = df.drop_duplicates(subset=["group_id"]).copy()
    X = cfg[FEATURE_COLUMNS]
    bundle = joblib.load(args.model_in)
    model = bundle["model"]
    cfg["pred_choice"] = model.predict(X)

    # lookup times
    times = df[["group_id","choice","time_sec"]]
    best = df.loc[df.groupby("group_id")["time_sec"].idxmin(), ["group_id","time_sec"]]
    best = best.rename(columns={"time_sec":"best_time"})

    merged = cfg.merge(best, on="group_id", how="left")
    # predicted time
    pred_time = times.merge(
        cfg[["group_id","pred_choice"]],
        left_on=["group_id","choice"],
        right_on=["group_id","pred_choice"],
        how="inner",
    )[["group_id","time_sec"]].rename(columns={"time_sec":"pred_time"})
    merged = merged.merge(pred_time, on="group_id", how="left")
    # baseline time
    base_time = times[times["choice"]=="baseline"][["group_id","time_sec"]].rename(columns={"time_sec":"baseline_time"})
    merged = merged.merge(base_time, on="group_id", how="left")

    # strict metrics (may have inf)
    merged["regret_strict"] = merged.apply(lambda r: _safe_div(r["pred_time"], r["best_time"]), axis=1)
    merged["speedup_vs_baseline_strict"] = merged.apply(lambda r: _safe_div(r["baseline_time"], r["pred_time"]), axis=1)

    # epsilon-regularized metrics
    merged["regret"] = (merged["pred_time"] + EPS) / (merged["best_time"] + EPS)
    merged["speedup_vs_baseline"] = (merged["baseline_time"] + EPS) / (merged["pred_time"] + EPS)

    # clean any NaNs (shouldn’t happen but to be safe)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["regret","speedup_vs_baseline"])

    summary = {
        "n_configs": int(len(merged)),
        # epsilon-regularized (recommended to report)
        "regret_mean": float(merged["regret"].mean()),
        "regret_median": float(merged["regret"].median()),
        "regret_90p": float(merged["regret"].quantile(0.90)),
        "speedup_vs_baseline_mean": float(merged["speedup_vs_baseline"].mean()),
        "speedup_vs_baseline_median": float(merged["speedup_vs_baseline"].median()),
        # optional: strict (may include inf if true zeros occur)
        "strict_regret_mean": float(np.nanmean(merged["regret_strict"])),
        "strict_speedup_vs_baseline_mean": float(np.nanmean(merged["speedup_vs_baseline_strict"])),
    }

    merged.to_csv(args.details_out, index=False)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved details to {args.details_out}")

if __name__ == "__main__":
    main()
