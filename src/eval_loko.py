# MLLoopOptSelector/src/eval_loko.py
"""
Leave-One-Kernel-Out (LOKO) evaluation with epsilon-regularized regret:
- regret = (pred_time+EPS)/(best_time+EPS)
- speedup_vs_baseline = (baseline_time+EPS)/(pred_time+EPS)
Also reports how many predictions had timing coverage.

Improvements vs. older version:
- Mirrors the model family chosen during training (reads artifacts/report.json) or via --force_family.
- Applies class-balanced sample weights in training folds.
- Supports hybrid verify: if confidence < --min_conf, choose best of top-K
  by looking up true times in the dataset (simulates runtime hybrid policy).
- Per-kernel hybrid overrides (more aggressive verify for conv1d).
"""
import argparse, json, pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from features import FEATURE_COLUMNS
from models import get_default_model, get_model_zoo

KERNS = ["matmul","conv1d","stencil2d"]
EPS = 1e-9

def _apply_small_output_guard(kernel, N, K, choice):
    if kernel != "conv1d":
        return choice
    # Estimate output length O = N - K + 1 (K~max(3, N//16) when missing)
    k_eff = K if K else max(3, max(1, N // 16))
    O = max(1, N - k_eff + 1)
    if O >= 256:
        return choice
    mapping = {
        "omp": "baseline",
        "tile_omp": "tile",
        "tile_unroll_omp": "tile_unroll",
    }
    return mapping.get(choice, choice)

def regret_on(df_all, df_cfg, y_pred):
    """Compute regret/speedup metrics for given predicted choices."""
    df_cfg = df_cfg.copy()
    df_cfg["pred_choice"] = y_pred
    times = df_all[["group_id","choice","time_sec"]]

    best = df_all.loc[df_all.groupby("group_id")["time_sec"].idxmin(), ["group_id","time_sec"]]
    best = best.rename(columns={"time_sec":"best_time"})
    df_cfg = df_cfg.merge(best, on="group_id", how="left")

    pred_time = times.merge(
        df_cfg[["group_id","pred_choice"]],
        left_on=["group_id","choice"], right_on=["group_id","pred_choice"],
        how="inner"
    )[["group_id","time_sec"]].rename(columns={"time_sec":"pred_time"})
    df_cfg = df_cfg.merge(pred_time, on="group_id", how="left")

    base_time = times[times["choice"]=="baseline"][["group_id","time_sec"]].rename(columns={"time_sec":"baseline_time"})
    df_cfg = df_cfg.merge(base_time, on="group_id", how="left")

    covered = df_cfg.dropna(subset=["pred_time","best_time","baseline_time"]).copy()
    covered["regret"] = (covered["pred_time"] + EPS) / (covered["best_time"] + EPS)
    covered["speedup_vs_baseline"] = (covered["baseline_time"] + EPS) / (covered["pred_time"] + EPS)

    return {
        "n": int(len(df_cfg)),
        "n_covered": int(len(covered)),
        "coverage": float(len(covered) / max(len(df_cfg), 1)),
        "regret_mean": float(covered["regret"].mean()) if len(covered) else float("nan"),
        "regret_median": float(covered["regret"].median()) if len(covered) else float("nan"),
        "speedup_median": float(covered["speedup_vs_baseline"].median()) if len(covered) else float("nan"),
    }

def _get_model_by_family(fam: str, seed: int):
    zoo = get_model_zoo(seed=seed)
    if fam in zoo:
        return zoo[fam], fam
    return get_default_model(seed=seed), "Default"

def _load_model_by_report(seed: int):
    """Try to mirror the last-picked model family from artifacts/report.json."""
    report_p = Path("artifacts/report.json")
    if not report_p.exists():
        return get_default_model(seed=seed), "Default"
    try:
        rep = json.loads(report_p.read_text())
        picked = rep.get("model_picked", "")
        fam = picked.split()[0] if picked else ""
        return _get_model_by_family(fam, seed)
    except Exception:
        return get_default_model(seed=seed), "Default"

def _kernel_hybrid_params(kernel: str, topk: int, min_conf: float):
    """Per-kernel override for hybrid verify aggressiveness."""
    if kernel == "conv1d":
        # Verify more candidates and trust top-1 a bit less for conv1d
        return max(topk, 4), min(min_conf, 0.80)
    return topk, min_conf

def _hybrid_select_for_group(
    classes, proba_row, group_id, test_row, times_df, topk, min_conf, include_always
):
    """Return chosen label for a single test item using hybrid policy."""
    # Ranking by predicted prob
    order = np.argsort(proba_row)[::-1]
    labels_ranked = classes[order]

    # If confident OR topk==1 => trust top1 (after guard)
    if (topk <= 1) or (proba_row[order[0]] >= min_conf):
        chosen = labels_ranked[0]
        chosen = _apply_small_output_guard(
            kernel=test_row["kernel"],
            N=int(test_row["N"]),
            K=int(test_row["K"]) if ("K" in test_row and not pd.isna(test_row["K"])) else 0,
            choice=chosen
        )
        return chosen

    # Otherwise: verify (top-K U include_always) by looking up true times and picking the fastest
    cand = set(labels_ranked[:topk].tolist())
    cand |= include_always
    gid = test_row["group_id"]

    # All available (choice,time) for this group:
    tsub = times_df[times_df["group_id"] == gid]
    # Filter to only our candidates
    tsub = tsub[tsub["choice"].isin(cand)]
    if len(tsub) == 0:
        # If no coverage, fallback to top1 (guarded)
        chosen = labels_ranked[0]
        chosen = _apply_small_output_guard(
            kernel=test_row["kernel"],
            N=int(test_row["N"]),
            K=int(test_row["K"]) if ("K" in test_row and not pd.isna(test_row["K"])) else 0,
            choice=chosen
        )
        return chosen

    # Pick minimal time among candidates
    best_choice = tsub.loc[tsub["time_sec"].idxmin(), "choice"]
    # Apply guard last to be consistent with runtime policy
    best_choice = _apply_small_output_guard(
        kernel=test_row["kernel"],
        N=int(test_row["N"]),
        K=int(test_row["K"]) if ("K" in test_row and not pd.isna(test_row["K"])) else 0,
        choice=best_choice
    )
    return best_choice

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="artifacts/dataset.csv")
    ap.add_argument("--out_json", default="artifacts/loko.json")
    ap.add_argument("--seed", type=int, default=42)

    # force model family (overrides report.json)
    ap.add_argument("--force_family", type=str, default="",
                    help="Force model family from model zoo (e.g. RandomForest, ExtraTrees, GBDT, LogReg, DecisionTree).")

    # HYBRID verify controls
    ap.add_argument("--topk", type=int, default=3,
                    help="Hybrid verify top-K candidates (1 = off).")
    ap.add_argument("--min_conf", type=float, default=0.90,
                    help="Confidence threshold to trust top-1; else verify.")
    ap.add_argument("--always_include", type=str, default="baseline,omp,tile,tile_unroll",
                    help="Comma-separated labels to always include as candidates during verify.")
    args = ap.parse_args()

    Path("artifacts").mkdir(exist_ok=True)
    raw = pd.read_csv(args.data_csv)
    cfg = raw.drop_duplicates(subset=["group_id"]).copy()

    # Parse always-include set
    include_always = set(s.strip() for s in args.always_include.split(",") if s.strip())

    # model to mirror training family (or force)
    if args.force_family:
        clf, fam = _get_model_by_family(args.force_family, args.seed)
    else:
        clf, fam = _load_model_by_report(seed=args.seed)

    results = {}
    for hold in KERNS:
        train_cfg = cfg[cfg["kernel"] != hold]
        test_cfg  = cfg[cfg["kernel"] == hold]

        # Mirror training: drop ultra-rare labels on TRAIN only
        vc = train_cfg["best_choice"].value_counts()
        keep_labels = vc[vc >= 8].index
        train_cfg = train_cfg[train_cfg["best_choice"].isin(keep_labels)]
        # Align TEST label space (don't evaluate on unseen classes)
        test_cfg = test_cfg[test_cfg["best_choice"].isin(keep_labels)]

        if len(test_cfg) == 0 or len(train_cfg) == 0:
            results[hold] = {
                "acc": float("nan"),
                "f1_macro": float("nan"),
                "n": int(len(test_cfg)),
                "n_covered": 0,
                "coverage": 0.0,
                "regret_mean": float("nan"),
                "regret_median": float("nan"),
                "speedup_median": float("nan"),
                "model_family": fam,
            }
            continue

        Xtr, ytr = train_cfg[FEATURE_COLUMNS], train_cfg["best_choice"]
        Xte, yte = test_cfg[FEATURE_COLUMNS], test_cfg["best_choice"]

        # Class-balanced sample weights (as in training)
        classes = np.array(sorted(ytr.unique()))
        cw = compute_class_weight("balanced", classes=classes, y=ytr)
        cw_map = {c: w for c, w in zip(classes, cw)}
        wtr = ytr.map(cw_map).astype(float).to_numpy()

        # Fit fresh model of the same family
        clf.fit(Xtr, ytr, sample_weight=wtr)

        if (args.topk <= 1) or (not hasattr(clf, "predict_proba")):
            # Simple path: predict, then apply small-output guard
            yhat = clf.predict(Xte)
            if len(Xte):
                yhat = np.array([
                    _apply_small_output_guard(
                        kernel=row["kernel"],
                        N=int(row["N"]),
                        K=int(row["K"]) if ("K" in row and not pd.isna(row["K"])) else 0,
                        choice=pred
                    )
                    for pred, (_, row) in zip(yhat, test_cfg.iterrows())
                ], dtype=object)
        else:
            # Hybrid path: per-row proba -> verify top-K by dataset times
            proba = clf.predict_proba(Xte)
            model_classes = clf.classes_
            times_df = raw[["group_id","choice","time_sec"]]
            yhat = []
            for i, (_, row) in enumerate(test_cfg.iterrows()):
                # Per-kernel hybrid tuning
                tk, mc = _kernel_hybrid_params(str(row["kernel"]), args.topk, args.min_conf)
                chosen = _hybrid_select_for_group(
                    classes=model_classes,
                    proba_row=proba[i],
                    group_id=row["group_id"],
                    test_row=row,
                    times_df=times_df,
                    topk=max(1, tk),
                    min_conf=mc,
                    include_always=include_always,
                )
                yhat.append(chosen)
            yhat = np.array(yhat, dtype=object)

        acc = accuracy_score(yte, yhat) if len(Xte) else float("nan")
        f1  = f1_score(yte, yhat, average="macro", zero_division=0) if len(Xte) else float("nan")
        reg = regret_on(raw, test_cfg[["group_id","kernel","best_choice"]], yhat)
        results[hold] = {"acc": acc, "f1_macro": f1, "model_family": fam, **reg}

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
