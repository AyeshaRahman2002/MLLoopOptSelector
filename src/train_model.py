# MLLoopOptSelector/src/train_model.py
"""
Cost/Regret-sensitive training with continuous weighting, model zoo option, grouped & stratified split,
and model selection by F1 / Accuracy / Validation Regret. Also drops ultra-rare classes.
Now also supports automatic class-balance weighting that multiplies with regret weights.
"""
import argparse, json, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from tabulate import tabulate
from typing import Optional
from models import get_default_model, get_model_zoo
from features import FEATURE_COLUMNS

# Optional (sklearn >= 1.1)
try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGKF = True
except Exception:
    _HAS_SGKF = False

EPS = 1e-9  # for epsilon-regularized regret

def compute_regret_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    times = raw_df[["group_id","choice","time_sec"]]
    best = raw_df.loc[raw_df.groupby("group_id")["time_sec"].idxmin(), ["group_id","time_sec"]]
    best = best.rename(columns={"time_sec":"best_time"})
    ranked = raw_df.sort_values(["group_id","time_sec"]).copy()
    # 2nd best time (for regret gap weighting)
    second = ranked.groupby("group_id").nth(1)[["time_sec"]].rename(
        columns={"time_sec":"second_best_time"}
    ).reset_index()
    base = times[times["choice"]=="baseline"][["group_id","time_sec"]].rename(
        columns={"time_sec":"baseline_time"}
    )
    out = best.merge(second, on="group_id", how="left").merge(base, on="group_id", how="left")
    return out

def compute_sample_weights_continuous(raw_df: pd.DataFrame, alpha: float=2.0, cap: float=10.0) -> pd.DataFrame:
    eps = 1e-12
    base = raw_df.drop_duplicates(subset=["group_id"]).copy()
    tbl = compute_regret_table(raw_df)
    base = base.merge(tbl, on="group_id", how="left")
    base["regret_gap"] = (base["second_best_time"] - base["best_time"]) / (base["best_time"] + eps)
    base["regret_gap"] = base["regret_gap"].clip(lower=0.0)
    base["cost_weight"] = (1.0 + alpha * base["regret_gap"]).clip(upper=cap).fillna(1.0)
    return base[["group_id","cost_weight"]]

def _choose_stratify(y, min_per_class=2):
    vc = y.value_counts()
    return y if (len(vc) and vc.min() >= min_per_class) else None

def _combine_weights(
    y_subset: pd.Series,
    base_weights: Optional[np.ndarray],
    balance_mode: str,
) -> Optional[np.ndarray]:
    """
    Multiply regret-based weights (if any) by per-class weights for imbalance.
    Returns None if both components are None/disabled.
    """
    w = None if base_weights is None else np.asarray(base_weights, dtype=float)
    if balance_mode == "balanced" and len(y_subset):
        classes = np.array(sorted(y_subset.unique()))
        # class_weight='balanced' => n_samples / (n_classes * n_samples_in_class)
        cw = compute_class_weight("balanced", classes=classes, y=y_subset)
        cw_map = {c: w_i for c, w_i in zip(classes, cw)}
        cw_vec = y_subset.map(cw_map).astype(float).to_numpy()
        if w is None:
            w = cw_vec
        else:
            w = w * cw_vec
    return w

def kfold_eval(model, X, y, groups=None, weights=None, seed=42, k=5, balance_mode="none"):
    # Use grouped + stratified CV when possible; fallback to plain StratifiedKFold.
    if groups is not None and _HAS_SGKF:
        # reduce k to the feasible number
        min_cls = int(y.value_counts().min())
        k_eff = max(2, min(k, min_cls))
        sgkf = StratifiedGroupKFold(n_splits=k_eff, shuffle=True, random_state=seed)
        accs, f1s = [], []
        for tr, te in sgkf.split(X, y, groups=groups):
            m = get_default_model(seed)
            base_w = None if weights is None else np.asarray(weights.iloc[tr], dtype=float)
            wtr = _combine_weights(y.iloc[tr], base_w, balance_mode)
            m.fit(X.iloc[tr], y.iloc[tr], sample_weight=wtr)
            yhat = m.predict(X.iloc[te])
            accs.append(accuracy_score(y.iloc[te], yhat))
            f1s.append(f1_score(y.iloc[te], yhat, average="macro", zero_division=0))
        return {"cv_acc_mean": float(np.mean(accs)), "cv_f1_macro_mean": float(np.mean(f1s))}
    # fallback
    k_eff = max(2, min(k, len(y.unique()))) if len(y) else 2
    if k_eff < 2:
        return {"cv_acc_mean": None, "cv_f1_macro_mean": None}
    skf = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        m = get_default_model(seed)
        base_w = None if weights is None else np.asarray(weights.iloc[tr], dtype=float)
        wtr = _combine_weights(y.iloc[tr], base_w, balance_mode)
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=wtr)
        yhat = m.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], yhat))
        f1s.append(f1_score(y.iloc[te], yhat, average="macro", zero_division=0))
    return {"cv_acc_mean": float(np.mean(accs)), "cv_f1_macro_mean": float(np.mean(f1s))}

def split_regret(raw_df: pd.DataFrame, g_ids: pd.Series, y_pred: pd.Series) -> float:
    pred_df = pd.DataFrame({"group_id": g_ids.values, "pred_choice": y_pred.values})
    times = raw_df[["group_id","choice","time_sec"]].copy()
    best = raw_df.loc[raw_df.groupby("group_id")["time_sec"].idxmin(), ["group_id","time_sec"]]
    best = best.rename(columns={"time_sec":"best_time"})
    merged = pred_df.merge(best, on="group_id", how="left")

    pred_time = times.merge(pred_df, left_on=["group_id","choice"], right_on=["group_id","pred_choice"], how="inner")
    pred_time = pred_time[["group_id","time_sec"]].rename(columns={"time_sec":"pred_time"})
    merged = merged.merge(pred_time, on="group_id", how="left").dropna(subset=["pred_time","best_time"])

    if merged.empty:
        return float("inf")
    reg = (merged["pred_time"] + EPS) / (merged["best_time"] + EPS)
    return float(reg.median())

def _build_split_groups(df_cfg: pd.DataFrame) -> pd.Series:
    """
    Group key for splits: per-problem size (kernel + N (+ M for stencil/matmul) + K for matmul).
    This prevents train/test leakage across nearly identical configs.
    """
    parts = []
    for _, r in df_cfg.iterrows():
        if r["kernel"] == "matmul":
            parts.append(f"matmul:{int(r['N'])}:{int(r['M'])}:{int(r['K'])}")
        elif r["kernel"] == "stencil2d":
            parts.append(f"stencil2d:{int(r['N'])}:{int(r['M'])}")
        else:  # conv1d
            parts.append(f"conv1d:{int(r['N'])}")
    return pd.Series(parts, index=df_cfg.index, name="split_group")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, default="artifacts/dataset.csv")
    ap.add_argument("--model_out", type=str, default="artifacts/model.joblib")
    ap.add_argument("--report_out", type=str, default="artifacts/report.json")
    ap.add_argument("--report_txt", type=str, default="artifacts/report.txt")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost_weighting", action="store_true", help="use continuous regret-based weights")
    ap.add_argument("--alpha", type=float, default=2.0, help="weight strength")
    ap.add_argument("--cap", type=float, default=10.0, help="max weight cap")
    ap.add_argument("--try_models", action="store_true", help="train & compare several models; save best by chosen criterion")
    ap.add_argument("--select_by", choices=["f1","acc","regret"], default="f1",
                    help="criterion to pick best model from zoo")
    ap.add_argument("--kfold", type=int, default=5, help="optional K-fold CV (report only)")
    ap.add_argument("--calibrate", action="store_true",
                    help="Calibrate probabilities (sigmoid). Ignored if sample_weight is used.")
    ap.add_argument("--min_class_count", type=int, default=8,
                    help="Drop labels with < this many examples before training.")
    ap.add_argument("--class_balance", choices=["none","balanced"], default="balanced",
                    help="Apply per-class weights (multiplied with regret weights).")
    args = ap.parse_args()

    Path("artifacts").mkdir(exist_ok=True)
    raw = pd.read_csv(args.data_csv)

    # one row per problem (as before)
    df_cfg = raw.drop_duplicates(subset=["group_id"]).copy()
    df_cfg["split_group"] = _build_split_groups(df_cfg)

    # Drop ultra-rare labels (prevents warnings + improves F1)
    vc = df_cfg["best_choice"].value_counts()
    keep_labels = vc[vc >= args.min_class_count].index
    dropped = sorted(set(vc.index) - set(keep_labels))
    if len(dropped):
        print(f"[info] Dropping rare labels (<{args.min_class_count} samples): {dropped}")
    df_cfg = df_cfg[df_cfg["best_choice"].isin(keep_labels)].reset_index(drop=True)

    # Recompute training frame after drop
    X = df_cfg[FEATURE_COLUMNS]
    y = df_cfg["best_choice"]
    g = df_cfg["group_id"]
    split_groups = df_cfg["split_group"]

    # Base (regret) weights aligned by group_id
    base_weights = None
    if args.cost_weighting:
        w = compute_sample_weights_continuous(raw, alpha=args.alpha, cap=args.cap)
        df_cfg = df_cfg.merge(w, on="group_id", how="left")
        base_weights = df_cfg["cost_weight"].fillna(1.0)

    # Grouped + stratified split
    if _HAS_SGKF and _choose_stratify(y) is not None:
        sgg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        tr_idx, te_idx = next(sgg.split(X, y, groups=split_groups))
    else:
        # Prefer a stratified-only fallback to preserve label balance if groups make SGKF infeasible
        try:
            print("[warn] Using StratifiedKFold fallback (no stratified-groups available or classes too rare).")
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
            tr_idx, te_idx = next(skf.split(X, y))
        except Exception:
            print("[warn] Using GroupShuffleSplit fallback.")
            gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
            tr_idx, te_idx = next(gss.split(X, y, groups=split_groups))

    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
    gtr, gte = g.iloc[tr_idx], g.iloc[te_idx]

    # Combine regret weights (+ optional) with class-balance weights
    base_w_tr = None if base_weights is None else np.asarray(base_weights.iloc[tr_idx], dtype=float)
    wtr = _combine_weights(ytr, base_w_tr, args.class_balance)

    # Model selection
    picked_extra = ""
    if args.try_models:
        zoo = get_model_zoo(args.seed)
        best = None
        for name, mdl in zoo.items():
            mdl.fit(Xtr, ytr, sample_weight=wtr)
            yhat = mdl.predict(Xte)
            f1 = f1_score(yte, yhat, average="macro", zero_division=0)
            acc = accuracy_score(yte, yhat)
            reg = split_regret(raw, gte, pd.Series(yhat))
            print(f"[models] {name}: acc={acc:.3f} f1_macro={f1:.3f} med_regret={reg:.3f}")
            # Lexicographic keys to break ties robustly
            if args.select_by == "f1":
                key = (round(f1, 6), round(-reg, 6), round(acc, 6))
            elif args.select_by == "acc":
                key = (round(acc, 6), round(f1, 6), round(-reg, 6))
            else:  # select_by == "regret"
                key = (round(-reg, 6), round(f1, 6), round(acc, 6))

            if (best is None) or (key > best["key"]):
                best = {
                    "name": name,
                    "model": mdl,
                    "acc": acc,
                    "f1": f1,
                    "regret": reg,
                    "key": key
                }

        clf = best["model"]
        picked = f"{best['name']} (sel_by={args.select_by})"
        val_med_regret = float(best["regret"])
    else:
        clf = get_default_model(args.seed)
        clf.fit(Xtr, ytr, sample_weight=wtr)
        picked = f"{clf.__class__.__name__}(default)"
        yhat_tmp = clf.predict(Xte)
        val_med_regret = split_regret(raw, gte, pd.Series(yhat_tmp))

    # Optional probability calibration (only if we didn't pass weights)
    if args.calibrate and (wtr is None):
        try:
            clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
            clf.fit(Xtr, ytr)
            picked_extra = "+Calibrated"
        except Exception as e:
            print(f"[warn] calibration skipped: {e}")

    yhat_tr = clf.predict(Xtr)
    yhat_te = clf.predict(Xte)

    metrics = {
        "train_acc": accuracy_score(ytr, yhat_tr),
        "test_acc": accuracy_score(yte, yhat_te),
        "train_f1_macro": f1_score(ytr, yhat_tr, average="macro", zero_division=0),
        "test_f1_macro": f1_score(yte, yhat_te, average="macro", zero_division=0),
        "val_med_regret": float(val_med_regret),
        "select_by": args.select_by,
        "classes": sorted(y.unique()),
        "confusion_matrix": confusion_matrix(yte, yhat_te, labels=sorted(y.unique())).tolist(),
        "classification_report": classification_report(yte, yhat_te, output_dict=True, zero_division=0),
        "label_counts": y.value_counts().to_dict(),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "cost_weighting": bool(args.cost_weighting),
        "class_balance": args.class_balance,
        "model_picked": picked + picked_extra,
        "kfold": int(args.kfold),
        **kfold_eval(get_default_model(args.seed), X, y, groups=split_groups,
                     weights=base_weights, seed=args.seed, k=args.kfold,
                     balance_mode=args.class_balance),
    }

    table = tabulate([
        ["model", metrics["model_picked"]],
        ["select_by", metrics["select_by"]],
        ["train_acc", f"{metrics['train_acc']:.3f}"],
        ["test_acc",  f"{metrics['test_acc']:.3f}"],
        ["train_f1_macro", f"{metrics['train_f1_macro']:.3f}"],
        ["test_f1_macro",  f"{metrics['test_f1_macro']:.3f}"],
        ["val_med_regret", f"{metrics['val_med_regret']:.3f}"],
        ["cv_acc_mean", f"{(metrics['cv_acc_mean'] or float('nan')):.3f}"],
        ["cv_f1_macro_mean", f"{(metrics['cv_f1_macro_mean'] or float('nan')):.3f}"],
        ["n_train", metrics["n_train"]],
        ["n_test", metrics["n_test"]],
        ["cost_weighting", metrics["cost_weighting"]],
        ["class_balance", metrics["class_balance"]],
    ], headers=["metric","value"])
    print(table)

    joblib.dump({"model": clf, "feature_columns": FEATURE_COLUMNS}, args.model_out)
    print(f"Saved model to {args.model_out}")

    if hasattr(clf, "feature_importances_"):
        fi = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        fi.to_csv("artifacts/feature_importance.csv")
        print("Saved feature importances to artifacts/feature_importance.csv")

    with open(args.report_out, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(args.report_txt, "w") as f:
        f.write(table + "\n")
    print(f"Saved reports to {args.report_out} and {args.report_txt}")

if __name__ == "__main__":
    main()
