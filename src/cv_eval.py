# MLLoopOptSelector/src/cv_eval.py
import argparse, json, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from models import get_default_model
from features import FEATURE_COLUMNS

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGKF = True
except Exception:
    _HAS_SGKF = False
from sklearn.model_selection import StratifiedKFold

def _build_split_groups(cfg: pd.DataFrame) -> pd.Series:
    parts = []
    for _, r in cfg.iterrows():
        if r["kernel"] == "matmul":
            parts.append(f"matmul:{int(r['N'])}:{int(r['M'])}:{int(r['K'])}")
        elif r["kernel"] == "stencil2d":
            parts.append(f"stencil2d:{int(r['N'])}:{int(r['M'])}")
        else:
            parts.append(f"conv1d:{int(r['N'])}")
    return pd.Series(parts, index=cfg.index, name="split_group")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="artifacts/dataset.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default="artifacts/cv.json")
    ap.add_argument("--min_class_count", type=int, default=8)
    args = ap.parse_args()

    cfg = pd.read_csv(args.data_csv).drop_duplicates(subset=["group_id"]).copy()

    # drop ultra-rare labels to avoid degenerate folds
    vc = cfg["best_choice"].value_counts()
    keep = vc[vc >= args.min_class_count].index
    cfg = cfg[cfg["best_choice"].isin(keep)].reset_index(drop=True)

    X, y = cfg[FEATURE_COLUMNS], cfg["best_choice"]
    groups = _build_split_groups(cfg)

    # choose k feasible for the rarest class
    if len(y):
        min_cls = int(y.value_counts().min())
        k_eff = max(2, min(args.k, min_cls))
    else:
        k_eff = 2

    accs, f1s = [], []
    if _HAS_SGKF and k_eff >= 2:
        sgkf = StratifiedGroupKFold(n_splits=k_eff, shuffle=True, random_state=args.seed)
        for tr, te in sgkf.split(X, y, groups=groups):
            model = get_default_model(args.seed)
            model.fit(X.iloc[tr], y.iloc[tr])
            yhat = model.predict(X.iloc[te])
            accs.append(accuracy_score(y.iloc[te], yhat))
            f1s.append(f1_score(y.iloc[te], yhat, average="macro", zero_division=0))
    else:
        skf = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=args.seed)
        for tr, te in skf.split(X, y):
            model = get_default_model(args.seed)
            model.fit(X.iloc[tr], y.iloc[tr])
            yhat = model.predict(X.iloc[te])
            accs.append(accuracy_score(y.iloc[te], yhat))
            f1s.append(f1_score(y.iloc[te], yhat, average="macro", zero_division=0))

    out = {"k": k_eff, "acc_mean": sum(accs)/len(accs), "f1_macro_mean": sum(f1s)/len(f1s)}
    with open(args.out_json, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
