# MLLoopOptSelector/src/explain_choice.py
import argparse, json, joblib, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from features import FEATURE_COLUMNS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="artifacts/dataset.csv")
    ap.add_argument("--model_in", default="artifacts/model.joblib")
    ap.add_argument("--out_png", default="artifacts/explain.png")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv).drop_duplicates(subset=["group_id"])
    X, y = df[FEATURE_COLUMNS], df["best_choice"]
    bundle = joblib.load(args.model_in); model = bundle["model"]

    shap_ok = False
    try:
        import shap  # optional
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X, check_additivity=False)
        shap_ok = True
        vals = np.sum([np.abs(v).mean(axis=0) for v in sv], axis=0) if isinstance(sv, list) else np.abs(sv).mean(axis=0)
        imp = pd.Series(vals, index=FEATURE_COLUMNS).sort_values(ascending=False)
        title = "SHAP feature importance"
    except Exception:
        pi = permutation_importance(model, X, y, n_repeats=10, random_state=0, n_jobs=-1)
        imp = pd.Series(pi.importances_mean, index=FEATURE_COLUMNS).sort_values(ascending=False)
        title = "Permutation feature importance"

    fig = plt.figure()
    ax = fig.gca()
    imp.head(12).plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("importance")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=180)
    print(f"Saved {args.out_png} (SHAP={shap_ok})")

if __name__ == "__main__":
    main()
