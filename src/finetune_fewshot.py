# MLLoopOptSelector/src/finetune_fewshot.py
import argparse, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from features import FEATURE_COLUMNS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="artifacts/model.joblib")
    ap.add_argument("--data_csv", default="artifacts/dataset.csv")
    ap.add_argument("--hold_kernel", choices=["matmul","conv1d","stencil2d"], required=True,
                    help="treat this as 'new kernel'; fine-tune with k examples")
    ap.add_argument("--k", type=int, default=16, help="few-shot examples")
    ap.add_argument("--out_model", default="artifacts/fewshot.joblib")
    args = ap.parse_args()

    base = joblib.load(args.base_model)
    model = base["model"]
    df = pd.read_csv(args.data_csv).drop_duplicates(subset=["group_id"])

    K = df[df["kernel"]==args.hold_kernel].sample(n=min(args.k, sum(df["kernel"]==args.hold_kernel)), random_state=0)
    Xk, yk = K[FEATURE_COLUMNS], K["best_choice"]

    if isinstance(model, RandomForestClassifier):
        tuned = RandomForestClassifier(n_estimators=model.n_estimators, random_state=0, n_jobs=-1, class_weight="balanced")
        tuned.fit(Xk, yk)
    else:
        tuned = model.__class__(**getattr(model, "get_params", lambda: {})())
        tuned.fit(Xk, yk)

    joblib.dump({"model": tuned, "feature_columns": FEATURE_COLUMNS, "parent": base}, args.out_model)
    print(f"Saved few-shot model to {args.out_model} (k={len(K)})")

if __name__ == "__main__":
    main()
