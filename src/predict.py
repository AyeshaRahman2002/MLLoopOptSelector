# MLLoopOptSelector/src/predict.py
"""
Loads the trained model and predicts the best optimization choice
for a given kernel and problem size. Prints top-choices with probabilities.
"""

import argparse, joblib, json, pandas as pd
from typing import Dict
from features import build_feature_row, FEATURE_COLUMNS

def _sizes(kernel: str, N: int, M: int, K: int) -> Dict[str, int]:
    s = {"N": N}
    if kernel == "matmul":
        s.update({"M": M or N, "K": K or N})
    elif kernel == "conv1d":
        s.update({"K": K or max(3, N//16)})
    else:  # stencil2d
        s.update({"M": M or N})
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_in", type=str, default="artifacts/model.joblib")
    ap.add_argument("--kernel", choices=["matmul","conv1d","stencil2d"], required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--M", type=int, default=0)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--tile_i", type=int, default=32)
    ap.add_argument("--tile_j", type=int, default=32)
    ap.add_argument("--tile_k", type=int, default=32)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    bundle = joblib.load(args.model_in)
    model = bundle["model"]
    cols = bundle["feature_columns"]

    row = build_feature_row(
        args.kernel,
        _sizes(args.kernel, args.N, args.M, args.K),
        choice="baseline",
        tile_i=args.tile_i, tile_j=args.tile_j, tile_k=args.tile_k
    )
    X = pd.DataFrame([[row[c] for c in cols]], columns=cols)

    proba = model.predict_proba(X)[0]
    classes = model.classes_
    ranking = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)

    result = {
        "topk": ranking[:args.topk],
        "predicted_choice": ranking[0][0],
        "confidence": float(ranking[0][1]),
        "features": row,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
