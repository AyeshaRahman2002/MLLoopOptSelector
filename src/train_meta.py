# MLLoopOptSelector/src/train_meta.py
import argparse, json, joblib, pandas as pd
from pathlib import Path
from models import get_default_model
from features import FEATURE_COLUMNS
from utils import sysinfo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="artifacts/dataset.csv")
    ap.add_argument("--out_dir", default="artifacts/models_meta")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data_csv).drop_duplicates(subset=["group_id"])
    meta = sysinfo()
    arch = f"{meta.get('machine','unknown')}-{meta.get('processor','cpu')}".replace(" ", "_")
    model = get_default_model()
    X, y = raw[FEATURE_COLUMNS], raw["best_choice"]
    model.fit(X, y)
    out = Path(args.out_dir)/f"{arch}.joblib"
    joblib.dump({"model":model,"feature_columns":FEATURE_COLUMNS,"arch":arch}, out)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
