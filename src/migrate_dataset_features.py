# MLLoopOptSelector/src/migrate_dataset_features.py
import argparse
import math
from pathlib import Path

import pandas as pd

def _safe_log(x):
    return math.log(max(1.0, float(x or 0)))

def _ratio(n, t):
    return float(n) / float(max(1, t))

def _mis(n, t):
    return (int(n) % max(1, int(t))) / float(max(1, int(t)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", default="artifacts/dataset.csv")
    ap.add_argument("--csv_out", default="artifacts/dataset.csv")
    args = ap.parse_args()

    p = Path(args.csv_in)
    df = pd.read_csv(p)

    required = ["kernel", "N", "M", "K", "tile_i", "tile_j", "tile_k"]
    missing_base = [c for c in required if c not in df.columns]
    if missing_base:
        raise SystemExit(f"Missing base columns needed for migration: {missing_base}")

    # One-time backup
    bak = p.with_suffix(".csv.bak")
    if not bak.exists():
        bak.write_text(p.read_text())
        print(f"Backed up original to {bak}")

    # Ensure all engineered columns exist
    engineered_cols = [
        "logN", "logM", "logK",
        "ratio_i", "ratio_j", "ratio_k",
        "mis_i", "mis_j", "mis_k",
        "conv_ratio_K_over_N",
        "O", "logO", "ratio_O_over_N",
    ]
    for c in engineered_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Logs
    df["logN"] = df["N"].apply(_safe_log)
    df["logM"] = df["M"].fillna(0).apply(_safe_log)
    df["logK"] = df["K"].fillna(0).apply(_safe_log)

    # Effective sizes for ratio/misalignment when M/K may be 0
    M_eff = df.apply(lambda r: r["M"] if pd.notna(r["M"]) and r["M"] > 0 else r["N"], axis=1)
    K_eff = df.apply(lambda r: r["K"] if pd.notna(r["K"]) and r["K"] > 0 else r["N"], axis=1)

    # Ratios wrt tiles
    df["ratio_i"] = df.apply(lambda r: _ratio(r["N"], r["tile_i"]), axis=1)
    df["ratio_j"] = df.apply(lambda r: _ratio(M_eff.loc[r.name], r["tile_j"]), axis=1)
    df["ratio_k"] = df.apply(lambda r: _ratio(K_eff.loc[r.name], r["tile_k"]), axis=1)

    # Misalignment features
    df["mis_i"] = df.apply(lambda r: _mis(r["N"], r["tile_i"]), axis=1)
    df["mis_j"] = df.apply(lambda r: _mis(M_eff.loc[r.name], r["tile_j"]), axis=1)
    df["mis_k"] = df.apply(lambda r: _mis(K_eff.loc[r.name], r["tile_k"]), axis=1)

    # Kernel-specific ratios
    df["conv_ratio_K_over_N"] = df.apply(
        lambda r: (r["K"] / r["N"]) if (r.get("kernel") == "conv1d" and r["N"]) else 0.0,
        axis=1,
    )

    # output length for conv1d (else default to N)
    def _O(row):
        if row.get("kernel") == "conv1d":
            return max(1, int(row["N"]) - int(row["K"]) + 1)
        # For stencil2d/matmul, a simple surrogate that keeps the scale comparable
        return int(row["N"])

    df["O"] = df.apply(_O, axis=1)
    df["logO"] = df["O"].apply(_safe_log)
    df["ratio_O_over_N"] = df.apply(lambda r: (r["O"] / r["N"]) if r["N"] else 0.0, axis=1)

    df.to_csv(args.csv_out, index=False)
    print(f"Wrote migrated CSV with new columns to {args.csv_out}")


if __name__ == "__main__":
    main()
