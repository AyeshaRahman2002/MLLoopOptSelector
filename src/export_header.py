# MLLoopOptSelector/src/export_header.py
"""
Export a simple C header with a size-bucketed schedule map.
"""

import argparse, pandas as pd
from pathlib import Path

def bucket(n, step=64, cap=1024):
    n = min(n, cap)
    return int((n + step - 1)//step)*step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="artifacts/dataset.csv")
    ap.add_argument("--out_h", default="artifacts/generated_schedule.h")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv).drop_duplicates(subset=["group_id"])
    df["N_bucket"] = df["N"].apply(bucket)

    # pick most frequent best_choice per (kernel, N_bucket)
    pick = df.groupby(["kernel","N_bucket"])["best_choice"].agg(lambda s: s.value_counts().idxmax()).reset_index()

    lines = [
        "#pragma once",
        "#ifdef __cplusplus",
        "#  include <cstring>",
        "#else",
        "#  include <string.h>",
        "#endif",
        "static inline const char* schedule_for(const char* kernel, int N){",
        "  int B = (N+63)/64*64; if(B>1024) B=1024;",
    ]
    for _,r in pick.iterrows():
        lines.append(f'  if (!strcmp(kernel,"{r.kernel}") && B=={int(r.N_bucket)}) return "{r.best_choice}";')
    lines.append('  return "baseline";')
    lines.append("}")
    Path(args.out_h).write_text("\n".join(lines))
    print(f"Wrote {args.out_h}")

if __name__ == "__main__":
    main()
