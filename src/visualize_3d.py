# MLLoopOptSelector/src/visualize_3d.py
import argparse, pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="artifacts/dataset.csv")
    ap.add_argument("--kernel", choices=["matmul","conv1d","stencil2d"], default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--choice", default=None)
    ap.add_argument("--out_png", default="artifacts/tile_runtime_3d.png")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    if args.kernel: df = df[df["kernel"]==args.kernel]
    if args.N is not None: df = df[df["N"]==args.N]
    if args.choice: df = df[df["choice"]==args.choice]

    df_t = df[df["choice"].str.contains("tile")]
    if len(df_t): df = df_t

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(df["tile_i"], df["tile_j"], df["tile_k"], c=df["time_sec"])
    ax.set_xlabel("tile_i"); ax.set_ylabel("tile_j"); ax.set_zlabel("tile_k")
    cb = fig.colorbar(p, ax=ax); cb.set_label("time_sec (lower is better)")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=180)
    print(f"Saved {args.out_png}")

if __name__ == "__main__":
    main()
