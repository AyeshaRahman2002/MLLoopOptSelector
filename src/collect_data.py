# MLLoopOptSelector/src/collect_data.py
"""
Parallel collector with noise controls and basic system profiling.
Adds --parallel/--jobs, --warmup, --trim_outliers, --more_kernels
"""
import argparse, json, subprocess, tempfile, os, shlex, sys, statistics, multiprocessing as mp
import pandas as pd
from pathlib import Path
from features import build_feature_row
from utils import sysinfo, dump_json

def _clang_version():
    try:
        out = subprocess.check_output(["clang", "--version"], stderr=subprocess.STDOUT).decode()
        return out
    except Exception:
        return ""

def openmp_flags():
    if "OMP_FLAGS" in os.environ: return os.environ["OMP_FLAGS"].split()
    v = _clang_version()
    return ["-Xpreprocessor", "-fopenmp", "-lomp"] if "Apple" in v else ["-fopenmp"]

CHOICES = [
    ("baseline", []),
    ("vec_pragma", ["-DPRAGMA_VEC"]),
    ("unroll", ["-DPRAGMA_UNROLL"]),
    ("tile", ["-DTILE_TRANSFORM"]),
    ("tile_unroll", ["-DTILE_TRANSFORM","-DPRAGMA_UNROLL"]),
    ("omp", ["-DOMP_PARALLEL"]),
    ("tile_omp", ["-DTILE_TRANSFORM","-DOMP_PARALLEL"]),
    ("tile_unroll_omp", ["-DTILE_TRANSFORM","-DPRAGMA_UNROLL","-DOMP_PARALLEL"]),
]

MORE_KERNELS = {
    "saxpy": "csrc/saxpy.c",
}

def run(cmd): return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def compile_kernel(src_c: str, out_path: str, macros):
    extra = openmp_flags() if any(m == "-DOMP_PARALLEL" for m in macros) else []
    cmd = ["clang", "-O3", src_c, "-o", out_path] + macros + extra
    p = run(cmd)
    if p.returncode != 0:
        print("Compile failed:", " ".join(shlex.quote(a) for a in cmd), file=sys.stderr)
        print(p.stderr.decode(), file=sys.stderr)
        raise SystemExit(1)
    return out_path

def run_kernel(exe: str, args):
    p = run([exe] + args)
    if p.returncode != 0:
        print("Run failed:", exe, args, file=sys.stderr)
        print(p.stderr.decode(), file=sys.stderr)
        raise SystemExit(1)
    data = json.loads(p.stdout.decode().strip())
    return float(data["time_sec"]), float(data.get("checksum", 0.0))

def robust_time(exe, args, repeats: int, warmup: int = 1, trim_outliers=False):
    for _ in range(warmup): run_kernel(exe, args)
    times = [run_kernel(exe, args)[0] for _ in range(repeats)]
    if trim_outliers and len(times) >= 5:
        qts = statistics.quantiles(times, n=4)
        q1, q3 = qts[0], qts[2]
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        times = [t for t in times if lo <= t <= hi] or times
    return statistics.median(times)

def task(args):
    (kernel, cpath, N, Ksize, choice, macros, ti, tj, tk, repeats, warmup, trim) = args
    with tempfile.TemporaryDirectory() as tmpdir:
        exe = os.path.join(tmpdir, f"{kernel}_{choice}_{N}_{ti}_{tj}_{tk}")
        mdefs = macros + [f"-DTILE_I={ti}", f"-DTILE_J={tj}", f"-DTILE_K={tk}"]
        compile_kernel(cpath, exe, mdefs)
        if kernel == "matmul":
            kargs = [str(N), str(N), str(N)]
        elif kernel == "conv1d":
            kargs = [str(N), str(Ksize)]
        elif kernel == "stencil2d":
            kargs = [str(N), str(N)]
        elif kernel == "saxpy":
            kargs = [str(N)]
        else:
            raise RuntimeError("unknown kernel")
        t = robust_time(exe, kargs, repeats, warmup, trim)
        sizes = {"N": N}
        if kernel == "matmul": sizes.update({"M": N, "K": N})
        elif kernel == "conv1d": sizes.update({"K": Ksize})
        elif kernel == "stencil2d": sizes.update({"M": N})
        row = build_feature_row(kernel, sizes, choice, ti, tj, tk)
        row.update({
            "time_sec": t, "checksum": 0.0, "compile_macros": " ".join(mdefs), "kernel": kernel
        })
        return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[128,256,384,512])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count()//2))
    ap.add_argument("--trim_outliers", action="store_true")
    ap.add_argument("--more_kernels", action="store_true", help="include extra kernels (e.g., saxpy)")
    ap.add_argument("--out_csv", type=str, default="artifacts/dataset.csv")
    ap.add_argument("--meta_out", type=str, default="artifacts/collect_meta.json")
    args = ap.parse_args()

    Path("artifacts").mkdir(exist_ok=True)
    dump_json(args.meta_out, {"system": sysinfo(), "args": vars(args)})

    kernels = {
        "matmul": "csrc/matmul.c",
        "conv1d": "csrc/conv1d.c",
        "stencil2d": "csrc/stencil2d.c",
    }
    if args.more_kernels:
        kernels.update(MORE_KERNELS)

    jobs = []
    tile_is = [16, 32, 64]
    tile_js = [16, 32, 64]
    tile_ks = [16, 32, 64]

    for kernel, cpath in kernels.items():
        for N in args.sizes:
            if kernel == "conv1d":
                k_candidates = sorted({max(3, N//32), max(3, N//16), max(3, N//8), max(3, N//4)})
            else:
                k_candidates = [max(3, N//16)]

            for Ksize in k_candidates:
                for choice, macros in CHOICES:
                    for ti in tile_is:
                        for tj in tile_js:
                            for tk in tile_ks:
                                jobs.append((kernel, cpath, N, Ksize, choice, macros, ti, tj, tk,
                                             args.repeats, args.warmup, args.trim_outliers))

    rows = []
    if args.parallel and args.jobs > 1:
        with mp.Pool(processes=args.jobs) as pool:
            for i, row in enumerate(pool.imap_unordered(task, jobs), 1):
                rows.append(row)
                if (i % 50)==0: print(f"[collect] {i}/{len(jobs)} done")
    else:
        for i, j in enumerate(jobs, 1):
            rows.append(task(j))
            if (i % 50)==0: print(f"[collect] {i}/{len(jobs)} done")

    df = pd.DataFrame(rows)
    group_cols = ["kernel","N","M","K","tile_i","tile_j","tile_k"]
    df["group_id"] = df.groupby(group_cols).ngroup()
    best = df.loc[df.groupby("group_id")["time_sec"].idxmin(), ["group_id","choice"]].rename(columns={"choice":"best_choice"})
    df = df.merge(best, on="group_id", how="left")
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved dataset to {args.out_csv} (rows={len(df)})")

if __name__ == "__main__":
    main()
