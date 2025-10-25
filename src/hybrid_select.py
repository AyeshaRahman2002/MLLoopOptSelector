# MLLoopOptSelector/src/hybrid_select.py
"""
Confidence-aware hybrid execution:
- If model confidence >= --min_conf, trust model OR verify only top-1
- Else run top-K candidates and pick fastest (runtime hybrid)
Modes: --hybrid auto|always|off
"""
import argparse, json, os, subprocess, tempfile
import pandas as pd, joblib
from features import build_feature_row, FEATURE_COLUMNS

CHOICE_TO_MACROS = {
    "baseline": [],
    "vec_pragma": ["-DPRAGMA_VEC"],
    "unroll": ["-DPRAGMA_UNROLL"],
    "tile": ["-DTILE_TRANSFORM"],
    "tile_unroll": ["-DTILE_TRANSFORM","-DPRAGMA_UNROLL"],
    "omp": ["-DOMP_PARALLEL"],
    "tile_omp": ["-DTILE_TRANSFORM","-DOMP_PARALLEL"],
    "tile_unroll_omp": ["-DTILE_TRANSFORM","-DPRAGMA_UNROLL","-DOMP_PARALLEL"],
}
KERNEL_TO_SRC = {"matmul":"csrc/matmul.c","conv1d":"csrc/conv1d.c","stencil2d":"csrc/stencil2d.c"}

def _clang_version():
    try: return subprocess.check_output(["clang","--version"], stderr=subprocess.STDOUT).decode()
    except Exception: return ""

def openmp_flags():
    if "OMP_FLAGS" in os.environ: return os.environ["OMP_FLAGS"].split()
    return ["-Xpreprocessor","-fopenmp","-lomp"] if "Apple" in _clang_version() else ["-fopenmp"]

def compile_kernel(src, out, macros):
    extra = openmp_flags() if any(m == "-DOMP_PARALLEL" for m in macros) else []
    cmd = ["clang","-O3",src,"-o",out] + macros + extra
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0: raise RuntimeError(p.stderr.decode())
    return out

def run_kernel(exe, args):
    p = subprocess.run([exe]+args, capture_output=True)
    if p.returncode != 0: raise RuntimeError(p.stderr.decode())
    data = json.loads(p.stdout.decode().strip())
    return float(data["time_sec"])

def infer_topk(bundle, kernel, N, M, K, ti, tj, tk, topk):
    cols = bundle["feature_columns"]; model = bundle["model"]
    row = build_feature_row(kernel, {"N":N,"M":M,"K":K}, "baseline", ti, tj, tk)
    X = pd.DataFrame([[row[c] for c in cols]], columns=cols)
    proba = model.predict_proba(X)[0]; classes = model.classes_
    ranking = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    return ranking[:topk]

def _sizes(kernel, N, M, K):
    if kernel=="matmul": return [str(N), str(M or N), str(K or N)]
    if kernel=="conv1d": return [str(N), str(K or max(3, N//16))]
    return [str(N), str(M or N)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_in", default="artifacts/model.joblib")
    ap.add_argument("--kernel", choices=["matmul","conv1d","stencil2d"], required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--M", type=int, default=0)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--tile_i", type=int, default=32)
    ap.add_argument("--tile_j", type=int, default=32)
    ap.add_argument("--tile_k", type=int, default=32)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--min_conf", type=float, default=0.80)
    ap.add_argument("--hybrid", choices=["auto","always","off"], default="auto")
    args = ap.parse_args()

    N, M, K = args.N, (args.M or args.N), (args.K or (args.N if args.kernel=="matmul" else max(3,args.N//16)))
    kargs = _sizes(args.kernel, N, M, K)

    bundle = joblib.load(args.model_in)
    ranking = infer_topk(bundle, args.kernel, N, M, K, args.tile_i, args.tile_j, args.tile_k, max(args.topk,1))
    # Heuristic guard: OMP is bad for very small conv outputs
    if args.kernel == "conv1d":
        O = max(1, args.N - (args.K or max(3, args.N//16)) + 1)
        if O < 256:
            bad = {"omp", "tile_omp", "tile_unroll_omp"}
            ranking = [(c,p) for (c,p) in ranking if c not in bad] + [(c,p) for (c,p) in ranking if c in bad]
    top1_choice, top1_p = ranking[0]

    result = {"policy": None, "selected": None, "ranking": ranking, "tried": [], "sizes":{"N":N,"M":M,"K":K}}
    with tempfile.TemporaryDirectory() as td:
        def compile_and_time(choice):
            macros = CHOICE_TO_MACROS[choice] + [f"-DTILE_I={args.tile_i}", f"-DTILE_J={args.tile_j}", f"-DTILE_K={args.tile_k}"]
            exe = os.path.join(td, f"{args.kernel}_{choice}")
            compile_kernel(KERNEL_TO_SRC[args.kernel], exe, macros)
            ts = [run_kernel(exe, kargs) for _ in range(args.repeats)]
            return min(ts)

        if args.hybrid=="off" or (args.hybrid=="auto" and top1_p>=args.min_conf):
            t = compile_and_time(top1_choice)
            result["policy"] = "trust_model" if args.hybrid!="off" else "model_only"
            result["tried"].append((top1_choice, float(top1_p), t))
            result["selected"] = {"choice": top1_choice, "time_sec": t, "model_prob": float(top1_p)}
        else:
            best = None
            for (choice, p) in ranking[:args.topk]:
                if choice not in CHOICE_TO_MACROS: continue
                t = compile_and_time(choice)
                result["tried"].append((choice, float(p), t))
                if (best is None) or (t < best["time_sec"]):
                    best = {"choice": choice, "time_sec": t, "model_prob": float(p)}
            result["policy"] = "hybrid_verify_topk"
            result["selected"] = best

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
