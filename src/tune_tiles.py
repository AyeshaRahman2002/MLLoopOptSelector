# MLLoopOptSelector/src/tune_tiles.py
"""
Model-guided tile-size tuner.
Given (kernel, N/M/K), evaluate a small set of tile_i/j/k and return the best time.
"""

import argparse, json, os, random, subprocess, tempfile
from itertools import product

def compile_and_run(src, macros, args):
    exe = None
    with tempfile.TemporaryDirectory() as td:
        exe = os.path.join(td, "a.out")
        cmd = ["clang", "-O3", src, "-o", exe] + macros
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode())
        r = subprocess.run([exe] + args, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode())
        import json as _j
        return float(_j.loads(r.stdout.decode().strip())["time_sec"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=["matmul","conv1d","stencil2d"], required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--M", type=int, default=0)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--candidates", nargs="*", type=int, default=[16,32,64])
    ap.add_argument("--random", action="store_true", help="sample 10 random triples instead of full grid")
    args = ap.parse_args()

    N, M, K = args.N, (args.M or args.N), (args.K or (args.N if args.kernel=="matmul" else max(3,args.N//16)))
    kargs = [str(N)] + ([str(M)] if args.kernel!="conv1d" else []) + ([str(K)] if args.kernel!="stencil2d" else [])
    src = {"matmul":"csrc/matmul.c","conv1d":"csrc/conv1d.c","stencil2d":"csrc/stencil2d.c"}[args.kernel]

    space = list(product(args.candidates, args.candidates, args.candidates))
    if args.random and len(space) > 10:
        random.seed(0)
        space = random.sample(space, 10)

    best = None
    for (ti,tj,tk) in space:
        macros = [f"-DTILE_I={ti}", f"-DTILE_J={tj}", f"-DTILE_K={tk}", "-DTILE_TRANSFORM","-DPRAGMA_UNROLL"]
        t = compile_and_run(src, macros, kargs)
        if (best is None) or (t < best["time_sec"]):
            best = {"tile_i":ti,"tile_j":tj,"tile_k":tk,"time_sec":t}
        print(f"try tiles=({ti},{tj},{tk}) -> {t:.6f}s")
    print(json.dumps({"best": best}, indent=2))

if __name__ == "__main__":
    main()
