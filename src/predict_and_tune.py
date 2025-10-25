# MLLoopOptSelector/src/predict_and_tune.py
"""
Predict class: quick local tile search around (ti,tj,tk).
"""
import argparse, json, os, subprocess, tempfile
import joblib

def compile_and_run(src, macros, args):
    with tempfile.TemporaryDirectory() as td:
        exe = os.path.join(td, "a.out")
        cmd = ["clang","-O3",src,"-o",exe] + macros
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode!=0: raise RuntimeError(p.stderr.decode())
        r = subprocess.run([exe]+args, capture_output=True)
        if r.returncode!=0: raise RuntimeError(r.stderr.decode())
        import json as _j
        return float(_j.loads(r.stdout.decode().strip())["time_sec"])

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
    ap.add_argument("--radius", type=int, default=1, help="search +/- radius in {16,32,64} index space")
    args = ap.parse_args()

    sizes = {"matmul":[str(args.N),str(args.M or args.N),str(args.K or args.N)],
             "conv1d":[str(args.N),str(args.K or max(3,args.N//16))],
             "stencil2d":[str(args.N),str(args.M or args.N)]}[args.kernel]
    src = {"matmul":"csrc/matmul.c","conv1d":"csrc/conv1d.c","stencil2d":"csrc/stencil2d.c"}[args.kernel]
    bundle = joblib.load(args.model_in)
    model = bundle["model"]; classes = getattr(model,"classes_",[])
    top = classes[model.predict_proba([[0]*len(bundle["feature_columns"])])[0].argmax()] if len(classes) else "tile_unroll"

    grid = [16,32,64]; idx = [grid.index(args.tile_i), grid.index(args.tile_j), grid.index(args.tile_k)]
    cand = set()
    for di in range(-args.radius, args.radius+1):
        for dj in range(-args.radius, args.radius+1):
            for dk in range(-args.radius, args.radius+1):
                ni, nj, nk = [max(0, min(2, idx[d]+delta)) for d,delta in enumerate([di,dj,dk])]
                cand.add((grid[ni], grid[nj], grid[nk]))
    best = None
    for (ti,tj,tk) in sorted(cand):
        macros = [f"-DTILE_I={ti}", f"-DTILE_J={tj}", f"-DTILE_K={tk}", "-DTILE_TRANSFORM"]
        t = compile_and_run(src, macros, sizes)
        if (best is None) or (t < best["time_sec"]):
            best = {"tile_i":ti,"tile_j":tj,"tile_k":tk,"time_sec":t}
        print(f"try tiles=({ti},{tj},{tk}) -> {t:.6f}s")
    print(json.dumps({"predicted_class": top, "best": best}, indent=2))

if __name__ == "__main__":
    main()
