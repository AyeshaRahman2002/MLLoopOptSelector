# MLLoopOptSelector/src/graphify.py
import json, math, argparse
from pathlib import Path
import pandas as pd
import torch

# Minimal scalar featurizers (safe logs & normalizers)
def _safelog(x): 
    x = float(x or 0.0)
    return math.log(max(1.0, x))

def _norm(v, cap=2048):  # coarse normalization for sizes/tiles
    v = float(v or 0.0); return min(v, cap)/cap

# Node schema:
#  [kernel_onehot(3), size_feats(4), tile_feats(3), ai, conv_ratio, mis(3), ratio(3)]
# don’t fill everything for every node; unused slots are zeros.
FEAT_DIM = 3 + 4 + 3 + 1 + 1 + 3 + 3

KERNELS = ["matmul","conv1d","stencil2d"]
LABELS  = ["baseline","vec_pragma","unroll","tile","tile_unroll","omp","tile_omp","tile_unroll_omp"]

def node_vec(kernel=None, N=None, M=None, K=None, O=None, tile_i=None, tile_j=None, tile_k=None,
             ai=None, conv_ratio=None, mis_i=None, mis_j=None, mis_k=None,
             ratio_i=None, ratio_j=None, ratio_k=None):
    v = [0.0]*FEAT_DIM
    # kernel onehot
    if kernel in KERNELS:
        v[KERNELS.index(kernel)] = 1.0
    # sizes/logs: [logN, logM, logK, logO]
    offs = 3
    if N is not None: v[offs+0] = _safelog(N)
    if M is not None: v[offs+1] = _safelog(M)
    if K is not None: v[offs+2] = _safelog(K)
    if O is not None: v[offs+3] = _safelog(O)
    # tiles: [norm ti,tj,tk]
    offs = 3+4
    if tile_i is not None: v[offs+0] = _norm(tile_i, 512)
    if tile_j is not None: v[offs+1] = _norm(tile_j, 512)
    if tile_k is not None: v[offs+2] = _norm(tile_k, 512)
    # ai
    offs = 3+4+3
    if ai is not None: v[offs+0] = float(ai)
    # conv ratio
    offs += 1
    if conv_ratio is not None: v[offs+0] = float(conv_ratio)
    # misalign: [mis_i,mis_j,mis_k]
    offs += 1
    if mis_i is not None: v[offs+0] = float(mis_i)
    if mis_j is not None: v[offs+1] = float(mis_j)
    if mis_k is not None: v[offs+2] = float(mis_k)
    # ratios: [ratio_i,ratio_j,ratio_k]
    offs += 3
    if ratio_i is not None: v[offs+0] = float(ratio_i)
    if ratio_j is not None: v[offs+1] = float(ratio_j)
    if ratio_k is not None: v[offs+2] = float(ratio_k)
    return torch.tensor(v, dtype=torch.float32)

def make_graph(row):
    # Nodes:
    # 0: "kernel" hub
    # 1-4: size nodes (N,M,K,O)
    # 5-7: tile nodes (ti,tj,tk)
    # 8:   "stats" node (ai/ratios/mis/conv_ratio)
    kernel = row["kernel"]
    N, M, K, O = int(row["N"]), int(row.get("M",0) or 0), int(row.get("K",0) or 0), int(row.get("O", row["N"]))
    ti, tj, tk = int(row["tile_i"]), int(row["tile_j"]), int(row["tile_k"])
    ai = float(row["ai"])
    mis_i, mis_j, mis_k = float(row["mis_i"]), float(row["mis_j"]), float(row["mis_k"])
    ratio_i, ratio_j, ratio_k = float(row["ratio_i"]), float(row["ratio_j"]), float(row["ratio_k"])
    conv_ratio = float(row.get("conv_ratio_K_over_N", 0.0))

    X = torch.stack([
        node_vec(kernel=kernel),                      # 0
        node_vec(kernel=kernel, N=N),                 # 1
        node_vec(kernel=kernel, M=M if kernel!="conv1d" else 0),  # 2
        node_vec(kernel=kernel, K=K),                 # 3
        node_vec(kernel=kernel, O=O),                 # 4
        node_vec(kernel=kernel, tile_i=ti),           # 5
        node_vec(kernel=kernel, tile_j=tj),           # 6
        node_vec(kernel=kernel, tile_k=tk),           # 7
        node_vec(kernel=kernel, ai=ai, conv_ratio=conv_ratio,
                 mis_i=mis_i, mis_j=mis_j, mis_k=mis_k,
                 ratio_i=ratio_i, ratio_j=ratio_j, ratio_k=ratio_k),  # 8
    ])

    # Undirected edges: connect hub↔sizes, sizes↔tiles, hub↔stats
    edges = []
    # hub <-> sizes
    for nid in [1,2,3,4]:
        edges += [(0,nid),(nid,0)]
    # sizes <-> tiles (dense bipartite)
    for s in [1,2,3,4]:
        for t in [5,6,7]:
            edges += [(s,t),(t,s)]
    # hub <-> stats
    edges += [(0,8),(8,0)]

    E = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape [2, num_edges]
    return X, E

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="artifacts/dataset.csv")
    ap.add_argument("--out_pt", default="artifacts/gnn_graphs.pt")
    ap.add_argument("--out_meta", default="artifacts/gnn_meta.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).drop_duplicates(subset=["group_id"]).copy()
    # Keep the same label target as before
    labels = sorted(df["best_choice"].dropna().unique().tolist())
    # Guarantee deterministic label ordering across runs
    label_to_idx = {lbl:i for i,lbl in enumerate(labels)}  # contiguous 0..C-1
    # Build per-group graph
    graphs = {}
    y_map = {}
    for _, r in df.iterrows():
        gid = int(r["group_id"])
        X, E = make_graph(r)
        lbl = int(label_to_idx[r["best_choice"]])
        graphs[gid] = {"x": X, "edge_index": E, "y": lbl}
        y_map[gid] = lbl

    Path(args.out_pt).parent.mkdir(exist_ok=True, parents=True)
    torch.save({"graphs": graphs, "label_to_idx": label_to_idx, "feat_dim": FEAT_DIM}, args.out_pt)
    Path(args.out_meta).write_text(json.dumps({
        "n_graphs": len(graphs),
        "labels": list(label_to_idx.keys()),
        "feat_dim": FEAT_DIM
    }, indent=2))
    print(f"Saved {args.out_pt} with {len(graphs)} graphs.")

if __name__ == "__main__":
    main()
