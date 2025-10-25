# MLLoopOptSelector/src/gnn_predict_all.py
import argparse, json
from pathlib import Path
import torch
import pandas as pd

def build_gnn(in_dim, hidden, out_dim, torch_nn):
    class _GCNLayer(torch_nn.Module):
        def __init__(self, in_d, out_d):
            super().__init__()
            self.lin = torch_nn.Linear(in_d, out_d)
        def forward(self, x, edge_index):
            N = x.size(0)
            row, col = edge_index
            self_edges = torch.arange(N, device=x.device)
            row = torch.cat([row, self_edges], 0)
            col = torch.cat([col, self_edges], 0)
            deg = torch.zeros(N, device=x.device).scatter_add_(0, col, torch.ones_like(col, dtype=torch.float32))
            deg_inv_sqrt = (deg + 1e-9).pow(-0.5)
            norm = deg_inv_sqrt[col] * deg_inv_sqrt[row]
            msg = torch.zeros_like(x)
            msg = msg.index_add(0, col, x[row] * norm.unsqueeze(-1))
            return self.lin(msg)
    class Net(torch_nn.Module):
        def __init__(self, in_d, hid, out_d):
            super().__init__()
            self.g1 = _GCNLayer(in_d, hid)
            self.g2 = _GCNLayer(hid, hid)
            self.fc = torch_nn.Linear(hid, out_d)
            self.act = torch_nn.ReLU()
            self.drop = torch_nn.Dropout(0.1)
        def forward(self, x, ei):
            h = self.g1(x, ei); h = self.act(h); h = self.drop(h)
            h = self.g2(h, ei); h = self.act(h); h = self.drop(h)
            g = h.mean(dim=0, keepdim=True)  # global mean pool
            return self.fc(g)                # [1, C]
    return Net(in_dim, hidden, out_dim)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_pt", default="artifacts/gnn_graphs.pt")
    ap.add_argument("--model_pt",  default="artifacts/gnn_model.pt")
    ap.add_argument("--out_csv",   default="artifacts/gnn_pred.csv")
    args = ap.parse_args()

    graphs_bundle = torch.load(args.graphs_pt)
    graphs = graphs_bundle["graphs"]
    label_to_idx = graphs_bundle.get("label_to_idx") or torch.load(args.model_pt)["label_to_idx"]
    idx_to_label = {v:k for k,v in label_to_idx.items()}

    meta = torch.load(args.model_pt)
    feat_dim = int(meta["feat_dim"])
    hidden   = int(meta["hidden"])
    n_classes= len(meta["label_to_idx"])

    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_gnn(feat_dim, hidden, n_classes, nn).to(dev)
    net.load_state_dict(meta["state_dict"])
    net.eval()

    rows = []
    with torch.no_grad():
        for gid, g in graphs.items():
            x = g["x"].to(dev); ei = g["edge_index"].to(dev)
            logits = net(x, ei)  # [1, C]
            e = torch.exp(logits - logits.max())
            p = (e / e.sum(dim=1, keepdim=True)).squeeze(0).cpu().numpy()
            # top-1
            top = p.argmax()
            rows.append({
                "group_id": int(gid),
                "pred_choice": idx_to_label[top],
                "conf": float(p[top])
            })
    df = pd.DataFrame(rows).sort_values("group_id")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} (rows={len(df)})")

if __name__ == "__main__":
    main()
