# MLLoopOptSelector/src/predict_gnn.py
import argparse, json, torch
from nn_models import TorchGCNClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_pt", default="artifacts/gnn_graphs.pt")
    ap.add_argument("--model_pt",   default="artifacts/gnn_model.pt")
    ap.add_argument("--group_id", type=int, required=True)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    graphs_bundle = torch.load(args.graphs_pt)
    graphs = graphs_bundle["graphs"]
    if args.group_id not in graphs: raise SystemExit("group_id not found in graphs.")
    meta = torch.load(args.model_pt)
    label_to_idx = meta["label_to_idx"]; idx_to_label = {v:k for k,v in label_to_idx.items()}

    # Rebuild model
    clf = TorchGCNClassifier(hidden=meta["hidden"], label_to_idx=label_to_idx)
    # Create structure-compatible model and load weights
    from nn_models import _GCNLayer  # ensure class is defined
    clf._build(meta["feat_dim"], len(label_to_idx))
    dev = clf._dev()
    clf.model_ = clf._build(meta["feat_dim"], len(label_to_idx)).to(dev)
    clf.model_.load_state_dict(meta["state_dict"])
    clf.classes_ = [idx_to_label[i] for i in range(len(label_to_idx))]
    clf._fitted = True

    probs = clf.predict_proba({args.group_id: graphs[args.group_id]})[0]
    pairs = list(enumerate(probs))
    pairs.sort(key=lambda t: t[1], reverse=True)
    out = [(idx_to_label[i], float(p)) for i,p in pairs[:args.topk]]
    print(json.dumps({"group_id": args.group_id, "topk": out}, indent=2))

if __name__ == "__main__":
    main()
