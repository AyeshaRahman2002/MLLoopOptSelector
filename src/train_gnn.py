# MLLoopOptSelector/src/train_gnn.py
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from nn_models import TorchGCNClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_pt", default="artifacts/gnn_graphs.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_model", default="artifacts/gnn_model.pt")
    ap.add_argument("--out_report", default="artifacts/gnn_report.json")
    args = ap.parse_args()

    bundle = torch.load(args.graphs_pt)
    graphs = bundle["graphs"]
    label_to_idx = bundle["label_to_idx"]
    feat_dim = int(bundle["feat_dim"])
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Labels are embedded per-graph by graphify.py
    y = {int(gid): int(graphs[gid].get("y", -1)) for gid in graphs.keys()}
    if any(v < 0 for v in y.values()):
        raise SystemExit("Missing labels in graphs .pt (rerun graphify.py).")

    # Deterministic split that adapts to the rarest class
    gids = sorted(graphs.keys())

    # Build label array aligned to gids
    labels_list = np.array([y[g] for g in gids])

    # prune singleton classes from the pool
    # so that stratified splitting is well-defined.
    counts = np.bincount(labels_list) if len(labels_list) else np.array([])
    rare_lbls = set(np.where(counts < 2)[0]) if counts.size else set()
    if rare_lbls:
        keep_mask = np.array([y[g] not in rare_lbls for g in gids])
        gids = list(np.array(gids)[keep_mask])
        labels_list = labels_list[keep_mask]

    # If after pruning we have too few samples or classes, fall back gracefully
    n_samples = len(gids)
    n_classes = len(np.unique(labels_list)) if n_samples else 0

    if n_samples < 2:
        raise SystemExit("Not enough graphs to split after pruning singleton classes.")

    # Decide split strategy
    counts = np.bincount(labels_list)
    min_count = int(counts.min()) if counts.size else 0

    if (n_classes >= 2) and (min_count >= 2):
        # stratified K-fold (take first fold) with n_splits <= min_count
        n_splits = max(2, min(4, min_count))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        tr_idx, te_idx = next(skf.split(np.zeros(n_samples), labels_list))
    else:
        # simple 75/25 split; only stratify if it is valid
        from sklearn.model_selection import train_test_split
        do_strat = (n_classes >= 2) and (min_count >= 2)
        strat = labels_list if do_strat else None
        tr_idx, te_idx = train_test_split(
            np.arange(n_samples),
            test_size=0.25,
            random_state=args.seed,
            stratify=strat
        )

    tr_ids = [gids[i] for i in tr_idx]
    te_ids = [gids[i] for i in te_idx]

    train_graphs = {g: graphs[g] for g in tr_ids}
    test_graphs  = {g: graphs[g] for g in te_ids}

    clf = TorchGCNClassifier(
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        label_to_idx=label_to_idx,
        patience=args.patience
    )

    # Train
    clf.fit(train_graphs, y, feat_dim=feat_dim, n_classes=len(label_to_idx))

    # Evaluate
    yhat = clf.predict(test_graphs)
    ytrue = [idx_to_label[y[g]] for g in test_graphs.keys()]
    acc = accuracy_score(ytrue, yhat)
    f1m = f1_score(ytrue, yhat, average="macro", zero_division=0)

    # Save
    Path(args.out_model).parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        "state_dict": clf.model_.state_dict(),
        "label_to_idx": label_to_idx,
        "feat_dim": feat_dim,
        "hidden": clf.hidden,
    }, args.out_model)

    rep = {
        "acc": float(acc),
        "f1_macro": float(f1m),
        "n_train": len(train_graphs),
        "n_test": len(test_graphs)
    }
    Path(args.out_report).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
