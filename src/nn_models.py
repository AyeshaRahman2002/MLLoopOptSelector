# MLLoopOptSelector/src/nn_models.py
import numpy as np

# Optional deps handled gracefully
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False
    torch = nn = TensorDataset = DataLoader = None


class _MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128, 64), pdrop=0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(pdrop)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier:
    """
    Scikit-like wrapper: fit / predict / predict_proba.
    Works with tabular FEATURE_COLUMNS. Uses balanced class weights by default.
    """
    def __init__(self, seed=42, epochs=40, lr=1e-3, batch_size=128,
                 hidden=(128, 64), pdrop=0.1, device="auto"):
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.hidden = tuple(hidden)
        self.pdrop = pdrop
        self.device = device
        self._fitted = False

    # sklearn compatibility
    def get_params(self, deep=True):
        return dict(seed=self.seed, epochs=self.epochs, lr=self.lr,
                    batch_size=self.batch_size, hidden=self.hidden,
                    pdrop=self.pdrop, device=self.device)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _dev(self):
        if self.device == "cpu":
            return "cpu"
        if self.device == "cuda":
            return "cuda" if _TORCH_OK and torch.cuda.is_available() else "cpu"
        return "cuda" if _TORCH_OK and torch.cuda.is_available() else "cpu"

    def fit(self, X, y, sample_weight=None):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch not installed. `pip install torch` first.")

        rng = np.random.RandomState(self.seed)
        torch.manual_seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        classes, y_idx = np.unique(y, return_inverse=True)
        self.classes_ = classes
        in_dim = X.shape[1]
        out_dim = len(classes)

        # class weights
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float32)
            cw = np.zeros(out_dim, dtype=np.float32)
            for c in range(out_dim):
                mask = (y_idx == c)
                cw[c] = sw[mask].mean() if mask.any() else 1.0
            class_weights = torch.tensor(cw, dtype=torch.float32)
        else:
            counts = np.bincount(y_idx)
            inv = 1.0 / np.maximum(counts, 1)
            inv = inv * (out_dim / inv.sum())
            class_weights = torch.tensor(inv, dtype=torch.float32)

        dev = self._dev()
        model = _MLPNet(in_dim, out_dim, hidden=self.hidden, pdrop=self.pdrop).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(dev))

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_idx).long())
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        self.model_ = model.eval()
        self._fitted = True
        return self

    def _predict_logits(self, X):
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            dev = self._dev()
            xb = torch.from_numpy(X).to(dev)
            logits = self.model_(xb).cpu().numpy()
        return logits

    def predict(self, X):
        logits = self._predict_logits(X)
        idx = logits.argmax(axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        logits = self._predict_logits(X)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


# Optional future: graph model placeholder
class GCNClassifier:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "GCNClassifier is a stub. Supply graph data + torch_geometric to implement."
        )


#vgnn
import math

class _GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        # x: [N, F], edge_index: [2, E] (u->v)
        N = x.size(0)
        # Build (sparse) degree-normalized aggregation with simple scatter
        row, col = edge_index  # row: src, col: dst
        # Add self-loops
        self_edges = torch.arange(N, device=x.device)
        row = torch.cat([row, self_edges], dim=0)
        col = torch.cat([col, self_edges], dim=0)
        deg = torch.zeros(N, device=x.device).scatter_add_(0, col, torch.ones_like(col, dtype=torch.float32))
        deg_inv_sqrt = (deg + 1e-9).pow(-0.5)
        # Message passing: sum_j (deg_i^-1/2 * deg_j^-1/2 * x_j)
        norm = deg_inv_sqrt[col] * deg_inv_sqrt[row]
        msg = torch.zeros_like(x)
        msg = msg.index_add(0, col, x[row] * norm.unsqueeze(-1))
        return self.lin(msg)

class TorchGCNClassifier:
    """
    Standalone GCN for graph inputs serialized by src/graphify.py.
    Not drop-in for the tabular FEATURE_COLUMNS pipeline; use train_gnn.py.
    """
    def __init__(self, hidden=64, epochs=60, lr=1e-3, weight_decay=5e-4,
                 seed=42, device="auto", label_to_idx=None, patience=20):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch not installed. `pip install torch` first.")
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.device = device
        self.label_to_idx = label_to_idx or {}
        self.idx_to_label = {v:k for k,v in (self.label_to_idx.items() or [])}
        self._fitted = False
        self.patience = int(patience)

    def _dev(self):
        if self.device == "cpu": return "cpu"
        if self.device == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _build(self, in_dim, out_dim):
        class Net(nn.Module):
            def __init__(self, in_dim, hidden, out_dim):
                super().__init__()
                self.g1 = _GCNLayer(in_dim, hidden)
                self.g2 = _GCNLayer(hidden, hidden)
                self.fc = nn.Linear(hidden, out_dim)
                self.act = nn.ReLU()
                self.drop = nn.Dropout(0.1)
            def forward(self, x, edge_index):
                h = self.g1(x, edge_index); h = self.act(h); h = self.drop(h)
                h = self.g2(h, edge_index); h = self.act(h); h = self.drop(h)
                # global mean pool
                g = h.mean(dim=0, keepdim=True)  # [1, hidden]
                return self.fc(g)                # [1, C]
        return Net(in_dim, self.hidden, out_dim)

    def fit(self, graphs, y, feat_dim, n_classes):
        torch.manual_seed(self.seed)
        dev = self._dev()

        # figure out which label indices actually occur in graphs
        y_all = [y[g] for g in graphs.keys()]
        if not len(y_all):
            raise RuntimeError("No graphs to train on.")

        # If label indices are contiguous 0..C-1 this is just max+1;
        # if not (e.g., some labels missing), we still cap to the largest seen.
        present_classes_upper_bound = int(max(y_all) + 1)
        n_classes_eff = min(n_classes, present_classes_upper_bound)

        # build model with the effective number of classes
        self.model_ = self._build(feat_dim, n_classes_eff).to(dev)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # class weights (balanced over *present* classes)
        counts = np.bincount(y_all, minlength=n_classes_eff)
        inv = 1.0 / np.maximum(counts, 1)
        inv = inv * (n_classes_eff / inv.sum())  # normalize to effective #classes
        class_weights = torch.tensor(inv, dtype=torch.float32, device=dev)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # training loop (full-batch, early stopping on loss)
        self.model_.train()
        gids = list(graphs.keys())
        ys = torch.tensor([y[g] for g in gids], dtype=torch.long, device=dev)

        best_loss = float("inf")
        best_state = None
        no_improve = 0

        for _ in range(self.epochs):
            logits = []
            for gid in gids:
                g = graphs[gid]
                x = g["x"].to(dev)
                ei = g["edge_index"].to(dev)
                logits.append(self.model_(x, ei))   # [1, C_eff]
            logits = torch.cat(logits, dim=0)        # [B, C_eff]
            loss = loss_fn(logits, ys)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            cur = loss.item()
            if cur + 1e-6 < best_loss:
                best_loss = cur
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        # restore best weights if we have them
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # map output indices back to original labels for predict()
        # if the graphify enumerates labels 0..C_eff-1 in order, this is identity
        self.classes_ = [self.idx_to_label[i] for i in range(n_classes_eff)]
        self._fitted = True
        return self

    def predict(self, graphs):
        if not self._fitted: raise RuntimeError("Call fit() first.")
        dev = self._dev()
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for gid in graphs.keys():
                g = graphs[gid]
                x = g["x"].to(dev); ei = g["edge_index"].to(dev)
                logit = self.model_(x, ei)  # [1,C]
                idx = logit.argmax(dim=1).item()
                preds.append(self.classes_[idx])
        return preds

    def predict_proba(self, graphs):
        if not self._fitted: raise RuntimeError("Call fit() first.")
        dev = self._dev()
        self.model_.eval()
        probs = []
        with torch.no_grad():
            for gid in graphs.keys():
                g = graphs[gid]
                x = g["x"].to(dev); ei = g["edge_index"].to(dev)
                logit = self.model_(x, ei)
                e = torch.exp(logit - logit.max())
                p = (e / e.sum(dim=1, keepdim=True)).squeeze(0).cpu().numpy()
                probs.append(p)
        return probs
