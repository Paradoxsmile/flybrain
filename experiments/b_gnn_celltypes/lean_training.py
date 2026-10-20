"""Memory-lean GNN training for label spaces with thousands of classes (size_threshold.py at
min type size 3 or 2).

flybrain.training.fit_gnn evaluates the last DirectedSAGE layer on every node, i.e. three N x K
tensors (self, input and output terms) plus their sum, softmax and gradients: about 4.8 GB each
for FAFB at K = 8,658 classes. With K > hidden_dim the last layer aggregates before projecting,
so its logits are z @ W.T + b with z = [h, mean_in(h), mean_out(h)] (N x 3 hidden) and W the three
weight matrices side by side. Here the logits are computed only for the rows needed (training
nodes for the loss, validation nodes for model selection), in chunks, with activation
checkpointing, so memory is O(chunk x K). Same parameters, same dropout draws and same
model selection as fit_gnn (no edge-drop / thinning augmentation, which the cell_type config
does not use); model.pt files are interchangeable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.checkpoint import checkpoint

from flybrain.features import normalize_features
from flybrain.models import DirectedSAGE, MeanBuilder, class_weights
from flybrain.training import Fitted

CHUNK = 8192


def head(model: DirectedSAGE) -> tuple[torch.Tensor, torch.Tensor]:
    """(W, b) of the last layer: logits = [h, mean_in(h), mean_out(h)] @ W.T + b."""
    last = model.layers[-1]
    assert not last.project_first, "lean head needs out_dim >= in_dim"
    ws = [last.lin_self.weight, last.lin_in.weight]
    ws += [last.lin_out.weight] if last.lin_out is not None else []
    return torch.cat(ws, dim=1), last.lin_in.bias


def features(model: DirectedSAGE, x, mean_in, mean_out) -> torch.Tensor:
    """z = [h, mean_in(h), mean_out(h)] of the last layer's input h."""
    for layer in model.layers[:-1]:
        x = F.dropout(F.relu(layer(x, mean_in, mean_out)), model.dropout, model.training)
    parts = [x, mean_in(x)] + ([mean_out(x)] if model.bidirectional else [])
    return torch.cat(parts, dim=1)


def _loss_sum(z, y, w, b, weight):
    return F.cross_entropy(z @ w.T + b, y, weight=weight, reduction="sum")


@torch.no_grad()
def predict_rows(model, x, adj, rows=None, chunk: int = CHUNK) -> torch.Tensor:
    """Predicted class of the given rows (default: all nodes)."""
    model.eval()
    z = features(model, x, *adj)
    w, b = head(model)
    rows = torch.arange(len(z), device=z.device) if rows is None else rows
    out = [(z[r] @ w.T + b).argmax(1) for r in rows.split(chunk)]
    return torch.cat(out) if out else rows.new_empty(0)


def predict_lean(model, x, adj) -> np.ndarray:
    return predict_rows(model, x, adj).cpu().numpy()


def fit_gnn_lean(cfg: dict, data, train, val, device, chunk: int = CHUNK) -> Fitted:
    """fit_gnn with the chunked, checkpointed output layer (see the module docstring)."""
    if cfg.get("drop_edge", 0.0) > 0 or cfg.get("thin_aug", 0.0) > 0:
        raise ValueError("fit_gnn_lean does not implement edge-drop / thinning augmentation")
    norm, side = cfg.get("feature_norm", "global"), getattr(data, "side", None)
    x = normalize_features(data.x, norm, train, side).to(device)
    y = data.y.to(device)
    bidirectional = cfg.get("bidirectional", True)
    ei, n, w = data.edge_index.to(device), data.num_nodes, data.edge_weight.to(device)
    adj = MeanBuilder(ei, n, w, cfg.get("edge_weight", "count"))(bidirectional)
    num_classes = len(data.label_names)
    tr = train.to(device).nonzero().squeeze(1)
    va = val.to(device).nonzero().squeeze(1)
    weight = class_weights(y[tr], num_classes, cfg.get("class_weight", "none"))
    norm_const = weight[y[tr]].sum() if weight is not None else torch.tensor(len(tr), device=device)
    model = DirectedSAGE(
        x.size(1), cfg["hidden_dim"], num_classes, cfg["num_layers"], cfg["dropout"], bidirectional
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    y_val = y[va].cpu().numpy()

    history, best_val, best_state = [], -1.0, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        opt.zero_grad()
        z = features(model, x, *adj)
        wh, bh = head(model)
        loss = (
            sum(
                checkpoint(_loss_sum, z[r], y[r], wh, bh, weight, use_reentrant=False)
                for r in tr.split(chunk)
            )
            / norm_const
        )
        loss.backward()
        opt.step()
        pred = predict_rows(model, x, adj, va, chunk).cpu().numpy()
        val_acc = float((pred == y_val).mean())
        val_f1 = float(f1_score(y_val, pred, average="macro"))
        history.append(
            {"epoch": epoch, "loss": loss.item(), "val_acc": val_acc, "val_macro_f1": val_f1}
        )
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return Fitted(model, x, y, adj, history)
