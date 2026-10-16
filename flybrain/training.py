"""GNN training shared by train_gnn.py (one graph) and transfer.py (FAFB -> MCNS)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data

from flybrain.features import FEATURE_NAMES, degree_features, normalize_features
from flybrain.models import DirectedSAGE, MeanBuilder, class_weights, drop_edges


@dataclass
class Fitted:
    """A trained model with the inputs it was trained on (on the training device)."""

    model: DirectedSAGE
    x: torch.Tensor  # normalized node features
    y: torch.Tensor
    adj: tuple  # (mean over inputs, mean over outputs or None)
    history: list[dict] = field(default_factory=list)


def graph_inputs(data: Data, x: torch.Tensor, cfg: dict, device) -> tuple[torch.Tensor, tuple]:
    """Features and aggregation matrices of a graph, as the model with config `cfg` expects."""
    means = MeanBuilder(
        data.edge_index.to(device),
        data.num_nodes,
        data.edge_weight.to(device),
        cfg.get("edge_weight", "count"),
    )
    return x.to(device), means(cfg.get("bidirectional", True))


@torch.no_grad()
def predict(model: DirectedSAGE, x: torch.Tensor, adj: tuple) -> np.ndarray:
    """Predicted class index of every node."""
    model.eval()
    return model(x, *adj).argmax(1).cpu().numpy()


@torch.no_grad()
def evaluate(model, x, adj, y, mask) -> tuple[float, float]:
    """(accuracy, macro-F1) on the masked nodes."""
    model.eval()
    pred = model(x, *adj).argmax(1)[mask].cpu()
    true = y[mask].cpu()
    return float((pred == true).float().mean()), float(f1_score(true, pred, average="macro"))


def fit_gnn(cfg: dict, data: Data, train: torch.Tensor, val: torch.Tensor, device) -> Fitted:
    """Train a DirectedSAGE on `data` (raw degree features) and keep the epoch with the best
    validation macro-F1. Features are normalized as cfg["feature_norm"] says (statistics from the
    training nodes for "global"); cfg["drop_edge"] > 0 trains on randomly thinned subgraphs."""
    norm, side = cfg.get("feature_norm", "global"), getattr(data, "side", None)
    raw_x = data.x
    x = normalize_features(raw_x, norm, train, side).to(device)
    y = data.y.to(device)
    train_d, val_d = train.to(device), val.to(device)
    bidirectional = cfg.get("bidirectional", True)
    ei, n, w = data.edge_index.to(device), data.num_nodes, data.edge_weight.to(device)
    means = MeanBuilder(ei, n, w, cfg.get("edge_weight", "count"))
    adj = means(bidirectional)

    # augmentation: each epoch trains on a subgraph with randomly dropped edges, with degree
    # features recomputed from it (scaled like the full graph), so the model cannot rely on
    # exact degrees; evaluation always uses the full graph
    drop_rate = cfg.get("drop_edge", 0.0)
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])

    def augmented() -> tuple[torch.Tensor, tuple]:
        keep = drop_edges(ei, n, drop_rate, gen)
        sub = degree_features(n, ei[:, keep].cpu(), w[keep].cpu())
        x_sub = normalize_features(torch.from_numpy(sub).float(), norm, train, side, raw_x)
        return x_sub.to(device), means(bidirectional, keep)

    num_classes = len(data.label_names)
    weight = class_weights(y[train_d], num_classes, cfg.get("class_weight", "none"))
    model = DirectedSAGE(
        x.size(1), cfg["hidden_dim"], num_classes, cfg["num_layers"], cfg["dropout"], bidirectional
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # select the epoch with the best validation macro-F1 (accuracy hides the rare classes)
    history, best_val, best_state = [], -1.0, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        opt.zero_grad()
        x_ep, adj_ep = augmented() if drop_rate > 0 else (x, adj)
        loss = F.cross_entropy(model(x_ep, *adj_ep)[train_d], y[train_d], weight=weight)
        loss.backward()
        opt.step()
        val_acc, val_f1 = evaluate(model, x, adj, y, val_d)
        history.append(
            {"epoch": epoch, "loss": loss.item(), "val_acc": val_acc, "val_macro_f1": val_f1}
        )
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return Fitted(model, x, y, adj, history)


def load_model(path, device="cpu") -> tuple[DirectedSAGE, dict, list[str]]:
    """A GNN saved by transfer.py (model.pt): (model in eval mode, config, label names)."""
    saved = torch.load(path, map_location=device, weights_only=False)
    cfg, names = saved["cfg"], saved["label_names"]
    model = DirectedSAGE(
        len(FEATURE_NAMES),
        cfg["hidden_dim"],
        len(names),
        cfg["num_layers"],
        cfg["dropout"],
        cfg.get("bidirectional", True),
    ).to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval(), cfg, names
