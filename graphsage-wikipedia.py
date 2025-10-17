import os
import csv
import math
import random
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import Data, TemporalData
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.datasets import JODIEDataset


# ============ Config & utils ============

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    s = scores.detach().cpu()
    y = labels.detach().cpu().long()
    order = torch.argsort(s, descending=True)
    y_sorted = y[order]
    tp_cum = torch.cumsum((y_sorted == 1).float(), dim=0)
    ranks = torch.arange(1, y_sorted.numel() + 1, dtype=torch.float)
    precisions = tp_cum / ranks
    pos = (y == 1).sum().item()
    if pos == 0:
        return float("nan")
    return precisions[(y_sorted == 1)].mean().item()


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    s = scores.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    order = np.argsort(s)  # ascending
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)

    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    for val_idx, cnt in enumerate(counts):
        if cnt > 1:
            tie_positions = np.where(inv == val_idx)[0]
            avg_rank = ranks[tie_positions].mean()
            ranks[tie_positions] = avg_rank

    P = float((y == 1).sum())
    N = float((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    pos_ranks = ranks[y == 1]
    auc = (pos_ranks.sum() - P * (P + 1) / 2.0) / (P * N)
    return float(auc)


def mean_reciprocal_rank(group_scores: torch.Tensor,
                         group_labels: torch.Tensor,
                         group_size: int) -> float:
    s = group_scores.detach().cpu().numpy()
    y = group_labels.detach().cpu().numpy()
    assert s.shape[0] % group_size == 0
    num_groups = s.shape[0] // group_size
    rr_total = 0.0
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        sg = s[start:end]
        yg = y[start:end]
        assert yg.sum() == 1, "Each group must contain exactly one positive."
        order = np.argsort(-sg)  # descending
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, group_size + 1, dtype=float)

        uniq, inv, counts = np.unique(sg, return_inverse=True, return_counts=True)
        for val_idx, cnt in enumerate(counts):
            if cnt > 1:
                tie_positions = np.where(inv == val_idx)[0]
                avg_rank = ranks[tie_positions].mean()
                ranks[tie_positions] = avg_rank

        pos_idx = int(np.where(yg == 1)[0][0])
        rr_total += 1.0 / float(ranks[pos_idx])
    return rr_total / num_groups


def write_metrics_csv(path: Path, rows: List[dict]):
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "epoch", "AP", "AUC", "MRR"])
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ============ Model ============

class GraphSAGE(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, emb_dim)
        convs = []
        in_dim = emb_dim
        for _ in range(num_layers):
            convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.convs = nn.ModuleList(convs)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def encode(self, n_id, edge_index):
        x = self.emb(n_id)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


# ============ Negatives for eval ============

def build_excluded_edge_set(src_all: torch.Tensor, dst_all: torch.Tensor) -> set:
    return set(zip(src_all.tolist(), dst_all.tolist()))


def sample_negatives_same_source(pos_src, pos_dst, num_nodes, k_neg, excluded, seed=42):
    rng = random.Random(seed)
    neg_src, neg_dst = [], []
    for u, _v in zip(pos_src.tolist(), pos_dst.tolist()):
        picked = set()
        while len(picked) < k_neg:
            w = rng.randrange(0, num_nodes)
            if w == u or (u, w) in excluded or w in picked:
                continue
            picked.add(w)
            neg_src.append(u)
            neg_dst.append(w)
    return torch.tensor(neg_src, dtype=torch.long), torch.tensor(neg_dst, dtype=torch.long)


def build_eval_edge_labels(pos_src, pos_dst, num_nodes, k_neg, excluded, seed=42):
    neg_src, neg_dst = sample_negatives_same_source(pos_src, pos_dst, num_nodes, k_neg, excluded, seed)
    blocks, labels = [], []
    nptr = 0
    for u, v in zip(pos_src.tolist(), pos_dst.tolist()):
        blocks.append([u, v]); labels.append(1)
        for _ in range(k_neg):
            nu, nv = neg_src[nptr].item(), neg_dst[nptr].item()
            blocks.append([nu, nv]); labels.append(0)
            nptr += 1
    edge_arr = torch.tensor(blocks, dtype=torch.long).t().contiguous()
    label_arr = torch.tensor(labels, dtype=torch.long)
    return edge_arr, label_arr


# ============ Training routine ============

def run_single_experiment(seed, args, device):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    total_start = time.time()

    # --- load dataset (replace with your GDELTDataset if needed) ---
    if args.dataset.lower() == "wikipedia":
        ds = JODIEDataset(root=os.path.join(args.data_root, "jodie"), name="wikipedia")
        tdata = ds[0]
    elif args.dataset.lower() == "gdelt":
        dataset = GDELTDataset(root=args.data_root)  # <-- you need to define/import this
        tdata = dataset[0]
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    src, dst, t = tdata.src.view(-1).long(), tdata.dst.view(-1).long(), tdata.t.view(-1).long()

    # sort + split
    perm = torch.argsort(t)
    src, dst, t = src[perm], dst[perm], t[perm]
    nE = src.numel()
    i_train_end = int(0.70 * nE)
    i_val_end = int(0.85 * nE)

    src_train, dst_train = src[:i_train_end], dst[:i_train_end]
    src_val, dst_val = src[i_train_end:i_val_end], dst[i_train_end:i_val_end]
    src_test, dst_test = src[i_val_end:], dst[i_val_end:]

    num_nodes = int(torch.max(torch.stack([src, dst])).item()) + 1

    # train graph
    edge_index_train = torch.stack([src_train, dst_train], dim=0)
    edge_index_train = to_undirected(edge_index_train, num_nodes=num_nodes)
    train_graph = Data(edge_index=edge_index_train, num_nodes=num_nodes)

    # loaders
    train_loader = LinkNeighborLoader(
        data=train_graph,
        num_neighbors=args.neighbors,
        batch_size=args.batch_size,
        edge_label_index=torch.stack([src_train, dst_train], dim=0),
        neg_sampling_ratio=args.neg_sampling_ratio,
        shuffle=True,
    )

    excluded = build_excluded_edge_set(
        torch.cat([src_train, src_val, src_test]),
        torch.cat([dst_train, dst_val, dst_test]),
    )
    val_edge_label_index, val_edge_label = build_eval_edge_labels(
        src_val, dst_val, num_nodes, args.eval_neg_per_pos, excluded, seed=seed
    )
    test_edge_label_index, test_edge_label = build_eval_edge_labels(
        src_test, dst_test, num_nodes, args.eval_neg_per_pos, excluded, seed=seed
    )

    val_loader = LinkNeighborLoader(
        data=train_graph,
        num_neighbors=args.neighbors,
        batch_size=args.batch_size,
        edge_label_index=val_edge_label_index,
        edge_label=val_edge_label,
        neg_sampling_ratio=1.0,
        shuffle=False,
    )
    test_loader = LinkNeighborLoader(
        data=train_graph,
        num_neighbors=args.neighbors,
        batch_size=args.batch_size,
        edge_label_index=test_edge_label_index,
        edge_label=test_edge_label,
        neg_sampling_ratio=1.0,
        shuffle=False,
    )

    # model + optimizer
    model = GraphSAGE(num_nodes, args.emb_dim, args.hidden_dim, args.layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    def train_epoch():
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            z = model.encode(batch.n_id, batch.edge_index)
            logits = model.decode(z, batch.edge_label_index)
            loss = bce(logits, batch.edge_label.float())

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                acc = (preds == batch.edge_label.long()).float().mean()

            total_loss += float(loss.detach().cpu())
            total_acc += float(acc.detach().cpu())
            n += 1
        return total_loss / max(1, n), total_acc / max(1, n)

    @torch.no_grad()
    def eval_metrics_full(edge_label_index, edge_label, group_size: int, train_graph, device):
        model.eval()
        n_id = torch.arange(train_graph.num_nodes, device=device)
        z = model.encode(n_id, train_graph.edge_index.to(device))
        logits = model.decode(z, edge_label_index.to(device))
        s, y = logits.cpu(), edge_label.cpu()
        ap = average_precision(s, y)
        auc = roc_auc(s, y)
        mrr = mean_reciprocal_rank(s, y, group_size)
        return ap, auc, mrr

    # training loop with early stopping
    best_val_ap = float("-inf")
    best_test_metrics = None
    patience_counter = 0
    patience_limit = 15

    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        tr_loss, tr_acc = train_epoch()

        if epoch % args.val_every == 0:
            ap, auc, mrr = eval_metrics_full(
                val_edge_label_index, val_edge_label,
                1 + args.eval_neg_per_pos, train_graph, device
            )
            if ap > best_val_ap:
                best_val_ap = ap
                patience_counter = 0
                ap_t, auc_t, mrr_t = eval_metrics_full(
                    test_edge_label_index, test_edge_label,
                    1 + args.eval_neg_per_pos, train_graph, device
                )
                best_test_metrics = {"AP": ap_t, "AUC": auc_t, "MRR": mrr_t}
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    break

        epoch_times.append(time.time() - epoch_start)

    total_runtime = time.time() - total_start

    result = {
        "ap": best_test_metrics["AP"] if best_test_metrics else 0,
        "auc": best_test_metrics["AUC"] if best_test_metrics else 0,
        "mrr": best_test_metrics["MRR"] if best_test_metrics else 0,
        "avg_epoch_time": np.mean(epoch_times),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0,
        "total_runtime": total_runtime,
    }

    return result, []  # history placeholder


def run_multiple_experiments(args, seeds=[42, 43, 44, 45, 46]):
    all_results = []
    for seed in seeds:
        print(f"\n===== Running experiment with seed {seed} =====")
        result, history = run_single_experiment(seed, args, device)
        all_results.append(result)

    test_aps = [r["ap"] for r in all_results]
    test_aucs = [r["auc"] for r in all_results]
    test_mrrs = [r["mrr"] for r in all_results]
    epoch_times = [r["avg_epoch_time"] for r in all_results]
    peak_mems = [r["peak_memory_mb"] for r in all_results]

    mean_ap, std_ap = np.mean(test_aps), np.std(test_aps)
    mean_auc, std_auc = np.mean(test_aucs), np.std(test_aucs)
    mean_mrr, std_mrr = np.mean(test_mrrs), np.std(test_mrrs)
    mean_time, std_time = np.mean(epoch_times), np.std(epoch_times)
    mean_mem, std_mem = np.mean(peak_mems), np.std(peak_mems)

    print("\n--- Final Performance Results (Mean ± Std Dev over 5 Runs) ---")
    print(f"Test AP:  {mean_ap:.4f} ± {std_ap:.4f}")
    print(f"Test AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Test MRR: {mean_mrr:.4f} ± {std_mrr:.4f}")
    print("-" * 60)
    print("\n--- Final Efficiency Results (Mean ± Std Dev over 5 Runs) ---")
    print(f"Avg. Runtime/Epoch: {mean_time:.2f}s ± {std_time:.2f}s")
    if torch.cuda.is_available() and mean_mem > 0:
        print(f"Peak Memory Usage:  {mean_mem:.2f} MB ± {std_mem:.2f} MB")
    else:
        print("Peak Memory Usage:  N/A (Not a CUDA device)")


if __name__ == "__main__":
    import sys
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [sys.argv[0]]  # reset argv so argparse doesn’t choke
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--dataset", type=str, default="wikipedia")  # change if needed
    p.add_argument("--epochs", type=int, default=51)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--neighbors", type=int, nargs="+", default=[15, 10])
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--neg-sampling-ratio", type=float, default=1.0)
    p.add_argument("--eval-neg-per-pos", type=int, default=10)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="runs")
    args, _ = p.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    seeds = [123451, 123452, 123453, 123454, 123455]
    run_multiple_experiments(args, seeds=seeds)
