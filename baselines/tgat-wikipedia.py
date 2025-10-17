"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
#import argparse

import torch
import pandas as pd
import numpy as np
#import numba
import argparse 

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler

class Args:
    data = "wikipedia"
    bs = 100
    n_epoch = 50
    n_layer = 2
    n_head = 2
    lr = 0.0001
    drop_out = 0.1
    gpu = 1
    node_dim = 64
    time_dim = 64
    agg_method = "attn"
    attn_mode = "prod"
    time = "time"
    uniform = True
    prefix = "Hello World"
    n_degree = 10
    new_node = False   
    eval_neg_per_pos = 10

args = Args()

### Argument and global variables
'''
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
'''
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
NEG_PER_POS = 10

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

def set_seed(seed: int):
    """Ensure reproducibility per run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def mean_reciprocal_rank(group_scores: np.ndarray,
                         group_labels: np.ndarray,
                         group_size: int) -> float:
    """
    group_scores: (num_groups, group_size)
    group_labels: (num_groups, group_size) with exactly 1 positive per row
    """
    assert group_scores.shape[1] == group_size
    num_groups = group_scores.shape[0]
    rr_total = 0.0

    for g in range(num_groups):
        sg = group_scores[g]
        yg = group_labels[g]
        assert yg.sum() == 1, "Each group must contain exactly one positive."
        order = np.argsort(-sg)  # descending
        pos_idx = np.where(yg == 1)[0][0]
        rank = np.where(order == pos_idx)[0][0] + 1
        rr_total += 1.0 / rank

    return rr_total / num_groups


def prepare_group_scores(pos_prob, neg_prob, num_groups, neg_per_pos):
    """
    Returns (group_scores, group_labels) shaped (num_groups, 1+neg_per_pos).
    Ensures each group has exactly 1 positive.
    """
    pos = pos_prob.cpu().detach().numpy().reshape(num_groups, 1)
    neg = neg_prob.cpu().detach().numpy().reshape(num_groups, neg_per_pos)
    group_scores = np.hstack([pos, neg])

    pos_labels = np.ones((num_groups, 1))
    neg_labels = np.zeros((num_groups, neg_per_pos))
    group_labels = np.hstack([pos_labels, neg_labels])

    return group_scores, group_labels
def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label, num_neighbors, neg_per_pos):
    val_acc, val_ap, val_auc, val_mrr = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]

            size = len(src_l_cut)
            _, dst_l_fake = sampler.sample(size * neg_per_pos)
            ts_l_fake = np.repeat(ts_l_cut, neg_per_pos)

            pos_prob, neg_prob = tgan.contrast(
                src_l_cut, dst_l_cut, ts_l_cut,
                dst_l_fake, ts_l_fake,
                num_neighbors
            )
            group_scores, group_labels = prepare_group_scores(pos_prob, neg_prob, size, neg_per_pos)

            pred_score = group_scores.flatten()
            true_label = group_labels.flatten()
            pred_label = pred_score > 0.5

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
            val_mrr.append(mean_reciprocal_rank(group_scores, group_labels, 1 + neg_per_pos))
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_auc), np.mean(val_mrr)


# ---------------------------
# Single experiment runner
# ---------------------------

def run_single_experiment(seed, args, device):
    set_seed(seed)
    start_time = time.time()

    # === Load data ===
    g_df = pd.read_csv(f'./processed/ml_{args.data}.csv')
    e_feat = np.load(f'./processed/ml_{args.data}.npy')
    n_feat = np.load(f'./processed/ml_{args.data}_node.npy')

    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
    src_l, dst_l, e_idx_l, label_l, ts_l = g_df.u.values, g_df.i.values, g_df.idx.values, g_df.label.values, g_df.ts.values
    max_idx = max(src_l.max(), dst_l.max())

    total_node_set = set(np.unique(np.hstack([src_l, dst_l])))
    num_total_unique_nodes = len(total_node_set)

    eligible_nodes = set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time]))
    mask_node_set = set(random.sample(list(eligible_nodes), int(0.1 * num_total_unique_nodes)))

    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)

    valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = (
        src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag],
        e_idx_l[valid_train_flag], label_l[valid_train_flag]
    )

    train_node_set = set(train_src_l).union(train_dst_l)
    new_node_set = total_node_set - train_node_set

    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time
    is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
    nn_val_flag = valid_val_flag * is_new_node_edge
    nn_test_flag = valid_test_flag * is_new_node_edge

    val_src_l, val_dst_l, val_ts_l, val_label_l = src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], label_l[valid_val_flag]
    test_src_l, test_dst_l, test_ts_l, test_label_l = src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], label_l[valid_test_flag]
    nn_val_src_l, nn_val_dst_l, nn_val_ts_l, nn_val_label_l = src_l[nn_val_flag], dst_l[nn_val_flag], ts_l[nn_val_flag], label_l[nn_val_flag]
    nn_test_src_l, nn_test_dst_l, nn_test_ts_l, nn_test_label_l = src_l[nn_test_flag], dst_l[nn_test_flag], ts_l[nn_test_flag], label_l[nn_test_flag]

    # Build adjacency
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list, uniform=args.uniform)

    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, uniform=args.uniform)

    train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
    val_rand_sampler = RandEdgeSampler(src_l, dst_l)
    nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
    test_rand_sampler = RandEdgeSampler(src_l, dst_l)
    nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)

    # Initialize model
    tgan = TGAN(train_ngh_finder, n_feat, e_feat,
                num_layers=args.n_layer, use_time=args.time, agg_method=args.agg_method, attn_mode=args.attn_mode,
                seq_len=args.n_degree, n_head=args.n_head, drop_out=args.drop_out, node_dim=args.node_dim,
                time_dim=args.time_dim, neg_per_pos=args.eval_neg_per_pos)

    tgan = tgan.to(device)
    optimizer = torch.optim.Adam(tgan.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / args.bs)
    idx_list = np.arange(num_instance)

    early_stopper = EarlyStopMonitor()

    # Training loop
    for epoch in range(args.n_epoch):
        np.random.shuffle(idx_list)
        tgan.ngh_finder = train_ngh_finder
        acc, ap, auc, mrr, m_loss = [], [], [], [], []
        epoch_start = time.time()

        for k in range(num_batch):
            s_idx, e_idx = k * args.bs, min(num_instance - 1, (k + 1) * args.bs)
            src_cut, dst_cut, ts_cut, label_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx], train_ts_l[s_idx:e_idx], train_label_l[s_idx:e_idx]
            size = len(src_cut)

            _, dst_fake = train_rand_sampler.sample(size * args.eval_neg_per_pos)
            ts_fake = np.repeat(ts_cut, args.eval_neg_per_pos)

            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size * args.eval_neg_per_pos, dtype=torch.float, device=device)

            optimizer.zero_grad()
            tgan = tgan.train()
            pos_prob, neg_prob = tgan.contrast(src_cut, dst_cut, ts_cut, dst_fake, ts_fake, args.n_degree)

            loss = criterion(pos_prob, pos_label) + criterion(neg_prob.reshape(-1), neg_label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tgan.eval()
                group_scores, group_labels = prepare_group_scores(pos_prob, neg_prob, size, args.eval_neg_per_pos)
                pred_score = group_scores.flatten()
                true_label = group_labels.flatten()
                pred_label = pred_score > 0.5
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                auc.append(roc_auc_score(true_label, pred_score))
                mrr.append(mean_reciprocal_rank(group_scores, group_labels, 1 + args.eval_neg_per_pos))
                m_loss.append(loss.item())

        # Validation
        tgan.ngh_finder = full_ngh_finder
        val_acc, val_ap, val_auc, val_mrr = eval_one_epoch('val', tgan, val_rand_sampler, val_src_l, val_dst_l, val_ts_l, val_label_l, args.n_degree, args.eval_neg_per_pos)

        if early_stopper.early_stop_check(val_ap):
            break

    # Testing
    tgan.ngh_finder = full_ngh_finder
    test_acc, test_ap, test_auc, test_mrr = eval_one_epoch('test', tgan, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, args.n_degree, args.eval_neg_per_pos)

    avg_epoch_time = (time.time() - start_time) / args.n_epoch
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if torch.cuda.is_available() else 0

    result = {
        "ap": test_ap,
        "auc": test_auc,
        "mrr": test_mrr,
        "avg_epoch_time": avg_epoch_time,
        "peak_memory_mb": peak_mem,
    }
    return result, None


# ---------------------------
# Multi-run wrapper
# ---------------------------

def run_multiple_experiments(args, seeds):
    all_results = []
    for seed in seeds:
        print(f"\n===== Running experiment with seed {seed} =====")
        result, _ = run_single_experiment(seed, args, device)
        all_results.append(result)

    test_aps = [r["ap"] for r in all_results]
    test_aucs = [r["auc"] for r in all_results]
    test_mrrs = [r["mrr"] for r in all_results]
    epoch_times = [r["avg_epoch_time"] for r in all_results]
    peak_mems = [r["peak_memory_mb"] for r in all_results]

    print("\n--- Final Performance Results (Mean ± Std Dev over runs) ---")
    print(f"Test AP:  {np.mean(test_aps):.4f} ± {np.std(test_aps):.4f}")
    print(f"Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"Test MRR: {np.mean(test_mrrs):.4f} ± {np.std(test_mrrs):.4f}")
    print("-" * 60)
    print("\n--- Final Efficiency Results (Mean ± Std Dev over runs) ---")
    print(f"Avg. Runtime/Epoch: {np.mean(epoch_times):.2f}s ± {np.std(epoch_times):.2f}s")
    if torch.cuda.is_available() and np.mean(peak_mems) > 0:
        print(f"Peak Memory Usage:  {np.mean(peak_mems):.2f} MB ± {np.std(peak_mems):.2f} MB")
    else:
        print("Peak Memory Usage:  N/A")


# ---------------------------
# Main entry
# ---------------------------

if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    seeds = [123451, 123452, 123453, 123454, 123455]
    run_multiple_experiments(args, seeds)

