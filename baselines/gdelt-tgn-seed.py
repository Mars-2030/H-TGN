import os
import os.path as osp
import random
import sys
import time
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.data import InMemoryDataset, TemporalData, download_url
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (IdentityMessage, LastAggregator,
                                           LastNeighborLoader)





class GDELTDataset(InMemoryDataset):
    r"""The temporal knowledge graph dataset from the GDELT benchmark dataset,
    which combines train, validation, and test files into a single dataset.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to
            disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        bin_15min (bool, optional): Whether to floor-divide timestamps by 15
            (a common preprocessing step for GDELT). (default: :obj:`True`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #entities
          - #relations
          - #events
        * - GDELT
          - varies
          - varies
          - varies
    """
    url = 'https://github.com/INK-USC/RE-Net/raw/master/data/GDELT'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        bin_15min: bool = True,
    ) -> None:
        self.bin_15min = bin_15min
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=TemporalData)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'GDELT', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'GDELT', 'processed')

    @property
    def raw_file_names(self) -> list:
        # List of files to download and process
        return ['train.txt', 'valid.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        # Download each file separately
        for file_name in self.raw_file_names:
            download_url(f"{self.url}/{file_name}", self.raw_dir)

    def process(self) -> None:

        # List to hold DataFrames for each file
        dfs = []
        # List to hold DataFrames for each file

        for file_name in self.raw_file_names:
            file_path = osp.join(self.raw_dir, file_name)
            df = pd.read_csv(
                file_path,
                sep='\t',
                header=None,
                usecols=[0, 1, 2, 3],
                names=['src', 'relation', 'dst', 'timestamp']
            )

            if self.bin_15min:
                # Common GDELT preprocessing: 15-minute bins
                df['timestamp'] = df['timestamp'] // 15

            dfs.append(df)

            # Draw histograms for each column
            # columns = ['src', 'relation', 'dst', 'timestamp']

            # for column in columns:
            #     plt.figure(figsize=(8, 5))
            #     plt.hist(df[column], bins=30, alpha=0.7, edgecolor='black')
            #     plt.title(f'Histogram of {column}')
            #     plt.xlabel(column)
            #     plt.ylabel('Frequency')
            #     plt.grid(axis='y', alpha=0.75)
            #     plt.show()

        # Concatenate all dataframes into a single dataframe
        full_df = pd.concat(dfs, ignore_index=True)
        print(f"Total events (triples) in combined dataset: {len(full_df)}")

        # Process columns for temporal data
        src = torch.tensor(full_df['src'].values, dtype=torch.long)
        dst = torch.tensor(full_df['dst'].values, dtype=torch.long)
        # relation = torch.tensor(full_df['relation'].values, dtype=torch.float).tolist()
        relation = torch.tensor(full_df['relation'].values, dtype=torch.float).view(-1, 1)
        timestamp = torch.tensor(full_df['timestamp'].values, dtype=torch.long)

        print("\nTensor shapes:")
        print(f"src: {src.shape}")
        print(f"dst: {dst.shape}")
        print(f"t: {timestamp.shape}")
        print(f"msg: {relation.shape}")

        # Create the TemporalData object
        data = TemporalData(src=src, dst=dst, t=timestamp, msg=relation)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Print debug information about the data object
        print("Data object before saving:")
        print(f"  - Number of source nodes (src): {data.src.size(0)}")
        print(f"  - Number of destination nodes (dst): {data.dst.size(0)}")
        print(f"  - Number of timestamps (t): {data.t.size(0)}")
        print(f"  - Number of relations (msg): {data.msg.size(0)}")

        # Save the processed data
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'GDELTDataset(bin_15min={self.bin_15min})'



# (Tee class definition remains the same)
class Tee(object):
    def __init__(self, filename, mode='w'):
        self.file, self.stdout = open(filename, mode), sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data); self.stdout.write(data); self.flush()
    def flush(self):
        self.file.flush(); self.stdout.flush()

# (Helper functions and Model Components remain the same)
def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)

# NEW: Worker init function for DataLoader for reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim)
    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        edge_attr = torch.cat([self.time_enc(rel_t.to(x.dtype)), msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src, self.lin_dst, self.lin_final = Linear(in_channels, in_channels), Linear(in_channels, in_channels), Linear(in_channels, 1)
    def forward(self, z_src, z_dst):
        return self.lin_final((self.lin_src(z_src) * self.lin_dst(z_dst)).relu())

def plot_tsne_node_embeddings(history, tracked_nodes, seed):
    if not history:
        print(f"No embedding history for seed {seed}. Skipping t-SNE plot.")
        return
    print(f"\nGenerating t-SNE plot for node embeddings (Seed {seed})...")
    
    epochs = [item[0] for item in history]
    tensors = [item[1] for item in history]
    
    all_embeddings = torch.cat(tensors, dim=0).numpy()
    
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_embeddings) - 1), 
                max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(all_embeddings)
    
    num_epochs = len(epochs)
    num_tracked = len(tracked_nodes)
    tsne_by_epoch = tsne_results.reshape(num_epochs, num_tracked, 2)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 14))
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_tracked))

    for i in range(num_tracked):
        trajectory = tsne_by_epoch[:, i, :]
        plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle='-', color=colors[i], alpha=0.7, linewidth=1.5,
                 label=f'Node {tracked_nodes[i]}' if num_tracked <= 15 else None)
        plt.scatter(trajectory[0, 0], trajectory[0, 1], color=colors[i], s=30, edgecolor='black', marker='o', zorder=3)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color=colors[i], s=40, edgecolor='black', marker='s', zorder=3)

    plt.title(f't-SNE Visualization of Node Embedding Trajectories (Seed {seed})', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    if num_tracked <= 15: plt.legend()
    plt.tight_layout()
    plot_filename = f"gdelt_tgn_seed_{seed}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"t-SNE plot saved to {plot_filename}")

# MODIFIED: train_epoch now returns a more detailed tuple
def train_epoch(loader, memory, gnn, link_pred, optimizer, neighbor_loader, assoc, data, criterion):
    memory.train(); gnn.train(); link_pred.train()
    memory.reset_state(); neighbor_loader.reset_state()
    total_loss, total_pos_loss, total_neg_loss = 0.0, 0.0, 0.0
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device, non_blocking=True)
        
        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z_mem, last_update = memory(n_id)
        z_tgn = gnn(z_mem, last_update, edge_index, data.t[e_id.cpu()].to(device), data.msg[e_id.cpu()].to(device))

        pos_out = link_pred(z_tgn[assoc[batch.src]], z_tgn[assoc[batch.dst]])
        neg_out = link_pred(z_tgn[assoc[batch.src]], z_tgn[assoc[batch.neg_dst]])
        
        # MODIFIED: Calculate positive and negative loss separately
        loss_pos = criterion(pos_out, torch.ones_like(pos_out))
        loss_neg = criterion(neg_out, torch.zeros_like(neg_out))
        loss = loss_pos + loss_neg

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in optimizer.param_groups[0]['params'] if p.requires_grad], 0.5)
        optimizer.step()
        
        memory.update_state(batch.src, batch.dst, batch.t.to(memory.last_update.dtype), batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
        memory.detach()
        
        total_loss += float(loss) * batch.num_events
        total_pos_loss += float(loss_pos) * batch.num_events
        total_neg_loss += float(loss_neg) * batch.num_events
        
    # MODIFIED: Return the new detailed loss components
    return (total_loss / loader.data.num_events, 
            total_pos_loss / loader.data.num_events, 
            total_neg_loss / loader.data.num_events)

@torch.no_grad()
def evaluate(loader, memory, gnn, link_pred, neighbor_loader, assoc, data, criterion):
    memory.eval(); gnn.eval(); link_pred.eval()
    aps, aucs, mrrs, total_loss, total_events = [], [], [], 0.0, 0
    all_z_tgn, all_n_id = [], []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        
        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z_mem, last_update = memory(n_id)
        z_tgn = gnn(z_mem, last_update, edge_index, data.t[e_id.cpu()].to(device), data.msg[e_id.cpu()].to(device))
        
        all_z_tgn.append(z_tgn.cpu())
        all_n_id.append(n_id.cpu())

        pos_out = link_pred(z_tgn[assoc[batch.src]], z_tgn[assoc[batch.dst]])
        neg_out = link_pred(z_tgn[assoc[batch.src]], z_tgn[assoc[batch.neg_dst]])
        loss = criterion(pos_out, torch.ones_like(pos_out)) + criterion(neg_out, torch.zeros_like(neg_out))
        total_loss += float(loss) * batch.num_events; total_events += batch.num_events
        
        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)
        aps.append(average_precision_score(y_true, y_pred)); aucs.append(roc_auc_score(y_true, y_pred))
        
        scores = torch.cat([pos_out, neg_out], dim=1).cpu()
        ranks = (scores.sort(dim=1, descending=True)[1] == 0).nonzero(as_tuple=False)[:, 1] + 1
        mrrs.append((1.0 / ranks.float()).mean().item())
        
        memory.update_state(batch.src, batch.dst, batch.t.to(memory.last_update.dtype), batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

    full_z_tgn = torch.zeros(data.num_nodes, z_tgn.size(1))
    for n_id_batch, z_tgn_batch in zip(all_n_id, all_z_tgn):
        full_z_tgn[n_id_batch] = z_tgn_batch
    
    return total_loss / total_events, float(np.mean(aps)), float(np.mean(aucs)), float(np.mean(mrrs)), full_z_tgn

def run_single_experiment(seed, data, train_data, val_loader, test_loader, device, loader_kwargs):
    print(f"\n{'='*30} RUNNING SEED {seed} {'='*30}")
    set_seed(seed)
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    
    memory_dim, time_dim, embedding_dim = 100, 100, 100
    memory = TGNMemory(data.num_nodes, data.msg.size(-1), memory_dim, time_dim, message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim), aggregator_module=LastAggregator()).to(device)
    gnn = GraphAttentionEmbedding(memory_dim, embedding_dim, data.msg.size(-1), memory.time_enc).to(device)
    link_pred = LinkPredictor(embedding_dim).to(device)
    
    modules = [memory, gnn, link_pred]
    all_params = [p for m in modules for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(all_params, lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = torch.nn.BCEWithLogitsLoss()
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=20, device=device)
    train_loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=1.0, **loader_kwargs)

    best_val_ap, best_test_metrics = 0, {}
    patience_counter = 0
    best_ema_val_ap = 0.0
    ema_val_ap = 0.0
    MIN_EPOCHS_FOR_STOPPING = 15
    EMA_ALPHA = 0.2

    epoch_times, embedding_history = [], []
    capture_epochs = {1, 5, 10, 20, 30, 40, 50}

    for epoch in range(1, 51):
        epoch_start_time = time.time()
        
        # MODIFIED: Unpack the new detailed loss tuple
        train_loss, train_pos_loss, train_neg_loss = train_epoch(train_loader, memory, gnn, link_pred, optimizer, neighbor_loader, assoc, data, criterion)
        val_loss, val_ap, val_auc, val_mrr, epoch_embeddings = evaluate(val_loader, memory, gnn, link_pred, neighbor_loader, assoc, data, criterion)
        
        scheduler.step(val_loss)
        epoch_times.append(time.time() - epoch_start_time)
        
        if epoch == 1:
            ema_val_ap = val_ap
        else:
            ema_val_ap = EMA_ALPHA * val_ap + (1 - EMA_ALPHA) * ema_val_ap

        print(f"Seed {seed}|Epoch {epoch:02d}|LR {optimizer.param_groups[0]['lr']:.6f}|Time {epoch_times[-1]:.2f}s|Train Loss {train_loss:.4f}")
        # MODIFIED: Print the new detailed training stats
        print(f"  Train Stats: Total={train_loss:.4f} | Pos_Loss={train_pos_loss:.4f} | Neg_Loss={train_neg_loss:.4f}")
        print(f"  Val   Stats: Loss={val_loss:.4f}|AP {val_ap:.4f} (EMA AP {ema_val_ap:.4f})|AUC {val_auc:.4f}|MRR {val_mrr:.4f}")

        if epoch in capture_epochs:
            embedding_history.append((epoch, epoch_embeddings))
            print(f"--- Captured node embeddings for t-SNE at epoch {epoch} ---")

        if ema_val_ap > best_ema_val_ap:
            best_ema_val_ap = ema_val_ap
            best_val_ap = val_ap
            patience_counter = 0
            
            _, test_ap, test_auc, test_mrr, _ = evaluate(test_loader, memory, gnn, link_pred, neighbor_loader, assoc, data, criterion)
            best_test_metrics = {"ap": test_ap, "auc": test_auc, "mrr": test_mrr}
            print(f"  >>> New best model! Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Test MRR: {test_mrr:.4f}")
            if not os.path.exists('saved_models'): os.makedirs('saved_models')
            torch.save({'memory': memory.state_dict(), 'gnn': gnn.state_dict(), 'link_pred': link_pred.state_dict()}, osp.join('saved_models', f'gdelt_tgn_seed_{seed}.pth'))
        else:
            if epoch > MIN_EPOCHS_FOR_STOPPING:
                patience_counter += 1
        
        if patience_counter >= 15:
            print("Early stopping triggered."); break
    
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024*1024) if device.type == 'cuda' else 0
    best_test_metrics['avg_epoch_time'], best_test_metrics['peak_memory_mb'] = avg_epoch_time, peak_memory_mb
    
    print(f"--- Finished Seed {seed} -> Best Test Metrics: {best_test_metrics} ---")
    
    return best_test_metrics, embedding_history

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [123451, 123452, 123453, 123454, 123455]

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"Could not enforce deterministic algorithms: {e}")

    try:
        log_filename = f"gdelt_log_{time.strftime('%Y%m%d-%H%M%S')}.txt"
        tee = Tee(log_filename)
        
        # --- Custom Data Loading Script ---
        path = osp.join('.', 'data', 'GDELT') # PyG will download and process the data here
    
        print("Loading GDELTDataset dataset...")
        # dataset = GDELTDataset(path)
        dataset = GDELTDataset(root='./data/')

        data = dataset[0] 
    
    
        # Use the dataset's built-in, robust method for splitting
     
        train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
        
        loader_kwargs = dict(num_workers=2, pin_memory=True, persistent_workers=True, 
                             prefetch_factor=4, worker_init_fn=seed_worker)

        val_loader = TemporalDataLoader(val_data, batch_size=200, neg_sampling_ratio=1.0, **loader_kwargs)
        test_loader = TemporalDataLoader(test_data, batch_size=200, neg_sampling_ratio=1.0, **loader_kwargs)
    
        num_nodes_to_track = 15
        tracked_nodes = np.random.choice(data.num_nodes, num_nodes_to_track, replace=False)
        print(f"\nTracking {num_nodes_to_track} nodes for visualization: {tracked_nodes.tolist()}")

        all_results = []
        for seed in seeds:
            result, history = run_single_experiment(seed, data, train_data, val_loader, test_loader, device, loader_kwargs)
            all_results.append(result)

            tracked_history = []
            for epoch, embeddings in history:
                tracked_embeddings = embeddings[tracked_nodes]
                tracked_history.append((epoch, tracked_embeddings))
            
            plot_tsne_node_embeddings(tracked_history, tracked_nodes, seed=seed)

        print(f"\n\n{'='*30} FINAL EVALUATION COMPLETE {'='*30}")
        
        test_aps, test_aucs, test_mrrs = [r.get('ap', 0) for r in all_results], [r.get('auc', 0) for r in all_results], [r.get('mrr', 0) for r in all_results]
        epoch_times, peak_mems = [r.get('avg_epoch_time', 0) for r in all_results], [r.get('peak_memory_mb', 0) for r in all_results]
        mean_ap, std_ap = np.mean(test_aps), np.std(test_aps)
        mean_auc, std_auc = np.mean(test_aucs), np.std(test_aucs)
        mean_mrr, std_mrr = np.mean(test_mrrs), np.std(test_mrrs)
        mean_time, std_time = np.mean(epoch_times), np.std(epoch_times)
        mean_mem, std_mem = np.mean(peak_mems), np.std(peak_mems)

        print("\n--- Final Performance Results (Mean ± Std Dev over 5 Runs) ---")
        print(f"Test AP:            {mean_ap:.4f} ± {std_ap:.4f}")
        print(f"Test AUC:           {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Test MRR:           {mean_mrr:.4f} ± {std_mrr:.4f}")
        print("-" * 60)
        print("\n--- Final Efficiency Results (Mean ± Std Dev over 5 Runs) ---")
        print(f"Avg. Runtime/Epoch: {mean_time:.2f}s ± {std_time:.2f}s")
        if device.type == 'cuda' and peak_mems[0] > 0:
            print(f"Peak Memory Usage:  {mean_mem:.2f} MB ± {std_mem:.2f} MB")
        else:
            print("Peak Memory Usage:  N/A (Not a CUDA device)")
        print("-" * 60)

    finally:
        if 'tee' in locals():
            sys.stdout = tee.stdout
            tee.file.close()