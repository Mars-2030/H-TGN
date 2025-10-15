import os
import os.path as osp
from typing import Callable, Optional
import torch
import torch_geometric

from torch_geometric.data import TemporalData
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import plotly.graph_objects as go
from openTSNE import TSNE as OpenTSNE

import networkx as nx  
from networkx.algorithms import community as nx_community  


import numpy as np
import random
import time
import json
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear, LayerNorm
import torch.nn.functional as F

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv, GCNConv
from torch_geometric.nn.dense import dense_diff_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader
from torch_geometric.datasets import JODIEDataset

from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

from torch_geometric.data import InMemoryDataset, TemporalData, download_url
from typing import Callable, Optional


try:
    from visualization import (
    plot_sankey_evolution,
    plot_hierarchical_snapshots,
    plot_utilization_heatmap,
    plot_assignment_switch_rate,
    plot_country_trajectories,
    generate_hierarchical_analysis_report,
    plot_alluvial_evolution,
    plot_coarsening_comparison,
    plot_animated_trajectories,
    plot_top_k_alluvial,
    plot_cluster_lineage_alluvial,
    plot_event_highlight_alluvial,
    plot_simplified_alluvial
    )
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    print("Warning: visualization_module.py not found. Visualizations will be skipped.")
    VISUALIZATIONS_AVAILABLE = False




class ICEWS18Dataset(InMemoryDataset):
    r"""The temporal knowledge graph dataset from the ICEWS18 benchmark dataset,
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

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #entities
          - #relations
          - #events
        * - ICEWS18
          - varies
          - varies
          - varies
    """
    url = 'https://github.com/INK-USC/RE-Net/raw/master/data/ICEWS18'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=TemporalData)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'ICEWS18', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'ICEWS18', 'processed')

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
            df = pd.read_csv(file_path, sep='\t', header=None, usecols=[0, 1, 2, 3],
                            names=['src', 'relation', 'dst', 'timestamp'])

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
        #relation = torch.tensor(full_df['relation'].values, dtype=torch.float).tolist()
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
        return 'ICEWS18()'


# ==================== CLASS for printing output to file ============================
import sys

class Tee(object):
    """A helper class to redirect print statements to both a file and the console."""
    def __init__(self, filename, mode='w'):
        self.file = open(filename, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        if sys.stdout is self:
            sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

# =====================================================================================
# 2. Setup and Configuration
# =====================================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # --- For full determinism (at a performance cost) ---
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


# =====================================================================================
# 3. Helper Functions and Model Components
# =====================================================================================
def _intersect1d(prev_ids: torch.Tensor, curr_ids: torch.Tensor):
    if hasattr(torch, "intersect1d"):
        return torch.intersect1d(prev_ids, curr_ids, return_indices=True)
    prev_np = prev_ids.detach().cpu().numpy()
    curr_np = curr_ids.detach().cpu().numpy()
    common_np, prev_idx_np, curr_idx_np = np.intersect1d(prev_np, curr_np, assume_unique=False, return_indices=True)
    common = torch.as_tensor(common_np, device=prev_ids.device, dtype=prev_ids.dtype)
    prev_idx = torch.as_tensor(prev_idx_np, device=prev_ids.device, dtype=torch.long)
    curr_idx = torch.as_tensor(curr_idx_np, device=prev_ids.device, dtype=torch.long)
    return common, prev_idx, curr_idx

# --- REPLACE TemporalClusteringGRU ---
class TemporalClusteringGRU(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        # The GRU's input size should match the features it will process
        self.gru_cell = torch.nn.GRUCell(in_channels, hidden_channels)
        self.linear_logits = torch.nn.Linear(hidden_channels, num_clusters)
        self.register_buffer("hidden_state", torch.zeros(1, hidden_channels))
        self.hidden_channels = hidden_channels

    def reset_state(self, num_nodes=None, device='cpu'):
        if num_nodes is not None:
            self.hidden_state = torch.zeros(num_nodes, self.hidden_channels, device=device)
        else:
            self.hidden_state.fill_(0)

    def forward(self, features, node_ids): # Use generic 'features'
        if self.hidden_state.size(0) < node_ids.max() + 1:
             new_state = torch.zeros(node_ids.max() + 1, self.hidden_channels, device=features.device)
             new_state[:self.hidden_state.size(0)] = self.hidden_state
             self.hidden_state = new_state
        
        prev_h_state = self.hidden_state[node_ids]
        new_h_state = self.gru_cell(features, prev_h_state) # Use generic 'features'
        S_logits = self.linear_logits(new_h_state)
        self.hidden_state[node_ids] = new_h_state.detach()
        return S_logits


class HierarchicalTemporalPooler(torch.nn.Module):
    def __init__(self, num_nodes, ratios, in_channels, hidden_channels):
        super().__init__()
        
        self.levels = torch.nn.ModuleList()
        self.num_nodes_per_level = [num_nodes]
        current_num_nodes = num_nodes
        
        # The input feature dimension is the SAME for all levels, it is `in_channels`
        # The GRU's hidden dimension is also constant.
        for ratio in ratios:
            num_clusters = max(1, int(current_num_nodes * ratio))
            cluster_module = TemporalClusteringGRU(in_channels, hidden_channels, num_clusters)
            self.levels.append(cluster_module)
            
            # Add the number of clusters for the new level to our list.
            self.num_nodes_per_level.append(num_clusters)

            current_num_nodes = num_clusters




    def reset_states(self, device='cpu'):
        # This part of your code was incorrect, this is the robust way to do it.
        # We need to reset the state for the GRU memory of the original nodes,
        # and then for the GRU memory of the clusters at each level.
        for i, level in enumerate(self.levels):
            # The number of "nodes" for the GRU's memory is the size of its input set.
            num_input_nodes_for_level = self.num_nodes_per_level[i]
            level.reset_state(num_input_nodes_for_level, device)

    def forward(self, z_in, node_ids, edge_index):
        # The rest of the forward pass is correct.
        assignment_matrices = []
        static_losses = []
        
        current_features = z_in
        current_node_ids = node_ids
        current_edge_index = edge_index

        for i, level in enumerate(self.levels):
            S_logits = level(current_features, current_node_ids)
            S_soft = S_logits.softmax(dim=-1)
            assignment_matrices.append(S_soft)

            batch_vec = torch.zeros(current_features.size(0), dtype=torch.long, device=z_in.device)
            x_dense, mask_x = to_dense_batch(current_features, batch_vec)
            adj = to_dense_adj(current_edge_index, batch_vec)
            S_dense, _ = to_dense_batch(S_soft, batch_vec)
            
            pooled_features, _, link_loss, ent_loss = dense_diff_pool(x_dense, adj, S_dense, mask_x)
            static_losses.append(link_loss + ent_loss)

            current_features = pooled_features.squeeze(0)
            current_node_ids = torch.arange(current_features.size(0), device=z_in.device)
            adj_dense = to_dense_adj(current_edge_index, batch_vec).squeeze(0)
            new_adj = S_soft.t() @ adj_dense @ S_soft
            current_edge_index = new_adj.to_sparse().indices()
        
        z_final_pooled_features = current_features
        
        S_final = assignment_matrices[0]
        for S_matrix in assignment_matrices[1:]:
            S_final = S_final @ S_matrix
            
        z_reconstructed = S_final @ z_final_pooled_features
        total_static_loss = sum(static_losses)

        return z_reconstructed, total_static_loss, assignment_matrices

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim)
    
    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)
    def forward(self, z_src, z_dst):
        h = (self.lin_src(z_src) * self.lin_dst(z_dst)).relu()
        return self.lin_final(h)


def calculate_epoch_modularity(snapshot, data_all):
    """
    Calculates the modularity of the graph partition for a single epoch snapshot.
    This is the efficient way to calculate modularity, done once per epoch.
    """
    if 'assignments_l1' not in snapshot:
        return 0.0

    node_ids_in_snapshot = set(snapshot['node_ids'].numpy())
    assignments = snapshot['assignments_l1'].numpy()
    
    if np.unique(assignments).size <= 1: # Modularity is 0 if there's only one cluster
        return 0.0

    # Build a NetworkX graph containing only the nodes and edges present in this snapshot
    G = nx.Graph()
    G.add_nodes_from(node_ids_in_snapshot)

    src, dst = data_all.src.numpy(), data_all.dst.numpy()
    for i in range(len(src)):
        if src[i] in node_ids_in_snapshot and dst[i] in node_ids_in_snapshot:
            G.add_edge(src[i], dst[i])
    
    # Create the community list format required by NetworkX
    communities_dict = {}
    for node_id, cluster_id in zip(snapshot['node_ids'].numpy(), assignments):
        if cluster_id not in communities_dict:
            communities_dict[cluster_id] = set()
        communities_dict[cluster_id].add(node_id)
    
    communities = list(communities_dict.values())
    
    try:
        modularity_score = nx_community.modularity(G, communities)
    except Exception: # Handle cases with no edges, etc.
        modularity_score = 0.0
        
    return modularity_score
# =====================================================================================
# 5. Training and Evaluation Functions
# =====================================================================================
def process_batch(batch, memory, gnn, hierarchical_pooler, residual_projection, final_norm, neighbor_loader, assoc, data, device):
    n_id, edge_index, e_id = neighbor_loader(batch.n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)
    z_mem, last_update = memory(n_id)
    
    # 1. Get the powerful TGN embeddings as the base for everything.
    z_tgn = gnn(z_mem, last_update, edge_index, data.t[e_id.cpu()].to(device), data.msg[e_id.cpu()].to(device))
    
    # 2. Pass the base embeddings through the entire hierarchical pooling pipeline.
    #    This returns the final reconstructed node embeddings after pooling and the combined static loss from all levels.
    z_pooled_nodes, static_loss, assignment_matrices = hierarchical_pooler(z_tgn, n_id, edge_index)
    
    # 3. Use the residual connection with the final reconstructed embeddings.
    #    This is crucial for performance, as it allows the model to use both the raw and the pooled information.
    z_final = final_norm(F.relu(z_tgn + residual_projection(z_pooled_nodes)))
    
    # --- For analysis and logging ---
    with torch.no_grad():
        # We can analyze the entropy and utilization of the first level of clustering (nodes -> clusters)
        s_soft_l1 = assignment_matrices[0]
        s_ent_level1 = -(s_soft_l1 * s_soft_l1.clamp_min(1e-9).log()).sum(-1).mean()
        avg_cluster_util_level1 = s_soft_l1.mean(dim=0)

    # Return everything needed by train_epoch and the new evaluate function
    return z_final, z_pooled_nodes, static_loss, s_ent_level1, avg_cluster_util_level1, assignment_matrices, n_id, z_tgn


def train_epoch(loader, train_data, memory, gnn, hierarchical_pooler, residual_projection, final_norm, link_pred, optimizer, neighbor_loader, assoc, data_all, criterion, params, device):
    # Set all modules used in training to train mode
    memory.train(); gnn.train(); hierarchical_pooler.train(); residual_projection.train(); final_norm.train(); link_pred.train()
    
    # Reset all temporal states at the start of a new epoch
    memory.reset_state()
    neighbor_loader.reset_state()
    hierarchical_pooler.reset_states(device)
    
    # Initialize accumulators for logging
    total_loss, total_static_loss, total_entropy = 0.0, 0.0, 0.0

    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device, non_blocking=True)
        
        z_final, _, static_loss, s_ent, _, _, _, _ = process_batch(
            batch, memory, gnn, hierarchical_pooler, residual_projection, final_norm, 
            neighbor_loader, assoc, data_all, device
        )
        
        # Link prediction on the final node embeddings
        pos_out = link_pred(z_final[assoc[batch.src]], z_final[assoc[batch.dst]])
        neg_out = link_pred(z_final[assoc[batch.src]], z_final[assoc[batch.neg_dst]])
        
        link_loss = criterion(pos_out, torch.ones_like(pos_out)) + criterion(neg_out, torch.zeros_like(neg_out))
        
        # The new, simpler loss function. Temporal consistency is now part of the architecture, not the loss.
        total_loss_batch = link_loss + params['static_loss_weight'] * static_loss
        
        total_loss_batch.backward()
        
        # Clip gradients for all trainable parameters
        all_params = [p for m in [memory, gnn, hierarchical_pooler, residual_projection, final_norm, link_pred] for p in m.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        
        optimizer.step()
        
        # Update model state for the next batch
        memory.update_state(batch.src, batch.dst, batch.t.to(memory.last_update.dtype), batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
        memory.detach()
        
        # Accumulate metrics for epoch-level logging
        total_loss += float(total_loss_batch) * batch.num_events
        total_static_loss += float(static_loss) * batch.num_events
        total_entropy += float(s_ent) * batch.num_events

    num_events = train_data.num_events
    # Return the new, simplified set of metrics
    return (total_loss / num_events, total_static_loss / num_events, total_entropy / num_events)




@torch.no_grad()
def evaluate(loader, memory, gnn, hierarchical_pooler, residual_projection, final_norm, link_pred, neighbor_loader, assoc, data, criterion, params, device,
             capture_epoch_data=False, epoch_snapshots=None, current_epoch=None):
    """
    Evaluates the model on a given data loader.
    Calculates link prediction performance (AP, AUC, MRR) and reconstruction error.
    Modularity is now calculated separately and more efficiently outside this function.
    """
    memory.eval(); gnn.eval(); hierarchical_pooler.eval(); residual_projection.eval(); final_norm.eval(); link_pred.eval()
    
    aps, aucs, mrrs = [], [], []
    
    # Accumulators for metrics calculated in this function
    total_loss, total_static_loss, total_entropy, total_events = 0.0, 0.0, 0.0, 0
    total_recon_error, num_batches_val = 0.0, 0

    # Data structures for visualization snapshots
    if capture_epoch_data:
        epoch_node_ids, epoch_embeddings_l0, epoch_embeddings_final = [], [], []
        epoch_assignments_l1, epoch_assignments_l2 = [], []
        batch_avg_utilizations_l1 = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        
        # Get embeddings and run through the hierarchical model
        z_final, z_pooled_nodes, static_loss, s_ent_l1, avg_util_l1, assignment_matrices, n_id, z_tgn = process_batch(
            batch, memory, gnn, hierarchical_pooler, residual_projection, final_norm, 
            neighbor_loader, assoc, data, device
        )
        
        # --- Advanced Data Capture Logic for Visualizations ---
        if capture_epoch_data:
            epoch_node_ids.append(n_id.cpu())
            epoch_embeddings_l0.append(z_tgn.cpu())
            epoch_embeddings_final.append(z_final[assoc[n_id]].cpu())
            batch_avg_utilizations_l1.append(avg_util_l1.cpu())
            
            if len(assignment_matrices) > 0:
                epoch_assignments_l1.append(assignment_matrices[0].argmax(dim=-1).cpu())
            if len(assignment_matrices) > 1:
                s_soft_l2_for_nodes = assignment_matrices[0] @ assignment_matrices[1]
                epoch_assignments_l2.append(s_soft_l2_for_nodes.argmax(dim=-1).cpu())

        # --- Performance and Loss Calculation ---
        pos_out = link_pred(z_final[assoc[batch.src]], z_final[assoc[batch.dst]])
        neg_out = link_pred(z_final[assoc[batch.src]], z_final[assoc[batch.neg_dst]])
        
        link_loss = criterion(pos_out, torch.ones_like(pos_out)) + criterion(neg_out, torch.zeros_like(neg_out))
        loss = link_loss + params['static_loss_weight'] * static_loss
        
        # Update total counters
        total_loss += float(loss) * batch.num_events
        total_static_loss += float(static_loss) * batch.num_events
        total_entropy += float(s_ent_l1) * batch.num_events
        total_events += batch.num_events
        num_batches_val += 1
        
        # --- Reconstruction Error Calculation (Efficient) ---
        recon_error = F.mse_loss(z_pooled_nodes, z_tgn)
        total_recon_error += float(recon_error)
        
        # --- Link Prediction Metric Calculation ---
        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)
        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))
        
        scores = torch.cat([pos_out, neg_out], dim=1).cpu()
        ranks = (scores.sort(dim=1, descending=True)[1] == 0).nonzero(as_tuple=False)[:, 1] + 1
        mrrs.append((1.0 / ranks.float()).mean().item())

        # Update model state for the next batch in the sequence
        memory.update_state(batch.src, batch.dst, batch.t.to(memory.last_update.dtype), batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

    # --- Consolidate and Save Snapshot Data ---
    if capture_epoch_data and epoch_node_ids:
        all_node_ids = torch.cat(epoch_node_ids)
        unique_ids, first_indices = np.unique(all_node_ids.numpy(), return_index=True)
        
        snapshot = { 
            'node_ids': torch.from_numpy(unique_ids), 
            'embeddings_l0': torch.cat(epoch_embeddings_l0)[first_indices],
            'embeddings_final': torch.cat(epoch_embeddings_final)[first_indices]
        }
        if epoch_assignments_l1:
            snapshot['assignments_l1'] = torch.cat(epoch_assignments_l1)[first_indices]
        if epoch_assignments_l2:
            snapshot['assignments_l2'] = torch.cat(epoch_assignments_l2)[first_indices]
        if batch_avg_utilizations_l1:
            avg_util_epoch = torch.stack(batch_avg_utilizations_l1).mean(dim=0)
            snapshot['avg_utilization'] = avg_util_epoch

        epoch_snapshots[current_epoch] = snapshot

    # --- Prepare Final Metrics for Return ---
    detailed_stats = {
        "loss": total_loss / (total_events or 1),
        "s_static": total_static_loss / (total_events or 1),
        "s_ent": total_entropy / (total_events or 1),
        "recon_error": total_recon_error / (num_batches_val or 1),
    }

    return detailed_stats, float(np.mean(aps)), float(np.mean(aucs)), float(np.mean(mrrs))
    
# =====================================================================================
# 6. RUNNER
# =====================================================================================

def run_trial(params, trial_num, trial_seed, data_all, train_data, val_data, test_data, num_nodes, device, capture_visuals=True):
    """
    Runs a full training and evaluation trial for the model.
    - Initializes the model, optimizer, and data loaders.
    - Loops through epochs, training the model.
    - Evaluates performance on validation and test sets.
    - Implements early stopping based on validation performance.
    - Captures data snapshots for visualization and analysis.
    """
    print(f"\n{'='*30} TRIAL {trial_num} {'='*30}")
    print(f"Parameters: {params} | Seed: {trial_seed}")
    print(f"{'='*70}")
    set_seed(trial_seed)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # --- Data Loader Setup ---
    loader_kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4, worker_init_fn=seed_worker)
    train_loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=1.0, **loader_kwargs)
    val_loader = TemporalDataLoader(val_data, batch_size=200, neg_sampling_ratio=1.0, **loader_kwargs)
    test_loader = TemporalDataLoader(test_data, batch_size=200, neg_sampling_ratio=1.0, **loader_kwargs)
    neighbor_loader = LastNeighborLoader(num_nodes, size=20, device=device)

    # --- Model Initialization ---
    memory_dim, time_dim, embedding_dim, hidden_dim = 100, 100, 100, 64
    memory = TGNMemory(num_nodes, data_all.msg.size(-1), memory_dim, time_dim, message_module=IdentityMessage(data_all.msg.size(-1), memory_dim, time_dim), aggregator_module=LastAggregator()).to(device)
    gnn = GraphAttentionEmbedding(memory_dim, embedding_dim, data_all.msg.size(-1), memory.time_enc).to(device)
    
    hierarchical_pooler = HierarchicalTemporalPooler(
        num_nodes=num_nodes,
        ratios=[0.50], 
        in_channels=embedding_dim,
        hidden_channels=hidden_dim
    ).to(device)
    
    residual_projection = Linear(embedding_dim, embedding_dim).to(device)
    final_norm = LayerNorm(embedding_dim).to(device)
    link_pred = LinkPredictor(embedding_dim).to(device)
    
    modules = [memory, gnn, hierarchical_pooler, residual_projection, final_norm, link_pred]
    
    # --- Optimizer and Criterion Setup ---
    seen_params, all_params = set(), []
    for module in modules:
        for param in module.parameters():
            if param.requires_grad and id(param) not in seen_params:
                seen_params.add(id(param)); all_params.append(param)
    
    optimizer = torch.optim.Adam(all_params, lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = torch.nn.BCEWithLogitsLoss()
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    # --- Training Loop Setup ---
    MIN_EPOCHS_FOR_STOPPING, EMA_ALPHA = 15, 0.2
    best_val_ap, best_ema_val_ap, patience_counter = 0.0, 0.0, 0
    best_test_metrics = {}
    
    capture_epochs = {0, 1, 2, 3,4, 5, 10, 20, 30,35, 40,45, 50}
    epoch_snapshots, epoch_times = {}, []

    # --- Initial State Capture (Before Training) ---
    if capture_visuals:
        print("--- Capturing initial state (Epoch 0) before training... ---")
        evaluate(val_loader, memory, gnn, hierarchical_pooler, residual_projection, final_norm, link_pred, neighbor_loader, assoc, data_all, criterion, params, device,
                 capture_epoch_data=True, epoch_snapshots=epoch_snapshots, current_epoch=0)
        print("--- Initial state captured. Starting training... ---")

    # --- Main Training Loop ---
    for epoch in range(1, 5):
        start_time = time.time()
        
        # Train one epoch
        train_loss, s_l, s_ent = train_epoch(
            train_loader, train_data, memory, gnn, hierarchical_pooler, residual_projection, final_norm, link_pred, optimizer,
            neighbor_loader, assoc, data_all, criterion, params, device
        )
        
        should_capture_data = capture_visuals and epoch in capture_epochs
        
        # Evaluate on validation set
        val_stats, val_ap, val_auc, val_mrr = evaluate(
            val_loader, memory, gnn, hierarchical_pooler, residual_projection, final_norm, link_pred,
            neighbor_loader, assoc, data_all, criterion, params, device,
            capture_epoch_data=should_capture_data, epoch_snapshots=epoch_snapshots, current_epoch=epoch
        )
        
        # EFFICIENTLY calculate modularity once for the epoch, if a snapshot was taken
        if should_capture_data:
            val_stats['modularity'] = calculate_epoch_modularity(epoch_snapshots[epoch], data_all)
        else:
            val_stats['modularity'] = 0.0 # Default value if not calculated for this epoch

        scheduler.step(val_stats['loss'])
        epoch_times.append(time.time() - start_time)
        
        if epoch == 1: ema_val_ap = val_ap
        else: ema_val_ap = EMA_ALPHA * val_ap + (1 - EMA_ALPHA) * ema_val_ap

        # --- Logging Block ---
        print(f"Epoch :{epoch:02d} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {time.time() - start_time:.2f}s")
        print(f"  Train Stats: Loss={train_loss:.4f} | S_Static={s_l:.4f} | S_Ent(L1)={s_ent:.4f}")
        print(f"  Val   Stats: Loss={val_stats['loss']:.4f} | AP={val_ap:.4f} (EMA AP={ema_val_ap:.4f}) | AUC={val_auc:.4f} | MRR={val_mrr:.4f}")
        print(f"    Coarsening: Modularity={val_stats['modularity']:.4f} | Recon Error={val_stats['recon_error']:.4f}")

        # --- Early Stopping and Best Model Logic ---
        if ema_val_ap > best_ema_val_ap:
            best_ema_val_ap, best_val_ap, patience_counter = ema_val_ap, val_ap, 0
            
            # Evaluate on the test set since we found a new best model
            test_stats, test_ap, test_auc, test_mrr = evaluate(
                test_loader, memory, gnn, hierarchical_pooler, residual_projection, final_norm, link_pred,
                neighbor_loader, assoc, data_all, criterion, params, device
            )
            
            # Calculate modularity for the best model using the snapshot from this epoch
            if should_capture_data:
                test_stats['modularity'] = calculate_epoch_modularity(epoch_snapshots[epoch], data_all)
            else:
                test_stats['modularity'] = -1.0 # Indicate it wasn't captured

            # Store all test metrics
            best_test_metrics = {**test_stats, "ap": test_ap, "auc": test_auc, "mrr": test_mrr}
            
            print(f"--- New best model found (EMA AP: {ema_val_ap:.4f})! Evaluating on test set... ---")
            print(f"  Test  Stats: Loss={best_test_metrics['loss']:.4f} | AP={best_test_metrics['ap']:.4f} | AUC={best_test_metrics['auc']:.4f} | MRR={best_test_metrics['mrr']:.4f}")
            print(f"    Coarsening: Modularity={best_test_metrics['modularity']:.4f} | Recon Error={best_test_metrics['recon_error']:.4f}")
            
            torch.save({module.__class__.__name__: module.state_dict() for module in modules}, f'saved_models/best_model_icews18_trial_{trial_num}.pth')
        else:
            if epoch > MIN_EPOCHS_FOR_STOPPING: 
                patience_counter += 1
        
        if patience_counter >= 15:
            print(f"Early stopping triggered after {patience_counter} epochs."); break
            
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
    best_test_metrics['avg_epoch_time'] = avg_epoch_time
    
    results = {"params": params, "best_val_ap": best_val_ap, "best_test_metrics": best_test_metrics}
    
    return results, hierarchical_pooler.num_nodes_per_level, epoch_snapshots
# =====================================================================================
# 7. MAIN EXECUTION
# =====================================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # best_params = { 'static_loss_weight': 0.4, 'temporal_loss_weight': 2.0, 'lr': 0.0002 }
    # --- CHANGE 1: Remove the obsolete temporal_loss_weight ---
    best_params = { 'static_loss_weight': 0.4, 'lr': 0.0002 }
    seeds = [123451]#, 123452, 123453, 123454, 123455]
    os.makedirs('saved_models', exist_ok=True)
    
    log_filename = f"icews18_linear_best_seed_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    tee = Tee(log_filename)
    print(f"Using device: {device}")
    
    print("Loading ICEWS18 dataset...")
    dataset = ICEWS18Dataset(root='../data/')
    data_all = dataset[0]
    train_data, val_data, test_data = data_all.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    num_nodes = data_all.num_nodes
    print(f"Dataset ICEWS18 loaded. #Nodes={num_nodes}, #Events={data_all.num_events}")
        
    
    all_results = []
    for i, seed in enumerate(seeds):
        # This part is correct
        result, num_nodes_per_level, epoch_snapshots = run_trial(
            params=best_params, trial_num=i+1, trial_seed=seed, 
            data_all=data_all, train_data=train_data, val_data=val_data, test_data=test_data, 
            num_nodes=num_nodes, device=device, capture_visuals=True
        )
        all_results.append(result['best_test_metrics'])
        
        print(f"\n--- Finished Seed {seed} -> Best Test Metrics: {result['best_test_metrics']} ---")
        
        # --- REPLACE THE OLD VISUALIZATION BLOCK WITH THIS NEW ONE ---
        print(f"\n--- Generating visualizations for seed {seed} ---")

        print("Loading entity mapping...")
        entity_mapping = {}
        try:
            mapping_file_path = os.path.join('.', 'entity2id.txt') 
            with open(mapping_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        name, entity_id = parts[0], int(parts[1])
                        entity_mapping[entity_id] = name
            
            # Move the success message INSIDE the 'try' block
            print(f"Loaded {len(entity_mapping)} entity names.") 

        except FileNotFoundError:
            print("Warning: Entity mapping file not found. Geopolitical plots will be skipped.")
            entity_mapping = None # This part is correct

        if epoch_snapshots and num_nodes_per_level:
            num_clusters_l1 = num_nodes_per_level[1]
            
            trained_snapshots_exist = any(epoch > 0 for epoch in epoch_snapshots.keys())

            if trained_snapshots_exist:
                # All these function calls should now use 'epoch_snapshots'
                plot_hierarchical_snapshots(epoch_snapshots, seed=seed)
                plot_sankey_evolution(epoch_snapshots, num_clusters_l1, seed=seed)
                plot_utilization_heatmap(epoch_snapshots, num_clusters_l1, seed=seed)
                plot_assignment_switch_rate(epoch_snapshots, seed=seed)


                final_epoch = max([e for e in epoch_snapshots.keys() if e > 0])
                final_snapshot = epoch_snapshots[final_epoch]
                plot_coarsening_comparison(final_snapshot, data_all, seed=seed)
                plot_alluvial_evolution(epoch_snapshots, seed=seed)
                
                plot_simplified_alluvial(epoch_snapshots, final_top_k=8, intermediate_top_k=10, seed=seed)
                
                plot_cluster_lineage_alluvial(epoch_snapshots, final_cluster_id=978, seed=seed)



                
                # final_epoch = max(epoch_snapshots.keys())
                tsne_layout = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=seed, init='pca', learning_rate='auto')
                stable_layout = tsne_layout.fit_transform(final_snapshot['embeddings_final'].numpy())
                node_to_pos = {node_id.item(): pos for node_id, pos in zip(final_snapshot['node_ids'], stable_layout)}
                if entity_mapping: # Add a check here
                    plot_country_trajectories(epoch_snapshots, node_to_pos, entity_mapping, seed=seed, countries_to_track=["United States", "China", "Russia", "India", "Iran", "United Kingdom"])
                    generate_hierarchical_analysis_report(final_snapshot, data_all, entity_mapping, seed=seed)
                    try:
                        plot_animated_trajectories(epoch_snapshots, entity_mapping, seed=seed)
                    except Exception as e:
                        print(f"Warning: Animated trajectories plot failed: {e}")

            else:
                print("Skipping plots as no trained epochs were captured.")
        else:
            print(f"Skipping visualizations for seed {seed} as no data was captured.")

    print(f"\n\n{'='*30} FINAL EVALUATION COMPLETE {'='*30}")
    print(f"Ran on {len(seeds)} random seeds with parameters: {best_params}")
    if all_results:
        test_aps = [r.get('ap', 0) for r in all_results]
        test_aucs = [r.get('auc', 0) for r in all_results]
        test_mrrs = [r.get('mrr', 0) for r in all_results]
        epoch_times = [r.get('avg_epoch_time', 0) for r in all_results]
        test_mods = [r.get('modularity', 0) for r in all_results]
        test_recons = [r.get('recon_error', 0) for r in all_results]
        
        print("\n--- Final Performance Results (Mean ± Std Dev over 5 Runs) ---")
        print(f"Test AP:            {np.mean(test_aps):.4f} ± {np.std(test_aps):.4f}")
        print(f"Test AUC:           {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        print(f"Test MRR:           {np.mean(test_mrrs):.4f} ± {np.std(test_mrrs):.4f}")
        print("-" * 60)

        print("\n--- Final Coarsening Effectiveness (Mean ± Std Dev over runs) ---")
        print(f"Modularity:         {np.mean(test_mods):.4f} ± {np.std(test_mods):.4f}")
        print(f"Reconstruction Err: {np.mean(test_recons):.4f} ± {np.std(test_recons):.4f}")
        print("-" * 60)
        
        print("\n--- Final Efficiency Results (Mean ± Std Dev over 5 Runs) ---")
        print(f"Avg. Runtime/Epoch: {np.mean(epoch_times):.2f}s ± {np.std(epoch_times):.2f}s")
        print("-" * 60)

    if 'tee' in locals():
        sys.stdout = tee.stdout
        del tee