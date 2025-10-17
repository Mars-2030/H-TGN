# tpnet.py
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

import os
import os.path as osp
import random
import sys
import time
from types import SimpleNamespace as NS
from typing import Optional

# Optional for debugging: make CUDA errors synchronous
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.datasets import JODIEDataset


class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(time_dim))
        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        timestamps = timestamps.unsqueeze(dim=2)  # (B, L, 1)
        return torch.cos(self.w(timestamps))      # (B, L, D_t)


class NeighborSampler:
    def __init__(self, adj_list: list, sample_neighbor_strategy: str = 'uniform',
                 time_scaling_factor: float = 0.0, seed: int = None):
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        for _, per_node_neighbors in enumerate(adj_list):
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors], dtype=np.int64))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors], dtype=np.int64))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors], dtype=np.float64))

            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(
                    self.compute_sampled_probabilities(self.nodes_neighbor_times[-1])
                )

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        if len(node_neighbor_times) == 0:
            return np.array([])
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)
        if return_sampled_probabilities:
            return (
                self.nodes_neighbor_ids[node_id][:i],
                self.nodes_edge_ids[node_id][:i],
                self.nodes_neighbor_times[node_id][:i],
                self.nodes_neighbor_sampled_probabilities[node_id][:i],
            )
        else:
            return (
                self.nodes_neighbor_ids[node_id][:i],
                self.nodes_edge_ids[node_id][:i],
                self.nodes_neighbor_times[node_id][:i],
                None,
            )

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'
        B = len(node_ids)
        nodes_neighbor_ids = np.zeros((B, num_neighbors), dtype=np.int64)
        nodes_edge_ids = np.zeros((B, num_neighbors), dtype=np.int64)
        nodes_neighbor_times = np.zeros((B, num_neighbors), dtype=np.float64)

        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(
                    node_id=node_id, interact_time=node_interact_time,
                    return_sampled_probabilities=(self.sample_neighbor_strategy == 'time_interval_aware')
                )

            if len(node_neighbor_ids) > 0:
                if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                    if node_neighbor_sampled_probabilities is not None:
                        node_neighbor_sampled_probabilities = torch.softmax(
                            torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0
                        ).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors,
                                                           p=node_neighbor_sampled_probabilities)
                    else:
                        sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors,
                                                                   p=node_neighbor_sampled_probabilities)

                    nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                    nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                    nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                    sorted_position = nodes_neighbor_times[idx, :].argsort()
                    nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                    nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                    nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]

                elif self.sample_neighbor_strategy == 'recent':
                    node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                    node_edge_ids = node_edge_ids[-num_neighbors:]
                    node_neighbor_times = node_neighbor_times[-num_neighbors:]
                    nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                    nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                    nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                else:
                    raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')

        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray,
                                node_interact_times: np.ndarray, num_neighbors: int = 20):
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(
            node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors
        )
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]

        for _ in range(1, num_hops):
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(
                node_ids=nodes_neighbor_ids_list[-1].flatten(),
                node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                num_neighbors=num_neighbors
            )
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        for node_id, node_interact_time in zip(node_ids, node_interact_times):
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(
                node_id=node_id, interact_time=node_interact_time, return_sampled_probabilities=False
            )
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(data, sample_neighbor_strategy: str = 'uniform',
                         time_scaling_factor: float = 0.0, seed: int = None):
    """
    Build NeighborSampler from any object that has:
      data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times
    (each 1-D numpy array / list with equal length)
    """
    max_node_id = int(max(np.max(data.src_node_ids), np.max(data.dst_node_ids)))
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(
            data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times):
        adj_list[int(src_node_id)].append((int(dst_node_id), int(edge_id), float(node_interact_time)))
        adj_list[int(dst_node_id)].append((int(src_node_id), int(edge_id), float(node_interact_time)))
    return NeighborSampler(adj_list=adj_list, sample_neighbor_strategy=sample_neighbor_strategy,
                           time_scaling_factor=time_scaling_factor, seed=seed)


# -----------------------------------------------------------------------------
# RandomProjectionModule (architecture unchanged)
# -----------------------------------------------------------------------------
class RandomProjectionModule(nn.Module):
    def __init__(self, node_num: int, edge_num: int, dim_factor: int, num_layer: int,
                 time_decay_weight: float, device: str, use_matrix: bool, beginning_time: np.float64,
                 not_scale: bool, enforce_dim: int):
        super(RandomProjectionModule, self).__init__()
        self.node_num = node_num
        self.edge_num = edge_num
        if enforce_dim != -1:
            self.dim = enforce_dim
        else:
            self.dim = min(int(math.log(self.edge_num * 2)) * dim_factor, node_num)
        self.num_layer = num_layer
        self.time_decay_weight = time_decay_weight
        self.begging_time = nn.Parameter(torch.tensor(beginning_time), requires_grad=False)  # keep original name
        self.now_time = nn.Parameter(torch.tensor(beginning_time), requires_grad=False)
        self.device = device
        self.random_projections = nn.ParameterList()
        self.use_matrix = use_matrix
        self.node_feature_dim = 128
        self.not_scale = not_scale

        if self.use_matrix:
            self.dim = self.node_num
            for i in range(self.num_layer + 1):
                if i == 0:
                    self.random_projections.append(nn.Parameter(torch.eye(self.node_num), requires_grad=False))
                else:
                    self.random_projections.append(
                        nn.Parameter(torch.zeros_like(self.random_projections[i - 1]), requires_grad=False)
                    )
        else:
            for i in range(self.num_layer + 1):
                if i == 0:
                    self.random_projections.append(
                        nn.Parameter(
                            torch.normal(0, 1 / math.sqrt(self.dim), (self.node_num, self.dim)),
                            requires_grad=False,
                        )
                    )
                else:
                    self.random_projections.append(
                        nn.Parameter(torch.zeros_like(self.random_projections[i - 1]), requires_grad=False)
                    )

        self.pair_wise_feature_dim = (2 * self.num_layer + 2) ** 2
        self.mlp = nn.Sequential(
            nn.Linear(self.pair_wise_feature_dim, self.pair_wise_feature_dim * 4),
            nn.ReLU(),
            nn.Linear(self.pair_wise_feature_dim * 4, self.pair_wise_feature_dim),
        )

    def update(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        src_node_ids = torch.from_numpy(src_node_ids).to(self.device)
        dst_node_ids = torch.from_numpy(dst_node_ids).to(self.device)
        next_time = node_interact_times[-1]
        node_interact_times = torch.from_numpy(node_interact_times).to(dtype=torch.float, device=self.device)
        time_weight = torch.exp(-self.time_decay_weight * (next_time - node_interact_times))[:, None]

        for i in range(1, self.num_layer + 1):
            self.random_projections[i].data = self.random_projections[i].data * np.power(
                np.exp(-self.time_decay_weight * (next_time - self.now_time.cpu().numpy())), i
            )

        for i in range(self.num_layer, 0, -1):
            src_update_messages = self.random_projections[i - 1][dst_node_ids] * time_weight
            dst_update_messages = self.random_projections[i - 1][src_node_ids] * time_weight
            self.random_projections[i].scatter_add_(
                dim=0, index=src_node_ids[:, None].expand(-1, self.dim), src=src_update_messages
            )
            self.random_projections[i].scatter_add_(
                dim=0, index=dst_node_ids[:, None].expand(-1, self.dim), src=dst_update_messages
            )

        self.now_time.data = torch.tensor(next_time, device=self.device)

    def get_random_projections(self, node_ids: np.ndarray):
        """
        FIX: convert numpy node_ids to LongTensor before indexing.
        """
        idx = torch.as_tensor(node_ids, dtype=torch.long)  # CPU LongTensor is fine for indexing
        random_projections = []
        for i in range(self.num_layer + 1):
            random_projections.append(self.random_projections[i][idx])
        return random_projections

    def get_pair_wise_feature(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray):
        src_random_projections = torch.stack(self.get_random_projections(src_node_ids), dim=1)
        dst_random_projections = torch.stack(self.get_random_projections(dst_node_ids), dim=1)
        random_projections = torch.cat([src_random_projections, dst_random_projections], dim=1)
        random_feature = torch.matmul(random_projections, random_projections.transpose(1, 2)).reshape(
            len(src_node_ids), -1
        )
        if self.not_scale:
            return self.mlp(random_feature)
        else:
            random_feature = torch.where(random_feature < 0, torch.zeros_like(random_feature), random_feature)
            random_feature = torch.log(random_feature + 1.0)
            return self.mlp(random_feature)

    def reset_random_projections(self):
        for i in range(1, self.num_layer + 1):
            nn.init.zeros_(self.random_projections[i])
        self.now_time.data = self.begging_time.clone()
        if not self.use_matrix:
            nn.init.normal_(self.random_projections[0], mean=0, std=1 / math.sqrt(self.dim))

    def backup_random_projections(self):
        return self.now_time.clone(), [self.random_projections[i].clone() for i in range(1, self.num_layer + 1)]

    def reload_random_projections(self, random_projections):
        now_time, random_projections = random_projections
        self.now_time.data = now_time.clone()
        for i in range(1, self.num_layer + 1):
            self.random_projections[i].data = random_projections[i - 1].clone()


# -----------------------------------------------------------------------------
# TPNet (architecture unchanged)
# -----------------------------------------------------------------------------
class TPNet(torch.nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray,
                 neighbor_sampler: NeighborSampler, time_feat_dim: int, dropout: float,
                 random_projections: RandomProjectionModule, num_layers: int, num_neighbors: int, device: str):
        super(TPNet, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.dropout = dropout
        self.device = device

        self.num_nodes = self.node_raw_features.shape[0]

        self.random_projections = random_projections
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.embedding_module = TPNetEmbedding(
            node_raw_features=self.node_raw_features,
            edge_raw_features=self.edge_raw_features,
            neighbor_sampler=neighbor_sampler,
            time_encoder=self.time_encoder,
            node_feat_dim=self.node_feat_dim,
            edge_feat_dim=self.edge_feat_dim,
            time_feat_dim=self.time_feat_dim,
            num_layers=num_layers,
            num_neighbors=num_neighbors,
            dropout=self.dropout,
            random_projections=self.random_projections,
        )

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray):
        node_embeddings = self.embedding_module.compute_node_temporal_embeddings(
            node_ids=np.concatenate([src_node_ids, dst_node_ids]),
            src_node_ids=np.tile(src_node_ids, 2),
            dst_node_ids=np.tile(dst_node_ids, 2),
            node_interact_times=np.tile(node_interact_times, 2),
        )
        src_node_embeddings, dst_node_embeddings = (
            node_embeddings[: len(src_node_ids)],
            node_embeddings[len(src_node_ids):],
        )
        return src_node_embeddings, dst_node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.embedding_module.neighbor_sampler = neighbor_sampler
        if getattr(self.embedding_module.neighbor_sampler, "sample_neighbor_strategy", None) in [
            "uniform", "time_interval_aware",
        ]:
            if getattr(self.embedding_module.neighbor_sampler, "seed", None) is not None and hasattr(
                self.embedding_module.neighbor_sampler, "reset_random_state"
            ):
                self.embedding_module.neighbor_sampler.reset_random_state()


# -----------------------------------------------------------------------------
# TPNetEmbedding (architecture unchanged)
# -----------------------------------------------------------------------------
class TPNetEmbedding(nn.Module):
    def __init__(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor,
                 neighbor_sampler: NeighborSampler, time_encoder: nn.Module,
                 node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_layers: int, num_neighbors: int, dropout: float,
                 random_projections: RandomProjectionModule):
        super(TPNetEmbedding, self).__init__()

        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.neighbor_sampler = neighbor_sampler
        self.time_encoder = time_encoder
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.random_projections = random_projections

        if self.random_projections is None:
            self.random_feature_dim = 0
        else:
            self.random_feature_dim = self.random_projections.pair_wise_feature_dim * 2

        self.projection_layer = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim + time_feat_dim + self.random_feature_dim, self.node_feat_dim * 2),
            nn.ReLU(),
            nn.Linear(self.node_feat_dim * 2, self.node_feat_dim),
        )

        self.mlp_mixers = nn.ModuleList([
            MLPMixer(
                num_tokens=self.num_neighbors,
                num_channels=self.node_feat_dim,
                token_dim_expansion_factor=0.5,
                channel_dim_expansion_factor=4.0,
                dropout=self.dropout,
            ) for _ in range(self.num_layers)
        ])

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, src_node_ids: np.ndarray,
                                         dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        device = self.node_raw_features.device

        neighbor_node_ids, neighbor_edge_ids, neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=self.num_neighbors
        )

        neighbor_node_features = self.node_raw_features[torch.from_numpy(neighbor_node_ids)]
        neighbor_delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(device)
        neighbor_delta_times = torch.log(neighbor_delta_times + 1.0)
        neighbor_time_features = self.time_encoder(neighbor_delta_times)
        neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]

        if self.random_projections is not None:
            concat_neighbor_random_features = self.random_projections.get_pair_wise_feature(
                src_node_ids=np.tile(neighbor_node_ids.reshape(-1), 2),
                dst_node_ids=np.concatenate(
                    [np.repeat(src_node_ids, self.num_neighbors), np.repeat(dst_node_ids, self.num_neighbors)]
                ),
            )
            neighbor_random_features = torch.cat(
                [
                    concat_neighbor_random_features[: len(node_ids) * self.num_neighbors],
                    concat_neighbor_random_features[len(node_ids) * self.num_neighbors:],
                ],
                dim=1,
            ).reshape(len(node_ids), self.num_neighbors, -1)

            neighbor_combine_features = torch.cat(
                [neighbor_node_features, neighbor_time_features, neighbor_edge_features, neighbor_random_features], dim=2
            )
        else:
            neighbor_combine_features = torch.cat(
                [neighbor_node_features, neighbor_time_features, neighbor_edge_features], dim=2
            )

        embeddings = self.projection_layer(neighbor_combine_features)
        embeddings.masked_fill(torch.from_numpy(neighbor_node_ids == 0)[:, :, None].to(device), 0)

        for mlp_mixer in self.mlp_mixers:
            embeddings = mlp_mixer(embeddings)

        embeddings = torch.mean(embeddings, dim=1)
        return embeddings


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        super(FeedForwardNet, self).__init__()
        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class MLPMixer(nn.Module):
    def __init__(self, num_tokens: int, num_channels: int,
                 token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0,
                 dropout: float = 0.0):
        super(MLPMixer, self).__init__()
        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(
            input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor, dropout=dropout
        )
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(
            input_dim=num_channels, dim_expansion_factor=channel_dim_expansion_factor, dropout=dropout
        )

    def forward(self, input_tensor: torch.Tensor):
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        output_tensor = hidden_tensor + input_tensor

        hidden_tensor = self.channel_norm(output_tensor)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        output_tensor = hidden_tensor + output_tensor
        return output_tensor
# ========================== TPNet on JODIE/Wikipedia: Training Module ==========================


# ---- import your TPNet definitions ----
# from tpnet import TPNet, RandomProjectionModule, get_neighbor_sampler


# ----- Tee (log to file + stdout) -----
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


# ----- Repro helpers -----
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)


# ----- Link predictor -----
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)
    def forward(self, z_src, z_dst):
        return self.lin_final((self.lin_src(z_src) * self.lin_dst(z_dst)).relu())


# ============================ TPNet Train / Eval loops ============================

def _assert_id_bounds(tag: str, arr: np.ndarray, nnodes: int):
    if arr.size == 0: return
    amin, amax = int(arr.min()), int(arr.max())
    if amin < 0 or amax >= nnodes:
        raise RuntimeError(f"{tag} IDs out of range: min={amin}, max={amax}, allowed=[0,{nnodes-1}]")

def _tpnet_forward_batch(tp_model: TPNet,
                         link_pred: LinkPredictor,
                         src_np: np.ndarray, dst_np: np.ndarray, t_np: np.ndarray,
                         neg_dst_np: Optional[np.ndarray],
                         nnodes: int):
    _assert_id_bounds("src", src_np, nnodes)
    _assert_id_bounds("dst", dst_np, nnodes)
    if neg_dst_np is not None:
        _assert_id_bounds("neg_dst", neg_dst_np, nnodes)

    z_src, z_dst = tp_model.compute_src_dst_node_temporal_embeddings(src_np, dst_np, t_np)
    pos_out = link_pred(z_src, z_dst)

    if neg_dst_np is None:
        return pos_out, None

    z_src_neg, z_dst_neg = tp_model.compute_src_dst_node_temporal_embeddings(src_np, neg_dst_np, t_np)
    neg_out = link_pred(z_src_neg, z_dst_neg)
    return pos_out, neg_out


def train_epoch(loader, memory, gnn, link_pred, optimizer, neighbor_loader, assoc, data_full, criterion,
                nnodes: int):
    """Train TPNet (signature kept TGN-compatible; unused placeholders allowed)."""
    tp_model: TPNet = gnn
    tp_model.train(); link_pred.train()

    if tp_model.random_projections is not None:
        tp_model.random_projections.reset_random_projections()

    total_loss = total_pos_loss = total_neg_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        # JODIE wikipedia has 0..num_nodes-1; shift to 1..N to reserve PAD=0
        src_np = batch.src.cpu().numpy().astype(np.int64) + 1
        dst_np = batch.dst.cpu().numpy().astype(np.int64) + 1
        neg_np = batch.neg_dst.cpu().numpy().astype(np.int64) + 1
        t_np   = batch.t.cpu().numpy().astype(np.float64)

        pos_out, neg_out = _tpnet_forward_batch(tp_model, link_pred, src_np, dst_np, t_np, neg_np, nnodes)

        loss_pos = criterion(pos_out, torch.ones_like(pos_out))
        loss_neg = criterion(neg_out, torch.zeros_like(neg_out))
        loss = loss_pos + loss_neg
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in optimizer.param_groups[0]['params'] if p.requires_grad], 0.5)
        optimizer.step()

        if tp_model.random_projections is not None:
            tp_model.random_projections.update(src_np, dst_np, t_np)

        total_loss     += float(loss)     * batch.num_events
        total_pos_loss += float(loss_pos) * batch.num_events
        total_neg_loss += float(loss_neg) * batch.num_events

    N = loader.data.num_events
    return (total_loss / N, total_pos_loss / N, total_neg_loss / N)


@torch.no_grad()
def evaluate(loader, memory, gnn, link_pred, neighbor_loader, assoc, data_full, criterion,
             nnodes: int):
    tp_model: TPNet = gnn
    tp_model.eval(); link_pred.eval()

    if tp_model.random_projections is not None:
        tp_model.random_projections.reset_random_projections()

    aps, aucs, mrrs, total_loss, total_events = [], [], [], 0.0, 0

    for batch in loader:
        src_np = batch.src.cpu().numpy().astype(np.int64) + 1
        dst_np = batch.dst.cpu().numpy().astype(np.int64) + 1
        neg_np = batch.neg_dst.cpu().numpy().astype(np.int64) + 1
        t_np   = batch.t.cpu().numpy().astype(np.float64)

        pos_out, neg_out = _tpnet_forward_batch(tp_model, link_pred, src_np, dst_np, t_np, neg_np, nnodes)
        loss = criterion(pos_out, torch.ones_like(pos_out)) + criterion(neg_out, torch.zeros_like(neg_out))
        total_loss += float(loss) * batch.num_events; total_events += batch.num_events

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)
        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        scores = torch.stack([pos_out, neg_out], dim=1).cpu()
        ranks = (scores.sort(dim=1, descending=True)[1] == 0).nonzero(as_tuple=False)[:, 1] + 1
        mrrs.append((1.0 / ranks.float()).mean().item())

        if tp_model.random_projections is not None:
            tp_model.random_projections.update(src_np, dst_np, t_np)

    dummy_emb = torch.empty(0, 0)
    return total_loss / total_events, float(np.mean(aps)), float(np.mean(aucs)), float(np.mean(mrrs)), dummy_emb


# ----- One run (per seed) -----
def run_single_experiment(seed, data_full, train_data, val_loader, test_loader, device, loader_kwargs, dataset_name: str):
    print(f"\n{'='*30} RUNNING SEED {seed} ({dataset_name}) {'='*30}")
    set_seed(seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # ---------------- Prepare features & sampler for TPNet ----------------
    # Shift to 1..N for sampler (reserve 0 as PAD)
    src_all = data_full.src.cpu().numpy().astype(np.int64) + 1
    dst_all = data_full.dst.cpu().numpy().astype(np.int64) + 1
    t_all   = data_full.t.cpu().numpy().astype(np.float64)

    # fabricate unique edge IDs per event (1..M)
    M = len(src_all)
    edge_ids = np.arange(1, M + 1, dtype=np.int64)

    sampler_data = NS(
        src_node_ids=src_all,
        dst_node_ids=dst_all,
        edge_ids=edge_ids,
        node_interact_times=t_all
    )
    neighbor_sampler = get_neighbor_sampler(sampler_data, sample_neighbor_strategy='recent', seed=seed)

    # Node & edge raw features (numpy)
    Dn = 128
    # nnodes = data_full.num_nodes + 1 (PAD row 0)
    nnodes = int(data_full.num_nodes) + 1
    node_raw_features = np.random.default_rng(seed).normal(0.0, 1.0, size=(nnodes, Dn)).astype(np.float32)

    # Edge features: JODIE provides data.msg or None; ensure 2D shape
    if getattr(data_full, "msg", None) is None:
        De = 1
        msg_np = np.zeros((M, De), dtype=np.float32)
    else:
        msg = data_full.msg
        if msg.dim() == 1:
            msg = msg.view(-1, 1)
        De = int(msg.size(-1))
        msg_np = msg.cpu().numpy().astype(np.float32)
    edge_raw_features = np.zeros((M + 1, De), dtype=np.float32)
    edge_raw_features[1:] = msg_np

    # Random projection module
    rp = RandomProjectionModule(
        node_num=nnodes,          # includes PAD row 0
        edge_num=M,
        dim_factor=8,
        num_layer=2,              # K-hop depth
        time_decay_weight=0.1,    # lambda
        device=str(device),
        use_matrix=False,
        beginning_time=np.float64(t_all.min()),
        not_scale=False,
        enforce_dim=-1
    ).to(device)

    # ---------------- Build TPNet + predictor ----------------
    tpnet = TPNet(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler,
        time_feat_dim=32,
        dropout=0.1,
        random_projections=rp,
        num_layers=2,
        num_neighbors=20,
        device=str(device),
    ).to(device)

    link_pred = LinkPredictor(tpnet.node_feat_dim).to(device)

    # Optimizer/scheduler/criterion
    modules = [tpnet, link_pred]
    all_params = [p for m in modules for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(all_params, lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Placeholders maintained for signature compatibility:
    memory = None
    neighbor_loader = None
    assoc = None

    best_test_metrics = {}
    patience_counter = 0
    best_ema_val_ap = 0.0
    ema_val_ap = 0.0
    MIN_EPOCHS_FOR_STOPPING = 15
    EMA_ALPHA = 0.2

    epoch_times, embedding_history = [], []

    # Fresh loaders each epoch (stable in notebooks)
    for epoch in range(1, 51):
        epoch_start = time.time()

        train_loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=1.0, **loader_kwargs)
        train_loss, train_pos_loss, train_neg_loss = train_epoch(
            train_loader, memory, tpnet, link_pred, optimizer, neighbor_loader, assoc, data_full, criterion,
            nnodes
        )

        val_loss, val_ap, val_auc, val_mrr, _ = evaluate(
            TemporalDataLoader(val_data, batch_size=200, neg_sampling_ratio=1.0, **loader_kwargs),
            memory, tpnet, link_pred, neighbor_loader, assoc, data_full, criterion, nnodes
        )

        scheduler.step(val_loss)
        epoch_times.append(time.time() - epoch_start)

        ema_val_ap = val_ap if epoch == 1 else (EMA_ALPHA * val_ap + (1 - EMA_ALPHA) * ema_val_ap)

        print(f"Seed {seed}|Epoch {epoch:02d}|LR {optimizer.param_groups[0]['lr']:.6f}|Time {epoch_times[-1]:.2f}s|Train Loss {train_loss:.4f}")
        print(f"  Train Stats: Total={train_loss:.4f} | Pos_Loss={train_pos_loss:.4f} | Neg_Loss={train_neg_loss:.4f}")
        print(f"  Val   Stats: Loss={val_loss:.4f} | AP {val_ap:.4f} (EMA {ema_val_ap:.4f}) | AUC {val_auc:.4f} | MRR {val_mrr:.4f}")

        if ema_val_ap > best_ema_val_ap:
            best_ema_val_ap = ema_val_ap
            patience_counter = 0

            _, test_ap, test_auc, test_mrr, _ = evaluate(
                TemporalDataLoader(test_data, batch_size=200, neg_sampling_ratio=1.0, **loader_kwargs),
                memory, tpnet, link_pred, neighbor_loader, assoc, data_full, criterion, nnodes
            )
            best_test_metrics = {"ap": test_ap, "auc": test_auc, "mrr": test_mrr}
            print(f"  >>> New best! Test AP: {test_ap:.4f}, AUC: {test_auc:.4f}, MRR: {test_mrr:.4f}")
            os.makedirs('saved_models', exist_ok=True)
            torch.save(
                {'tpnet': tpnet.state_dict(), 'link_pred': link_pred.state_dict()},
                osp.join('saved_models', f'{dataset_name}_tpnet_seed_{seed}.pth')
            )
        else:
            if epoch > MIN_EPOCHS_FOR_STOPPING:
                patience_counter += 1

        if patience_counter >= 15:
            print("Early stopping triggered.")
            break

    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024*1024) if device.type == 'cuda' else 0.0
    best_test_metrics['avg_epoch_time'] = avg_epoch_time
    best_test_metrics['peak_memory_mb'] = peak_memory_mb

    print(f"--- Finished Seed {seed} -> Best Test Metrics: {best_test_metrics} ---")
    return best_test_metrics, []


def strip_scalar_attrs(d: TemporalData):
    for k in list(d.keys()):
        v = d[k]
        if isinstance(v, (int, float)):
            del d[k]
        else:
            try:
                if hasattr(v, "dim") and v.dim() == 0:
                    del d[k]
            except Exception:
                pass


# ============================= Main block (Wikipedia + TPNet) =============================
if __name__ == '__main__':
    dataset_name = 'lastfm'  # JODIE dataset key

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [123451, 123452, 123453, 123454, 123455]

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"Could not enforce deterministic algorithms: {e}")

    log_filename = f"{dataset_name}_tpnet_seed_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    tee = Tee(log_filename)
    try:
        print(f"Using device: {device}")

        # ---- Load JODIE/Wikipedia ----
        path = osp.join('.', 'data', 'JODIE')
        dataset = JODIEDataset(path, name=dataset_name)
        data = dataset[0]

        # Ensure msg exists and is 2D for features; if None, we’ll handle inside run()
        if getattr(data, "msg", None) is None:
            print("No edge features (msg) found; will default to zeros with dim=1.")

        # Make sure no stray persisted scalars interfere
        strip_scalar_attrs(data)

        # Time-respecting split
        train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)

        # DataLoader settings (avoid teardown warnings in notebooks)
        loader_kwargs = dict(
            num_workers=2, pin_memory=True, persistent_workers=False,
            prefetch_factor=2, worker_init_fn=seed_worker
        )

        all_results = []
        for seed in seeds:
            result, _ = run_single_experiment(seed, data, train_data, val_data, test_data, device, loader_kwargs, dataset_name)
            all_results.append(result)

        print(f"\n\n{'='*30} FINAL EVALUATION COMPLETE {'='*30}")
        test_aps   = [r.get('ap',  0.0) for r in all_results]
        test_aucs  = [r.get('auc', 0.0) for r in all_results]
        test_mrrs  = [r.get('mrr', 0.0) for r in all_results]
        epoch_times = [r.get('avg_epoch_time', 0.0) for r in all_results]
        peak_mems   = [r.get('peak_memory_mb', 0.0) for r in all_results]

        print("\n--- Final Performance Results (Mean ± Std Dev over 5 Runs) ---")
        print(f"Test AP:            {np.mean(test_aps):.4f} ± {np.std(test_aps):.4f}")
        print(f"Test AUC:           {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        print(f"Test MRR:           {np.mean(test_mrrs):.4f} ± {np.std(test_mrrs):.4f}")
        print("-" * 60)
        print("\n--- Final Efficiency Results (Mean ± Std Dev over 5 Runs) ---")
        print(f"Avg. Runtime/Epoch: {np.mean(epoch_times):.2f}s ± {np.std(epoch_times):.2f}s")
        if device.type == 'cuda' and len(peak_mems) > 0 and peak_mems[0] > 0:
            print(f"Peak Memory Usage:  {np.mean(peak_mems):.2f} MB ± {np.std(peak_mems):.2f} MB")
        else:
            print("Peak Memory Usage:  N/A (Not a CUDA device)")
        print("-" * 60)
    finally:
        sys.stdout = tee.stdout
        tee.file.close()
