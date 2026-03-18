import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv, global_mean_pool

from pysoundloc.pysoundloc.utils.math import grid_argmax

from .base.mlp import MLP
from ..feature_extractors.graph_features import build_graph_batch_from_input


class Edge2Node(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU())

    def forward(self, edge_emb, edge_index, num_nodes):
        src, dst = edge_index
        msg_src = scatter_mean(self.proj(edge_emb), src, dim=0, dim_size=num_nodes)
        msg_dst = scatter_mean(self.proj(edge_emb), dst, dim=0, dim_size=num_nodes)
        return 0.5 * (msg_src + msg_dst)


class EdgeEncoder(nn.Module):
    def __init__(self, edge_dim, emb_dim):
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim + 4, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, edge_attr):
        return self.edge_encoder(edge_attr)


class GraphRelNetGnnSsl(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = OmegaConf.to_object(config)
        self.features_config = self.config["features"]
        self.targets_config = self.config["targets"]

        self.sr = self.features_config["dataset"]["sr"]
        self.is_metadata_aware = self.config.get("is_metadata_aware", True)
        self.use_rt60_as_metadata = self.features_config["dataset"]["use_rt60_as_metadata"]
        self.metadata_dim = self.features_config["dataset"]["metadata_dim"]
        self.n_edge_bins = self.config["n_edge_bins"]
        self.grid_size = self.targets_config["n_points_per_axis"]
        self.output_target = self.targets_config["type"]
        self.normalize_output = self.config["normalize_output"]

        metadata_size = 0
        if self.is_metadata_aware:
            metadata_size = 3 * self.metadata_dim
            if self.use_rt60_as_metadata:
                metadata_size += 1

        edge_dim = self.n_edge_bins + metadata_size
        emb_dim = self.config["emb_dim"]
        node_hid = self.config["node_hid"]

        self.edge_encoder = EdgeEncoder(edge_dim=edge_dim, emb_dim=emb_dim)
        self.edge2node = Edge2Node(in_channels=emb_dim, out_channels=node_hid)

        self.gnn1 = GCNConv(node_hid, node_hid)
        self.gnn2 = GCNConv(node_hid, node_hid)
        self.n_norm1 = nn.LayerNorm(node_hid)
        self.n_norm2 = nn.LayerNorm(node_hid)

        self.head = MLP(
            n_input_features=node_hid + emb_dim,
            n_output_features=self.grid_size ** 2,
            n_hidden_features=self.config["head_hidden"],
            n_layers=self.config["head_layers"],
            activation=self.config["activation"],
            output_activation=self.config["output_activation"],
            batch_norm=self.config["batch_norm"],
            dropout_rate=self.config["dropout_rate"],
        )

    def forward(self, x, estimate_coords=True):
        data = build_graph_batch_from_input(
            x,
            sr=self.sr,
            n_edge_bins=self.n_edge_bins,
            is_metadata_aware=self.is_metadata_aware,
            use_rt60_as_metadata=self.use_rt60_as_metadata,
            metadata_dim=self.metadata_dim,
        )

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        array = data.array

        source_nodes = array[edge_index[0], :]
        target_nodes = array[edge_index[1], :]
        relative_positions = source_nodes - target_nodes
        distances = torch.norm(relative_positions, dim=1, keepdim=True)
        relative_unit = relative_positions / (distances + 1e-8)
        log_dist = torch.log(distances + 1e-6)

        edge_attr = torch.cat([edge_attr, relative_unit, log_dist], dim=1)
        edge_attr = self.edge_encoder(edge_attr)

        x_nodes = self.edge2node(edge_attr, edge_index, data.x.size(0))
        edge_weight = distances.squeeze(-1)

        h0 = x_nodes
        x_nodes = self.gnn1(x_nodes, edge_index, edge_weight)
        x_nodes = self.n_norm1(x_nodes + h0)
        x_nodes = F.gelu(x_nodes)

        h1 = x_nodes
        x_nodes = self.gnn2(x_nodes, edge_index, edge_weight)
        x_nodes = self.n_norm2(x_nodes + h1)

        edge_batch = batch[edge_index[0]]
        node_graph = global_mean_pool(x_nodes, batch)
        edge_graph = global_mean_pool(edge_attr, edge_batch)
        graph = torch.cat([node_graph, edge_graph], dim=-1)

        grid = self.head(graph).reshape((-1, self.grid_size, self.grid_size))

        if self.normalize_output:
            grid_max = grid.abs().flatten(start_dim=1).max(dim=1)[0].unsqueeze(1).unsqueeze(2)
            grid = grid / (grid_max + 1e-8)

        if self.output_target == "source_coordinates":
            return {"source_coordinates": grid_argmax(grid, x["metadata"]["global"]["room_dims"][:, :2])}

        if estimate_coords:
            estimated_coords = grid_argmax(grid, x["metadata"]["global"]["room_dims"][:, :2])
            return {
                "grid": grid,
                "source_coordinates": estimated_coords,
            }

        return {"grid": grid}
