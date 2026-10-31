import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.nn import GCNConv

from ..feature_extractors.graph_features import build_graph_batch_from_input
from .base.base_relation_network import RelationFusionNetwork
from .base.mlp import MLP
from .graph_rel_net import Edge2Node, EdgeEncoder
from pysoundloc.pysoundloc.utils.math import grid_argmax


class MPNN_GCC(nn.Module):
    """Message-passing NN for SSL (MPNN-GCC).

    Incremental message-passing upgrade of the GNN-GCC relation network: the
    pairwise microphone relations are obtained through message passing over a
    complete directed microphone graph (GCC-PHAT edge features plus array
    geometry), before being fused and mapped to the SSL output.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config = OmegaConf.to_object(config)
        self.features_config = self.config["features"]
        self.targets_config = self.config["targets"]

        dataset_config = self.features_config["dataset"]
        self.sr = dataset_config["sr"]
        self.is_metadata_aware = self.config.get("is_metadata_aware", True)
        self.use_rt60_as_metadata = dataset_config["use_rt60_as_metadata"]
        self.metadata_dim = dataset_config["metadata_dim"]
        self.n_edge_bins = int(self.config.get("n_edge_bins", 200))
        self.emb_dim = int(self.config.get("emb_dim", 128))
        self.node_hid = int(self.config.get("node_hid", 128))

        metadata_size = 0
        if self.is_metadata_aware:
            metadata_size = 3 * self.metadata_dim
            if self.use_rt60_as_metadata:
                metadata_size += 1

        self.edge_encoder = EdgeEncoder(edge_dim=self.n_edge_bins + metadata_size, emb_dim=self.emb_dim)
        self.edge2node = Edge2Node(in_channels=self.emb_dim, out_channels=self.node_hid)
        self.gnn1 = GCNConv(self.node_hid, self.node_hid)
        self.gnn2 = GCNConv(self.node_hid, self.node_hid)
        self.n_norm1 = nn.LayerNorm(self.node_hid)
        self.n_norm2 = nn.LayerNorm(self.node_hid)

        n_input_features = int(dataset_config["n_input_seconds"] * self.sr)
        self.n_pairwise_features = int(self.config["n_pairwise_features"])
        self.output_target = self.targets_config["type"]
        if self.output_target == "source_coordinates":
            self.n_output_features = 2
        else:
            self.n_likelihood_grid_points_per_axis = int(self.targets_config["n_points_per_axis"])
            self.n_output_features = self.n_likelihood_grid_points_per_axis ** 2

        self.relation_fusion_network = RelationFusionNetwork(
            n_input_features=self.n_pairwise_features,
            n_output_features=self.n_output_features,
            activation=self.config.get("activation", "relu"),
            output_activation=self.config.get("output_activation", None),
            init_layers=self.config.get("init_layers", True),
            batch_norm=self.config.get("batch_norm", False),
            dropout_rate=self.config.get("dropout_rate", 0.0),
            n_layers=int(self.config.get("n_layers", 3)),
        )

        self.pairwise_relation_network = MLP(
            n_input_features=2 * self.node_hid,
            n_output_features=self.n_pairwise_features,
            n_hidden_features=self.n_pairwise_features,
            n_layers=int(self.config.get("n_layers", 3)),
            activation=self.config.get("activation", "relu"),
            output_activation=self.config.get("output_activation", None),
            batch_norm=self.config.get("batch_norm", False),
            dropout_rate=self.config.get("dropout_rate", 0.0),
        )

    def forward(self, x, estimate_coords=False):
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
        array = data.array

        src, dst = edge_index
        source_nodes = array[src, :]
        target_nodes = array[dst, :]
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

        # Pairwise relations from post-message-passing node embeddings
        n_mics = x["signal"].shape[1]
        n_pairs = n_mics * (n_mics - 1) // 2
        n_samples = x["signal"].shape[0]

        pairwise_relations = []
        for b in range(n_samples):
            for i in range(n_mics):
                for j in range(i + 1, n_mics):
                    h_i = x_nodes[b * n_mics + i]
                    h_j = x_nodes[b * n_mics + j]
                    x_ij = torch.cat([h_i, h_j], dim=-1)
                    pairwise_relations.append(self.pairwise_relation_network(x_ij))
        pairwise_relations = torch.stack(pairwise_relations, dim=1)

        y = self.relation_fusion_network(pairwise_relations)

        if self.output_target == "source_coordinates":
            return {"source_coordinates": y}

        grid = y.reshape((-1, self.n_likelihood_grid_points_per_axis, self.n_likelihood_grid_points_per_axis))
        source_coordinates = grid_argmax(grid, x["metadata"]["global"]["room_dims"][:, :2])

        if self.output_target == "grid":
            return {"grid": grid, "source_coordinates": source_coordinates}

        if estimate_coords and self.output_target == "likelihood_grid":
            return source_coordinates, grid

        return y
