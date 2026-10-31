import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.nn import GCNConv

from ..feature_extractors.graph_features import (
    build_graph_batch_from_input,
    build_slf_graph_batch_from_input,
)
from .base.base_relation_network import RelationFusionNetwork
from .base.mlp import MLP
from .graph_rel_net import Edge2Node, EdgeEncoder
from pysoundloc.pysoundloc.utils.math import grid_argmax


class MPNN_GCC(nn.Module):
    """Message-passing NN for SSL (MPNN-GCC).

    Incremental message-passing upgrade of the GNN-GCC relation network: the
    pairwise microphone relations are obtained through message passing over a
    complete directed microphone graph (GCC-PHAT edge features plus array
    geometry), then enriched with the raw pair features (the relation branch
    of Graph-RelNet) before being fused and mapped to the SSL output.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config = (
            OmegaConf.to_object(config)
            if OmegaConf.is_config(config)
            else config
        )
        self.features_config = self.config["features"]
        self.targets_config = self.config["targets"]

        dataset_config = self.features_config["dataset"]
        self.sr = dataset_config["sr"]
        self.is_metadata_aware = self.config.get("is_metadata_aware", True)
        self.use_rt60_as_metadata = dataset_config["use_rt60_as_metadata"]
        self.metadata_dim = dataset_config["metadata_dim"]
        self.edge_feature_type = str(self.config.get("edge_feature_type", "gcc"))
        if self.edge_feature_type not in {"gcc", "slf"}:
            raise ValueError(f"Unknown edge_feature_type: {self.edge_feature_type!r}")
        self.n_edge_features = int(
            self.config.get(
                "n_edge_features",
                self.config.get("n_edge_bins", 200),
            )
        )
        if self.edge_feature_type == "slf":
            self.n_grid_points = int(self.config["targets"]["n_points_per_axis"])
            self.n_edge_features = self.n_grid_points ** 2
            self.n_correlation_neighbours = int(
                self.config.get("n_correlation_neighbours", 10)
            )
        self.emb_dim = int(self.config.get("emb_dim", 128))
        self.node_hid = int(self.config.get("node_hid", 128))

        # Relation-branch skip: GNN-GCC and Graph-RelNet feed the raw pair
        # features to the prediction head. Without it, the pair GCC-PHAT is
        # averaged away by Edge2Node and the head only sees node embeddings.
        self.pair_skip = str(self.config.get("pair_skip", "raw"))
        if self.pair_skip not in {"none", "raw", "encoded"}:
            raise ValueError(f"Unknown pair_skip mode: {self.pair_skip!r}")
        self.edge_weighting = str(self.config.get("edge_weighting", "distance"))
        if self.edge_weighting not in {"distance", "uniform"}:
            raise ValueError(f"Unknown edge_weighting: {self.edge_weighting!r}")
        self._pair_index_cache = {}

        metadata_size = 0
        if self.is_metadata_aware:
            metadata_size = 3 * self.metadata_dim
            if self.use_rt60_as_metadata:
                metadata_size += 1

        self.pair_skip_dim = {
            "none": 0,
            "raw": self.n_edge_features + metadata_size,
            "encoded": self.emb_dim,
        }[self.pair_skip]

        self.edge_encoder = EdgeEncoder(edge_dim=self.n_edge_features + metadata_size, emb_dim=self.emb_dim)
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
            n_input_features=2 * self.node_hid + self.pair_skip_dim,
            n_output_features=self.n_pairwise_features,
            n_hidden_features=self.n_pairwise_features,
            n_layers=int(self.config.get("n_layers", 3)),
            activation=self.config.get("activation", "relu"),
            # GNN-GCC's pairwise relation MLP ends in the activation, not a
            # linear layer; keep this aligned for the A/B comparison.
            output_activation=self.config.get("activation", "relu"),
            batch_norm=self.config.get("batch_norm", False),
            dropout_rate=self.config.get("dropout_rate", 0.0),
        )

    def _pair_indices(self, n_mics, device):
        """Undirected pair indices and their position in the directed edge list.

        ``build_graph_batch_from_input`` lays edges out row-major with the
        self-pair removed: (0,1)..(0,M-1), (1,0), (1,2).., so pair (i, j),
        i < j, sits at ``i * (M - 1) + (j if j < i else j - 1)``.
        """
        key = (n_mics, device.type, device.index)
        cached = self._pair_index_cache.get(key)
        if cached is not None:
            return cached

        i_idx, j_idx = torch.triu_indices(n_mics, n_mics, offset=1)
        pair_edge_idx = i_idx * (n_mics - 1) + torch.where(j_idx < i_idx, j_idx, j_idx - 1)

        # Guard against a silent ordering change in the graph builder.
        idx = torch.arange(n_mics)
        src = idx.repeat_interleave(n_mics)
        dst = idx.repeat(n_mics)
        keep = src != dst
        src, dst = src[keep], dst[keep]
        assert (src[pair_edge_idx] == i_idx).all() and (dst[pair_edge_idx] == j_idx).all(), (
            "pair_edge_idx does not match the directed edge ordering"
        )

        cached = (i_idx.to(device), j_idx.to(device), pair_edge_idx.to(device))
        self._pair_index_cache[key] = cached
        return cached

    def forward(self, x, estimate_coords=False):
        if self.edge_feature_type == "slf":
            data = build_slf_graph_batch_from_input(
                x,
                sr=self.sr,
                n_grid_points=self.n_grid_points,
                is_metadata_aware=self.is_metadata_aware,
                use_rt60_as_metadata=self.use_rt60_as_metadata,
                metadata_dim=self.metadata_dim,
                n_correlation_neighbours=self.n_correlation_neighbours,
                normalize_slf=bool(self.config.get("normalize_slf", False)),
            )
        else:
            data = build_graph_batch_from_input(
                x,
                sr=self.sr,
                n_edge_bins=self.n_edge_features,
                is_metadata_aware=self.is_metadata_aware,
                use_rt60_as_metadata=self.use_rt60_as_metadata,
                metadata_dim=self.metadata_dim,
                # Match GNN-GCC's GccPhat preprocessing exactly: signed GCC-PHAT,
                # normalized per pair by the max of the central bins.
                abs_gcc=False,
                normalize_gcc=True,
            )

        edge_index = data.edge_index
        raw_edge_attr = data.edge_attr  # kept for the relation-branch skip
        array = data.array

        src, dst = edge_index
        source_nodes = array[src, :]
        target_nodes = array[dst, :]
        relative_positions = source_nodes - target_nodes
        distances = torch.norm(relative_positions, dim=1, keepdim=True)
        relative_unit = relative_positions / (distances + 1e-8)
        log_dist = torch.log(distances + 1e-6)

        edge_emb = self.edge_encoder(torch.cat([raw_edge_attr, relative_unit, log_dist], dim=1))

        x_nodes = self.edge2node(edge_emb, edge_index, data.x.size(0))
        edge_weight = distances.squeeze(-1) if self.edge_weighting == "distance" else None

        h0 = x_nodes
        x_nodes = self.gnn1(x_nodes, edge_index, edge_weight)
        x_nodes = self.n_norm1(x_nodes + h0)
        x_nodes = F.gelu(x_nodes)
        h1 = x_nodes
        x_nodes = self.gnn2(x_nodes, edge_index, edge_weight)
        x_nodes = self.n_norm2(x_nodes + h1)

        # Per-pair relations from post-message-passing node embeddings, plus the
        # raw pair features (GNN-GCC / Graph-RelNet relation branch).
        # PyG Batch lays nodes out batch-major: node index = b * n_mics + i.
        n_mics = x["signal"].shape[1]
        n_samples = x["signal"].shape[0]
        i_idx, j_idx, pair_edge_idx = self._pair_indices(n_mics, x_nodes.device)

        h = x_nodes.reshape(n_samples, n_mics, self.node_hid)
        pair_inputs = [h[:, i_idx], h[:, j_idx]]
        if self.pair_skip == "raw":
            pair_inputs.append(
                raw_edge_attr.reshape(n_samples, -1, raw_edge_attr.size(-1))[:, pair_edge_idx]
            )
        elif self.pair_skip == "encoded":
            pair_inputs.append(edge_emb.reshape(n_samples, -1, self.emb_dim)[:, pair_edge_idx])

        pairwise_relations = self.pairwise_relation_network(torch.cat(pair_inputs, dim=-1))

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
