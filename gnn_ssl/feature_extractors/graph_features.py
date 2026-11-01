import torch

from torch_geometric.data import Data

from pysoundloc.pysoundloc.gcc import gcc_phat_batch
from pysoundloc.pysoundloc.srp import compute_pairwise_srp_grids

_EDGE_INDEX_CACHE = {}


def _build_complete_directed_edge_index(n_nodes, device):
    key = (n_nodes, device.type, device.index)
    edge_index = _EDGE_INDEX_CACHE.get(key)
    if edge_index is None:
        idx = torch.arange(n_nodes, device=device)
        src = idx.repeat_interleave(n_nodes)
        dst = idx.repeat(n_nodes)
        mask = src != dst
        edge_index = torch.stack([src[mask], dst[mask]], dim=0)
        _EDGE_INDEX_CACHE[key] = edge_index
    return edge_index


def _compute_edge_attr(
    signals,
    edge_index,
    sr,
    n_edge_bins,
    abs_gcc=True,
    normalize_gcc=False,
):
    """Compute GCC-PHAT features for every edge, for the whole batch at once.

    Parameters
    ----------
    signals : torch.Tensor
        Batched microphone signals, shape ``(batch_size, n_mics, n_samples)``.
    edge_index : torch.Tensor
        Local (per-sample) edge indices, shape ``(2, n_edges)``.

    Returns
    -------
    torch.Tensor
        Shape ``(batch_size * n_edges, n_edge_bins)``, batch-major.
    """
    src, dst = edge_index
    batch_size = signals.shape[0]
    n_edges = edge_index.shape[1]

    x_i = signals[:, src].reshape(batch_size * n_edges, -1)
    x_j = signals[:, dst].reshape(batch_size * n_edges, -1)

    gcc = gcc_phat_batch(x_i, x_j, sr, abs=abs_gcc)[0]

    start = (gcc.shape[-1] - n_edge_bins) // 2
    end = start + n_edge_bins
    gcc = gcc[:, start:end]

    if gcc.shape[-1] < n_edge_bins:
        gcc = torch.nn.functional.pad(gcc, (0, n_edge_bins - gcc.shape[-1]))

    # Same normalization as GccPhat (gnn_ssl.feature_extractors): divide
    # the central bins by their max, per pair.
    if normalize_gcc:
        gcc = gcc / gcc.max(dim=1, keepdim=True)[0]

    return gcc


def _compute_edge_metadata(
    mic_coords,
    room_dims,
    rt60,
    edge_index,
    metadata_dim,
    use_rt60_as_metadata,
):
    """Vectorized version of the per-edge metadata concatenation.

    Returns ``(batch_size * n_edges, n_metadata_features)``, batch-major.
    """
    src, dst = edge_index
    batch_size = mic_coords.shape[0]
    n_edges = edge_index.shape[1]

    coords = mic_coords[:, :, :metadata_dim]
    features = [
        coords[:, src].reshape(batch_size * n_edges, -1),
        coords[:, dst].reshape(batch_size * n_edges, -1),
        room_dims[:, :metadata_dim]
        .unsqueeze(1)
        .expand(-1, n_edges, -1)
        .reshape(batch_size * n_edges, -1),
    ]

    if use_rt60_as_metadata:
        if rt60 is None:
            features.append(mic_coords.new_zeros((batch_size * n_edges, 1)))
        else:
            features.append(
                rt60.reshape(batch_size, -1)
                .unsqueeze(1)
                .expand(-1, n_edges, -1)
                .reshape(batch_size * n_edges, -1)
            )

    return torch.cat(features, dim=1)


def _compute_slf_edge_attr(
    signals,
    mic_coords,
    room_dims,
    edge_index,
    sr,
    n_grid_points,
    n_correlation_neighbours=10,
    normalize=False,
):
    """Compute pairwise SLF grids once per undirected pair and reuse them."""
    src, dst = edge_index
    batch_size = signals.shape[0]
    n_mics = signals.shape[1]

    pair_src, pair_dst = torch.triu_indices(
        n_mics, n_mics, offset=1, device=signals.device
    )
    pair_ids = torch.full(
        (n_mics, n_mics), -1, dtype=torch.long, device=signals.device
    )
    pair_ids[pair_src, pair_dst] = torch.arange(
        pair_src.numel(), device=signals.device
    )
    pair_ids[pair_dst, pair_src] = pair_ids[pair_src, pair_dst]
    directed_pair_ids = pair_ids[src, dst]
    n_pairs = pair_src.numel()

    x_i = signals[:, pair_src].reshape(batch_size * n_pairs, -1)
    x_j = signals[:, pair_dst].reshape(batch_size * n_pairs, -1)
    coords_i = mic_coords[:, pair_src, :2].reshape(batch_size * n_pairs, -1)
    coords_j = mic_coords[:, pair_dst, :2].reshape(batch_size * n_pairs, -1)
    dims = room_dims[:, :2].unsqueeze(1).expand(-1, n_pairs, -1)
    dims = dims.reshape(batch_size * n_pairs, -1)

    grids = compute_pairwise_srp_grids(
        x_i,
        x_j,
        sr,
        coords_i,
        coords_j,
        dims,
        n_grid_points=n_grid_points,
        n_correlation_neighbours=n_correlation_neighbours,
    )
    grids = grids.reshape(batch_size, n_pairs, -1)
    grids = grids[:, directed_pair_ids].reshape(batch_size * edge_index.shape[1], -1)
    if normalize:
        grids = grids / (grids.amax(dim=1, keepdim=True) + 1e-8)
    return grids


def build_graph_batch_from_input(
    x,
    sr,
    n_edge_bins,
    is_metadata_aware=True,
    use_rt60_as_metadata=False,
    metadata_dim=3,
    abs_gcc=True,
    normalize_gcc=False,
):
    signals = x["signal"]
    mic_coords = x["metadata"]["local"]["mic_coordinates"]
    room_dims = x["metadata"]["global"]["room_dims"]
    rt60 = x["metadata"]["global"].get("rt60", None)

    if mic_coords.ndim == 4:
        mic_coords = mic_coords[:, 0]

    batch_size, n_mics, _ = signals.shape
    device = signals.device

    edge_index = _build_complete_directed_edge_index(n_mics, device)
    n_edges = edge_index.shape[1]

    edge_attr = _compute_edge_attr(
        signals,
        edge_index,
        sr=sr,
        n_edge_bins=n_edge_bins,
        abs_gcc=abs_gcc,
        normalize_gcc=normalize_gcc,
    )

    if is_metadata_aware:
        edge_metadata = _compute_edge_metadata(
            mic_coords,
            room_dims,
            rt60,
            edge_index,
            metadata_dim=metadata_dim,
            use_rt60_as_metadata=use_rt60_as_metadata,
        )
        edge_attr = torch.cat([edge_attr, edge_metadata], dim=1)

    # Batch the per-sample graphs. Sample b's nodes are offset by b * n_mics
    # and its edges are stored contiguously, matching PyG's Batch layout.
    offsets = (torch.arange(batch_size, device=device) * n_mics).repeat_interleave(n_edges)
    edge_index = edge_index.repeat(1, batch_size) + offsets

    return Data(
        x=torch.zeros((batch_size * n_mics, 1), device=device, dtype=signals.dtype),
        edge_index=edge_index,
        edge_attr=edge_attr,
        array=mic_coords.reshape(batch_size * n_mics, -1)[:, :3],
        batch=torch.arange(batch_size, device=device).repeat_interleave(n_mics),
    )


def build_slf_graph_batch_from_input(
    x,
    sr,
    n_grid_points,
    is_metadata_aware=True,
    use_rt60_as_metadata=False,
    metadata_dim=3,
    n_correlation_neighbours=10,
    normalize_slf=False,
):
    """Build a complete microphone graph with pairwise SLF edge features."""
    signals = x["signal"]
    mic_coords = x["metadata"]["local"]["mic_coordinates"]
    room_dims = x["metadata"]["global"]["room_dims"]
    rt60 = x["metadata"]["global"].get("rt60", None)

    if mic_coords.ndim == 4:
        mic_coords = mic_coords[:, 0]

    batch_size, n_mics, _ = signals.shape
    device = signals.device
    edge_index = _build_complete_directed_edge_index(n_mics, device)
    n_edges = edge_index.shape[1]

    edge_attr = _compute_slf_edge_attr(
        signals,
        mic_coords,
        room_dims,
        edge_index,
        sr=sr,
        n_grid_points=n_grid_points,
        n_correlation_neighbours=n_correlation_neighbours,
        normalize=normalize_slf,
    )

    if is_metadata_aware:
        edge_metadata = _compute_edge_metadata(
            mic_coords,
            room_dims,
            rt60,
            edge_index,
            metadata_dim=metadata_dim,
            use_rt60_as_metadata=use_rt60_as_metadata,
        )
        edge_attr = torch.cat([edge_attr, edge_metadata], dim=1)

    offsets = (torch.arange(batch_size, device=device) * n_mics).repeat_interleave(n_edges)
    edge_index = edge_index.repeat(1, batch_size) + offsets

    return Data(
        x=torch.zeros((batch_size * n_mics, 1), device=device, dtype=signals.dtype),
        edge_index=edge_index,
        edge_attr=edge_attr,
        array=mic_coords.reshape(batch_size * n_mics, -1)[:, :3],
        batch=torch.arange(batch_size, device=device).repeat_interleave(n_mics),
    )
