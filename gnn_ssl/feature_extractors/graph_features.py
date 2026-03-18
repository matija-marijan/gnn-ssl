import torch

from torch_geometric.data import Batch, Data

from pysoundloc.pysoundloc.gcc import gcc_phat_batch


def _build_complete_directed_edge_index(n_nodes, device):
    src = []
    dst = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            src.append(i)
            dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return edge_index


def _compute_edge_attr(signals, edge_index, sr, n_edge_bins):
    src = edge_index[0]
    dst = edge_index[1]

    per_edge_features = []
    n_signal = signals.shape[1]

    for i, j in zip(src.tolist(), dst.tolist()):
        gcc = gcc_phat_batch(
            signals[i].unsqueeze(0),
            signals[j].unsqueeze(0),
            sr,
            abs=True,
        )[0]
        start = (n_signal - n_edge_bins) // 2
        end = start + n_edge_bins
        per_edge_features.append(gcc[0, start:end])

    return torch.stack(per_edge_features, dim=0)


def _compute_edge_metadata(coords, room_dims, rt60, edge_index, metadata_dim, use_rt60_as_metadata):
    src = edge_index[0]
    dst = edge_index[1]

    per_edge_metadata = []
    for i, j in zip(src.tolist(), dst.tolist()):
        features = [
            coords[i, :metadata_dim],
            coords[j, :metadata_dim],
            room_dims[:metadata_dim],
        ]

        if use_rt60_as_metadata:
            if rt60 is None:
                features.append(coords.new_zeros((1,)))
            else:
                features.append(rt60.reshape(-1))

        per_edge_metadata.append(torch.cat(features, dim=0))

    return torch.stack(per_edge_metadata, dim=0)


def build_graph_batch_from_input(
    x,
    sr,
    n_edge_bins,
    is_metadata_aware=True,
    use_rt60_as_metadata=False,
    metadata_dim=3,
):
    signals = x["signal"]
    mic_coords = x["metadata"]["local"]["mic_coordinates"]
    room_dims = x["metadata"]["global"]["room_dims"]
    rt60 = x["metadata"]["global"].get("rt60", None)

    if mic_coords.ndim == 4:
        mic_coords = mic_coords[:, 0]

    batch_size, n_mics, _ = signals.shape
    data_list = []

    for b in range(batch_size):
        sig_b = signals[b]
        coords_b = mic_coords[b]
        room_dims_b = room_dims[b]
        rt60_b = rt60[b] if rt60 is not None else None

        edge_index = _build_complete_directed_edge_index(n_mics, sig_b.device)
        edge_attr = _compute_edge_attr(sig_b, edge_index, sr=sr, n_edge_bins=n_edge_bins)

        if is_metadata_aware:
            edge_metadata = _compute_edge_metadata(
                coords_b,
                room_dims_b,
                rt60_b,
                edge_index,
                metadata_dim=metadata_dim,
                use_rt60_as_metadata=use_rt60_as_metadata,
            )
            edge_attr = torch.cat([edge_attr, edge_metadata], dim=1)

        data_list.append(
            Data(
                x=torch.zeros((n_mics, 1), device=sig_b.device, dtype=sig_b.dtype),
                edge_index=edge_index,
                edge_attr=edge_attr,
                array=coords_b[:, :3],
            )
        )

    return Batch.from_data_list(data_list)
