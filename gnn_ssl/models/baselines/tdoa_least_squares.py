import torch.nn as nn

from pysoundloc.pysoundloc.models import tdoa_least_squares_ssl


class TdoaLeastSquares(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.targets_config = config["targets"]
        self.features_config = config["features"]
        self.n_points_per_axis = self.targets_config["n_points_per_axis"]
        self.sr = self.features_config["dataset"]["sr"]

    def forward(self, x):
        mic_signals = x["signal"]
        mic_coords = x["metadata"]["local"]["mic_coordinates"][..., :2]
        room_dims = x["metadata"]["global"]["room_dims"][..., :2]

        source_coordinates, grid = tdoa_least_squares_ssl(
            mic_signals,
            mic_coords,
            room_dims,
            self.sr,
            self.n_points_per_axis,
        )

        return {
            "source_coordinates": source_coordinates,
            "grid": grid,
        }
