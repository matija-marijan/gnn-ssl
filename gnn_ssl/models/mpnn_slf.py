"""Message-passing spatial-likelihood model."""

from omegaconf import OmegaConf

from .mpnn_gcc import MPNN_GCC


class MPNN_SLF(MPNN_GCC):
    """MPNN-GCC architecture using pairwise SLF edge features."""

    def __init__(self, config, **kwargs):
        config = OmegaConf.to_container(config, resolve=False)
        config["edge_feature_type"] = "slf"
        # Keep the relation branch identical to MPNN-GCC. The only model-level
        # substitution is the signal-derived edge feature.
        config["pair_skip"] = config.get("pair_skip", "raw")
        super().__init__(config, **kwargs)
