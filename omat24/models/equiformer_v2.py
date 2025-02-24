import torch
from fairchem.core.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone


class MetaEquiformerV2Models:
    """Meta-class enumerating different EquiformerV2Backbone configurations."""

    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.configurations = [
            # # 1702 params
            # {
            #     "num_layers": 1,
            #     "sphere_channels": 1,
            #     "attn_hidden_channels": 1,
            #     "num_heads": 1,
            #     "attn_alpha_channels": 1,
            #     "attn_value_channels": 1,
            #     "ffn_hidden_channels": 1,
            #     "norm_type": "rms_norm_sh",
            #     "lmax_list": [0],
            #     "mmax_list": [0],
            #     "grid_resolution": None,
            #     "num_sphere_samples": 1,
            #     "edge_channels": 1,
            # },
            # 7M params
            {
                "num_layers": 12,
                "sphere_channels": 128,
                "attn_hidden_channels": 128,
                "num_heads": 8,
                "attn_alpha_channels": 32,
                "attn_value_channels": 16,
                "ffn_hidden_channels": 512,
                "norm_type": "rms_norm_sh",
                "lmax_list": [6],
                "mmax_list": [2],
                "grid_resolution": None,
                "num_sphere_samples": 128,
                "edge_channels": 128,
            },
            # # Luis 31M params
            # {
            #     "num_layers": 8,
            #     "sphere_channels": 128,
            #     "attn_hidden_channels": 64,
            #     "num_heads": 8,
            #     "attn_alpha_channels": 64,
            #     "attn_value_channels": 16,
            #     "ffn_hidden_channels": 128,
            #     "norm_type": "layer_norm_sh",
            #     "lmax_list": [4],
            #     "mmax_list": [2],
            #     "grid_resolution": 18,
            #     "num_sphere_samples": 128,
            #     "edge_channels": 128,
            # },
            # # Luis 86M params
            # {
            #     "num_layers": 10,
            #     "sphere_channels": 128,
            #     "attn_hidden_channels": 64,
            #     "num_heads": 8,
            #     "attn_alpha_channels": 64,
            #     "attn_value_channels": 16,
            #     "ffn_hidden_channels": 128,
            #     "norm_type": "layer_norm_sh",
            #     "lmax_list": [6],
            #     "mmax_list": [4],
            #     "grid_resolution": 18,
            #     "num_sphere_samples": 128,
            #     "edge_channels": 128,
            # },
            # # Luis 153M params
            # {
            #     "num_layers": 20,
            #     "sphere_channels": 128,
            #     "attn_hidden_channels": 64,
            #     "num_heads": 8,
            #     "attn_alpha_channels": 64,
            #     "attn_value_channels": 16,
            #     "ffn_hidden_channels": 128,
            #     "norm_type": "layer_norm_sh",
            #     "lmax_list": [6],
            #     "mmax_list": [3],
            #     "grid_resolution": 18,
            #     "num_sphere_samples": 128,
            #     "edge_channels": 128,
            # },
        ]
        self.device = device

    def __getitem__(self, idx: int) -> EquiformerV2Backbone:
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range")
        config = self.configurations[idx]
        # Instantiate the backbone using the given configuration.
        model = EquiformerV2Backbone(**config)
        model.to(self.device)
        return model

    def __len__(self) -> int:
        return len(self.configurations)

    def __iter__(self):
        for idx in range(len(self.configurations)):
            yield self[idx]
