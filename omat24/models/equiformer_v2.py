import torch
import torch.nn as nn
from fairchem.core.common.registry import registry
from fairchem.core.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone
from fairchem.core.models.equiformer_v2.heads import EqV2ScalarHead, EqV2VectorHead
from fairchem.core.models.equiformer_v2.heads.rank2 import Rank2SymmetricTensorHead
from fairchem.core.models.base import GraphModelMixin, HeadInterface


@registry.register_model("hydra")
class EquiformerS2EF(nn.Module, GraphModelMixin):
    """
    Hydra-style model that combines an EquiformerV2 backbone with specialized heads
    for energy, forces, and stress prediction, following fairchem's configuration structure.
    """

    def __init__(self, backbone, heads, pass_through_head_outputs=True, **kwargs):
        super().__init__()

        # Instantiate the backbone
        if isinstance(backbone, dict):
            model_type = backbone.pop("model")
            self.backbone = registry.get_model_class(model_type)(**backbone)
        else:
            self.backbone = backbone

        # Set up output heads
        self.output_heads = nn.ModuleDict()
        self.pass_through_head_outputs = pass_through_head_outputs

        for name, head_config in heads.items():
            module_type = head_config.pop("module")
            head_class = registry.get_model_class(module_type)
            self.output_heads[name] = head_class(backbone=self.backbone, **head_config)

        # Include kwargs to make the model compatible with fairchem's initialization
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Calculate number of parameters
        self.num_params = sum(
            p.numel() for name, p in self.named_parameters() if "embedding" not in name
        )

    def forward(self, batch):
        """
        Args:
          batch: A PyG batch with .atomic_numbers, .pos, etc.
        Returns:
          Dict with model outputs (energy, forces, stress, etc.)
        """
        # Get embeddings from backbone
        emb = self.backbone(batch)

        # Apply all output heads
        outputs = {}
        for name, head in self.output_heads.items():
            head_outputs = head(batch, emb)
            outputs.update(head_outputs)

        return outputs


class MetaEquiformerV2Models:
    """
    Meta-class for creating EquiformerS2EF models with different configurations.
    Follows fairchem's hydra-style configuration.
    """

    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        # Base configuration that matches fairchem's configuration
        self.base_config = {
            "pass_through_head_outputs": True,
            "otf_graph": True,
            "backbone": {
                "model": "equiformer_v2_backbone",
                "use_pbc": True,
                "use_pbc_single": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 96,
                "avg_num_nodes": 31.17,  # This is where the 31.17 comes from
                "avg_degree": 61.95,
                "sphere_channels": 128,
                "attn_hidden_channels": 64,
                "num_heads": 8,
                "attn_alpha_channels": 64,
                "attn_value_channels": 16,
                "ffn_hidden_channels": 128,
                "norm_type": "layer_norm_sh",
                "lmax_list": [4],
                "mmax_list": [2],
                "grid_resolution": 18,
                "num_sphere_samples": 128,
                "edge_channels": 128,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": False,
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 512,
                "attn_activation": "silu",
                "use_s2_act_attn": False,
                "use_attn_renorm": True,
                "ffn_activation": "silu",
                "use_gate_act": False,
                "use_grid_mlp": True,
                "use_sep_s2_act": True,
                "alpha_drop": 0.1,
                "drop_path_rate": 0.1,
                "proj_drop": 0.0,
                "weight_init": "uniform",
            },
            "heads": {
                "energy": {
                    "module": "equiformerV2_scalar_head",
                    "output_name": "energy",
                    "reduce": "sum",
                },
                "forces": {
                    "module": "equiformerV2_vector_head",
                    "output_name": "forces",
                },
                "stress": {
                    "module": "rank2_symmetric_head",
                    "output_name": "stress",
                    "use_source_target_embedding": True,
                    "decompose": True,
                },
            },
        }

        self.configs = [
            # ~3K params
            {
                "backbone": {
                    "ffn_hidden_channels": 1,
                    "edge_channels": 1,
                    "sphere_channels": 1,
                    "num_layers": 1,
                    "attn_hidden_channels": 1,
                    "num_heads": 1,
                    "attn_alpha_channels": 1,
                    "attn_value_channels": 1,
                }
            },
            # ~13K params
            {
                "backbone": {
                    "ffn_hidden_channels": 2,
                    "edge_channels": 2,
                    "sphere_channels": 2,
                    "num_layers": 2,
                    "attn_hidden_channels": 2,
                    "num_heads": 1,
                    "attn_alpha_channels": 1,
                    "attn_value_channels": 1,
                }
            },
            # ~95K params
            {
                "backbone": {
                    "ffn_hidden_channels": 5,
                    "edge_channels": 5,
                    "sphere_channels": 5,
                    "num_layers": 5,
                    "attn_hidden_channels": 5,
                    "num_heads": 2,
                    "attn_alpha_channels": 3,
                    "attn_value_channels": 2,
                }
            },
            # ~1M params
            {
                "backbone": {
                    "ffn_hidden_channels": 16,
                    "edge_channels": 16,
                    "sphere_channels": 16,
                    "num_layers": 10,
                    "attn_hidden_channels": 12,
                    "num_heads": 4,
                    "attn_alpha_channels": 8,
                    "attn_value_channels": 4,
                }
            },
            # ~10M params
            {
                "backbone": {
                    "ffn_hidden_channels": 48,
                    "edge_channels": 36,
                    "sphere_channels": 36,
                    "num_layers": 20,
                    "attn_hidden_channels": 48,
                    "num_heads": 6,
                    "attn_alpha_channels": 24,
                    "attn_value_channels": 12,
                }
            },
            # ~98M params
            {
                "backbone": {
                    "ffn_hidden_channels": 96,
                    "edge_channels": 96,
                    "sphere_channels": 96,
                    "num_layers": 40,
                    "attn_hidden_channels": 96,
                    "num_heads": 12,
                    "attn_alpha_channels": 48,
                    "attn_value_channels": 24,
                }
            },
            # ~244M params
            {
                "backbone": {
                    "ffn_hidden_channels": 128,
                    "edge_channels": 128,
                    "sphere_channels": 128,
                    "num_layers": 60,
                    "attn_hidden_channels": 128,
                    "num_heads": 16,
                    "attn_alpha_channels": 64,
                    "attn_value_channels": 32,
                }
            },
        ]

        # From the Open Materials 2024 Paper
        self.luis_configs = [
            # eqV2-S: 31M params
            {
                "backbone": {
                    "num_layers": 8,
                    "lmax_list": [4],
                    "mmax_list": [2],
                    "share_atom_edge_embedding": False,
                    "use_m_share_rad": False,
                    "use_attn_renorm": True,
                    "use_sep_s2_act": True,
                }
            },
            # eqV2-M: 86M params
            {
                "backbone": {
                    "num_layers": 10,
                    "lmax_list": [6],
                    "mmax_list": [4],
                    "share_atom_edge_embedding": False,
                    "use_m_share_rad": False,
                    "use_attn_renorm": True,
                    "use_sep_s2_act": True,
                }
            },
            # eqV2-L: 153M params
            {"backbone": {"num_layers": 20, "lmax_list": [6], "mmax_list": [3]}},
        ]

        self.device = device

    def __getitem__(self, idx: int) -> EquiformerS2EF:
        if idx >= len(self.configs):
            raise IndexError("Configuration index out of range")

        config = self.base_config.copy()
        for key, value in self.configs[idx]["backbone"].items():
            config["backbone"][key] = value

        model = EquiformerS2EF(**config)
        model.to(self.device)
        return model

    def get_paper_model(self, idx: int) -> EquiformerS2EF:
        """Get one of the model configurations from the paper"""
        if idx >= len(self.luis_configs):
            raise IndexError("Configuration index out of range")

        config = self.base_config.copy()
        for key, value in self.luis_configs[idx]["backbone"].items():
            config["backbone"][key] = value

        model = EquiformerS2EF(**config)
        model.to(self.device)
        return model

    def __len__(self) -> int:
        return len(self.configs)

    def __iter__(self):
        for idx in range(len(self.configs)):
            yield self[idx]
