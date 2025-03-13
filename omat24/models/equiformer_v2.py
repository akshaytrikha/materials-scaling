# External import
import torch
import torch.nn as nn
from fairchem.core.common.registry import registry
from fairchem.core.models.base import HydraModel
from fairchem.core.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone

# Internal
from models.model_utils import initialize_eqV2_output_heads


class EquiformerS2EFS(nn.Module):
    """Model that combines an EquiformerV2 backbone with specialized heads for energy, forces, and stress prediction, following fairchem's configuration structure."""

    def __init__(self, config: dict, **kwargs):
        # Initialize the base class
        super().__init__()
        self.name = "EquiformerV2"

        # Register the backbone directly
        registry.register("equiformer_v2_backbone", EquiformerV2Backbone)

        # Create a copy of the config and modify the backbone model name
        backbone_config = config.get("backbone", {}).copy()
        backbone_config["model"] = "equiformer_v2_backbone"  # Use the registered model

        # Create HydraModel with our modified config
        self.hydra_model = HydraModel(
            backbone=backbone_config,
            heads=config.get("heads"),
            otf_graph=config.get("otf_graph", True),
            pass_through_head_outputs=config.get("pass_through_head_outputs", True),
        )

        # Extract head references for convenience
        self.backbone = self.hydra_model.backbone
        self.output_heads = self.hydra_model.output_heads
        self.energy_head = self.output_heads["energy"]
        self.force_head = self.output_heads["forces"]
        self.stress_head = self.output_heads["stress"]

        # Initialize the output heads
        initialize_eqV2_output_heads(
            self.energy_head,
            self.force_head,
            self.stress_head,
            per_atom=True,
        )

        # Calculate number of parameters
        self.num_params = sum(
            p.numel() for name, p in self.named_parameters() if "embedding" not in name
        )

    def merge_stress_components(self, stress_isotropic, stress_anisotropic):
        """
        Merge isotropic and anisotropic stress components into a single tensor
        so that EqV2 outputs are consistent with the other architectures.
        """
        batch_size = stress_isotropic.shape[0]
        # Create the full stress tensor in Voigt notation
        voigt_stress = torch.zeros((batch_size, 6), device=stress_isotropic.device)
        # Fill in the isotropic part (pressure) for the normal stress components
        voigt_stress[:, 0] = stress_isotropic.squeeze()  # σxx
        voigt_stress[:, 1] = stress_isotropic.squeeze()  # σyy
        voigt_stress[:, 2] = stress_isotropic.squeeze()  # σzz
        # Fill in the anisotropic components
        if stress_anisotropic.shape[1] == 5:
            voigt_stress[:, :5] += stress_anisotropic  # Add first 5 components
        return voigt_stress

    def forward(self, batch):
        """
        Args:
            batch: A PyG batch with .atomic_numbers, .pos, etc.
        Returns:
            Tuple of (forces, energy, stress) tensors that are directly usable for loss computation
        """
        # Forward through the hydra model
        outputs = self.hydra_model(batch)

        # Extract the actual tensors from the output dictionaries
        forces = outputs["forces"]
        energy = outputs["energy"]
        iso_stress = outputs["stress_isotropic"]
        aniso_stress = outputs["stress_anisotropic"]

        # Merge stress components
        stress = self.merge_stress_components(iso_stress, aniso_stress)

        return forces, energy, stress

    # Delegate methods to hydra_model for proper device handling
    def to(self, *args, **kwargs):
        self.hydra_model = self.hydra_model.to(*args, **kwargs)
        return self

    def named_parameters(self, *args, **kwargs):
        return self.hydra_model.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.hydra_model.parameters(*args, **kwargs)


class MetaEquiformerV2Models:
    """
    Meta-class for creating EquiformerS2EF models with different configurations.
    Follows fairchem's hydra-style configuration.
    """

    def __init__(self, device: torch.device):
        # Base configuration that matches fairchem's configuration from YAML
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
                "avg_num_nodes": 31.17,
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
                "energy": {"module": "equiformer_v2_energy_head"},
                "forces": {"module": "equiformer_v2_force_head"},
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

    def __getitem__(self, idx: int) -> EquiformerS2EFS:
        if idx >= len(self.configs):
            raise IndexError("Configuration index out of range")

        config = self.base_config.copy()
        for key, value in self.configs[idx]["backbone"].items():
            config["backbone"][key] = value

        model = EquiformerS2EFS(config)
        model.to(self.device)
        return model

    def get_luis_model(self, idx: int) -> EquiformerS2EFS:
        """Get one of the model configurations from the paper"""
        if idx >= len(self.luis_configs):
            raise IndexError("Configuration index out of range")

        config = self.base_config.copy()
        for key, value in self.luis_configs[idx]["backbone"].items():
            config["backbone"][key] = value

        model = EquiformerS2EFS(config)
        model.to(self.device)
        return model

    def __len__(self) -> int:
        return len(self.configs)

    def __iter__(self):
        for idx in range(len(self.configs)):
            yield self[idx]
