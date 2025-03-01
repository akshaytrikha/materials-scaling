# External
import torch
import torch.nn as nn
from functools import partial

# Internal
from models.gaunt_equiformer_v2.fast_equiformer_v2 import (
    FastEquiformerV2Backbone,
    FastEquiformerV2EnergyHead,
    FastEquiformerV2ForceHead,
)
from models.gaunt_equiformer_v2.fast_attention import FastSO2EquivariantGraphAttention


class MLPReadout(nn.Module):
    """A simple MLP readout: node_emb -> linear -> tanh -> linear -> out_dim."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, node_emb):
        return self.net(node_emb)


class FastEquiformerV2StressHead(nn.Module):
    """
    Fast implementation of a stress head using Gaunt Tensor Products.
    """

    def __init__(self, backbone):
        """
        Initialize with the backbone to access its configuration
        """
        super().__init__()
        self.activation_checkpoint = backbone.activation_checkpoint

        # Create an attention block similar to the force head but for stress (6 components)
        self.stress_block = FastSO2EquivariantGraphAttention(
            backbone.sphere_channels,
            backbone.attn_hidden_channels,
            backbone.num_heads,
            backbone.attn_alpha_channels,
            backbone.attn_value_channels,
            6,  # 6 stress components instead of 1 for energy or 3 for forces
            backbone.lmax_list,
            backbone.mmax_list,
            backbone.SO3_rotation,
            backbone.mappingReduced,
            backbone.SO3_grid,
            backbone.max_num_elements,
            backbone.edge_channels_list,
            backbone.block_use_atom_edge_embedding,
            backbone.use_m_share_rad,
            backbone.attn_activation,
            backbone.use_s2_act_attn,
            backbone.use_attn_renorm,
            backbone.use_gate_act,
            backbone.use_sep_s2_act,
            alpha_drop=0.0,
        )

        # Import this here to avoid circular imports
        from fairchem.core.models.equiformer_v2.equiformer_v2 import eqv2_init_weights

        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, data, emb):
        """Forward pass of the stress head"""
        if self.activation_checkpoint:
            stress = torch.utils.checkpoint.checkpoint(
                self.stress_block,
                emb["node_embedding"],
                emb["graph"].atomic_numbers_full,
                emb["graph"].edge_distance,
                emb["graph"].edge_index,
                emb["graph"].node_offset,
                use_reentrant=not self.training,
            )
        else:
            stress = self.stress_block(
                emb["node_embedding"],
                emb["graph"].atomic_numbers_full,
                emb["graph"].edge_distance,
                emb["graph"].edge_index,
                node_offset=emb["graph"].node_offset,
            )

        # Extract the 6 stress components (skip the l=0 component which is at index 0)
        stress = stress.embedding.narrow(1, 1, 6)
        stress = stress.view(-1, 6).contiguous()

        # Handle graph parallel if needed
        from fairchem.core.common import gp_utils

        if gp_utils.initialized():
            stress = gp_utils.gather_from_model_parallel_region(stress, dim=0)

        # Sum stress per structure using scatter
        structure_index = emb["graph"].batch_full
        stress_per_structure = torch.zeros(
            len(data.natoms),
            6,
            device=stress.device,
            dtype=stress.dtype,
        )
        stress_per_structure.index_add_(0, data.batch, stress)

        return {"stress": stress_per_structure}


class EquiformerS2EF(nn.Module):
    """Wraps a FastEquiformerV2 backbone and specialized heads for energy, forces, and stress."""

    def __init__(self, backbone_config):
        super().__init__()
        # 1) Create accelerated backbone
        self.backbone = FastEquiformerV2Backbone(**backbone_config)
        self.name = "EquiformerV2"

        # 2) Use specialized accelerated heads for all outputs
        self.energy_head = FastEquiformerV2EnergyHead(backbone=self.backbone)
        self.force_head = FastEquiformerV2ForceHead(backbone=self.backbone)
        self.stress_head = FastEquiformerV2StressHead(backbone=self.backbone)

        # Calculate number of parameters
        self.num_params = sum(
            p.numel() for name, p in self.named_parameters() if "embedding" not in name
        )

    def forward(self, batch):
        """
        Args:
          batch: A PyG batch with .atomic_numbers, .pos, etc.
        Returns:
          forces  -> [N_atoms, 3]
          energy  -> [N_structures]
          stress  -> [N_structures, 6]
        """
        # Get embeddings from backbone
        emb = self.backbone(batch)

        # Use specialized energy head
        energy_output = self.energy_head(batch, emb)
        energy = energy_output["energy"]

        # Use specialized force head
        force_output = self.force_head(batch, emb)
        forces = force_output["forces"]

        # Use specialized stress head
        stress_output = self.stress_head(batch, emb)
        stress = stress_output["stress"]

        return forces, energy, stress


class MetaEquiformerV2Models:
    """
    Meta-class enumerating different EquiformerV2Backbone configurations.
    Now uses the Gaunt-accelerated implementation.
    """

    def __init__(self, device: torch.device):
        # The configurations remain the same - they're just architecture parameters
        self.configurations = [
            # 1,301 params
            {
                "regress_forces": True,
                "use_pbc": True,
                "use_pbc_single": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 119,
                "num_layers": 1,
                "sphere_channels": 1,
                "attn_hidden_channels": 1,
                "num_heads": 1,
                "attn_alpha_channels": 1,
                "attn_value_channels": 1,
                "ffn_hidden_channels": 2,
                "norm_type": "layer_norm_sh",
                "lmax_list": [0],
                "mmax_list": [0],
                "grid_resolution": 8,
                "num_sphere_samples": 16,
                "edge_channels": 2,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": True,
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 4,
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
            }
        ]

        self.device = device

    def __getitem__(self, idx: int) -> EquiformerS2EF:
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range")
        config = self.configurations[idx]
        model = EquiformerS2EF(config)
        model.to(self.device)
        return model

    def __len__(self) -> int:
        return len(self.configurations)

    def __iter__(self):
        for idx in range(len(self.configurations)):
            yield self[idx]
