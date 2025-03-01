"""
Fast EquiformerV2 implementation using Gaunt Tensor Products.
This module provides a drop-in replacement for EquiformerV2Backbone
that uses the more efficient Gaunt Tensor Product approach.
"""

import torch
import torch.nn as nn
from functools import partial

from fairchem.core.common.registry import registry
from fairchem.core.models.equiformer_v2.equiformer_v2 import (
    EquiformerV2Backbone,
    eqv2_init_weights,
)
from models.gaunt_equiformer_v2.fast_attention import FastSO2EquivariantGraphAttention
from models.gaunt_equiformer_v2.fast_ffn import FastFeedForwardNetwork, FastTransBlockV2


@registry.register_model("fast_equiformer_v2_backbone")
class FastEquiformerV2Backbone(EquiformerV2Backbone):
    """
    Fast implementation of EquiformerV2Backbone using Gaunt Tensor Products.

    This class inherits from EquiformerV2Backbone and overrides the initialization
    to use our accelerated components.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize FastEquiformerV2Backbone with the same parameters as EquiformerV2Backbone
        but using our fast implementations for the critical components.
        """
        # Call the parent class initialization to set up all parameters
        super().__init__(*args, **kwargs)

        # Replace the transformer blocks with our fast versions
        fast_blocks = []
        for i in range(self.num_layers):
            fast_block = FastTransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
            )
            fast_blocks.append(fast_block)

        # Replace the blocks with our fast versions
        self.blocks = nn.ModuleList(fast_blocks)

        # Apply weight initialization
        self.apply(partial(eqv2_init_weights, weight_init=self.weight_init))


@registry.register_model("fast_equiformer_v2_energy_head")
class FastEquiformerV2EnergyHead(nn.Module):
    """
    Fast implementation of EquiformerV2EnergyHead using Gaunt Tensor Products.
    """

    def __init__(self, backbone, reduce: str = "sum"):
        """
        Initialize FastEquiformerV2EnergyHead with the same parameters
        but using our fast FFN implementation.
        """
        super().__init__()
        self.reduce = reduce
        self.avg_num_nodes = backbone.avg_num_nodes

        # Use our fast FFN implementation
        self.energy_block = FastFeedForwardNetwork(
            backbone.sphere_channels,
            backbone.ffn_hidden_channels,
            1,
            backbone.lmax_list,
            backbone.mmax_list,
            backbone.SO3_grid,
            backbone.ffn_activation,
            backbone.use_gate_act,
            backbone.use_grid_mlp,
            backbone.use_sep_s2_act,
        )

        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, data, emb):
        """Forward pass of the energy head"""
        node_energy = self.energy_block(emb["node_embedding"])
        node_energy = node_energy.embedding.narrow(1, 0, 1)

        # Handle graph parallel if needed
        from fairchem.core.common import gp_utils

        if gp_utils.initialized():
            node_energy = gp_utils.gather_from_model_parallel_region(node_energy, dim=0)

        energy = torch.zeros(
            len(data.natoms),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy.index_add_(0, data.batch, node_energy.view(-1))
        if self.reduce == "sum":
            return {"energy": energy / self.avg_num_nodes}
        elif self.reduce == "mean":
            return {"energy": energy / data.natoms}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )


@registry.register_model("fast_equiformer_v2_force_head")
class FastEquiformerV2ForceHead(nn.Module):
    """
    Fast implementation of EquiformerV2ForceHead using Gaunt Tensor Products.
    """

    def __init__(self, backbone):
        """
        Initialize FastEquiformerV2ForceHead with the same parameters
        but using our fast attention implementation.
        """
        super().__init__()

        self.activation_checkpoint = backbone.activation_checkpoint

        # Use our fast attention implementation
        self.force_block = FastSO2EquivariantGraphAttention(
            backbone.sphere_channels,
            backbone.attn_hidden_channels,
            backbone.num_heads,
            backbone.attn_alpha_channels,
            backbone.attn_value_channels,
            1,
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

        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, data, emb):
        """Forward pass of the force head"""
        if self.activation_checkpoint:
            forces = torch.utils.checkpoint.checkpoint(
                self.force_block,
                emb["node_embedding"],
                emb["graph"].atomic_numbers_full,
                emb["graph"].edge_distance,
                emb["graph"].edge_index,
                emb["graph"].node_offset,
                use_reentrant=not self.training,
            )
        else:
            forces = self.force_block(
                emb["node_embedding"],
                emb["graph"].atomic_numbers_full,
                emb["graph"].edge_distance,
                emb["graph"].edge_index,
                node_offset=emb["graph"].node_offset,
            )

        forces = forces.embedding.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()

        # Handle graph parallel if needed
        from fairchem.core.common import gp_utils

        if gp_utils.initialized():
            forces = gp_utils.gather_from_model_parallel_region(forces, dim=0)

        return {"forces": forces}
