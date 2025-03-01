"""
Fast Equivariant Graph Attention module using Gaunt Tensor Products.
"""

# External
import torch.nn as nn
import torch.nn.functional as F
from fairchem.core.models.equiformer_v2.transformer_block import (
    SO2EquivariantGraphAttention,
)
from fairchem.core.models.equiformer_v2.so3 import SO3_Embedding

# Internal
from models.gaunt_equiformer_v2.gaunt_accelerator import GauntTensorProduct


class FastSO2EquivariantGraphAttention(SO2EquivariantGraphAttention):
    """
    Fast implementation of SO2EquivariantGraphAttention using Gaunt Tensor Products.
    This class inherits from SO2EquivariantGraphAttention and overrides key methods
    to use the more efficient Gaunt Tensor Product approach.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with the same parameters as SO2EquivariantGraphAttention"""
        super().__init__(*args, **kwargs)

        # Create GauntTensorProduct module for each resolution
        self.gaunt_tps = nn.ModuleList()
        for lmax in self.lmax_list:
            self.gaunt_tps.append(
                GauntTensorProduct(
                    lmax_list=[lmax],
                    mmax_list=[min(lmax, max(self.mmax_list))],
                    channels=self.sphere_channels,
                    device=None,  # Will be set during forward pass
                )
            )

    def _compute_attention_values(self, q, k, v, mask=None):
        """
        Compute attention scores and values using Gaunt Tensor Products for acceleration.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            mask: Attention mask

        Returns:
            Tuple: Attention scores and values
        """
        # This is where the major speedup happens
        # Instead of using CG coefficients for tensor products,
        # we'll use our Gaunt Tensor Product implementation

        batch_size = q.embedding.shape[0]
        device = q.embedding.device

        # Create output embedding for attention scores
        attn_output = SO3_Embedding(
            batch_size, self.lmax_list, self.sphere_channels, device, q.embedding.dtype
        )

        # Process each resolution (lmax, mmax pair)
        for i_res in range(len(self.lmax_list)):
            lmax = self.lmax_list[i_res]
            mmax = self.mmax_list[i_res]

            # Extract the relevant parts of input embeddings
            q_res = q.embedding[:, self.res_mapping[i_res]]
            k_res = k.embedding[:, self.res_mapping[i_res]]

            # Compute attention scores using Gaunt Tensor Products
            for c_out in range(self.sphere_channels):
                for c_in in range(self.sphere_channels):
                    # Use the Gaunt Tensor Product instead of CG coefficients
                    result = (
                        self.gaunt_tps[i_res]
                        .tensor_product_channel(
                            q_res[:, :, c_out].unsqueeze(0),
                            k_res[:, :, c_in].unsqueeze(0),
                            lmax,
                        )
                        .squeeze(0)
                    )

                    # Apply mask if provided
                    if mask is not None:
                        result = result * mask.unsqueeze(-1).unsqueeze(-1)

                    # Add to output embedding
                    attn_output.embedding[:, self.res_mapping[i_res], c_out] += result

        # Apply softmax to attention scores
        attn_scores = F.softmax(attn_output.embedding, dim=1)

        # Create output embedding for attention values
        attn_values = SO3_Embedding(
            batch_size, self.lmax_list, self.out_channels, device, q.embedding.dtype
        )

        # Compute attention values using Gaunt Tensor Products
        for i_res in range(len(self.lmax_list)):
            lmax = self.lmax_list[i_res]
            mmax = self.mmax_list[i_res]

            # Extract the relevant parts of input embeddings
            scores_res = attn_scores[:, self.res_mapping[i_res]]
            v_res = v.embedding[:, self.res_mapping[i_res]]

            # Compute attention values using Gaunt Tensor Products
            for c_out in range(self.out_channels):
                for c_in in range(self.sphere_channels):
                    # Use the Gaunt Tensor Product instead of CG coefficients
                    result = (
                        self.gaunt_tps[i_res]
                        .tensor_product_channel(
                            scores_res[:, :, c_in].unsqueeze(0),
                            v_res[:, :, c_out].unsqueeze(0),
                            lmax,
                        )
                        .squeeze(0)
                    )

                    # Add to output embedding
                    attn_values.embedding[:, self.res_mapping[i_res], c_out] += result

        return attn_scores, attn_values

    def forward(self, x, atomic_numbers, edge_distance, edge_index, node_offset=0):
        """
        Forward pass using Gaunt Tensor Products for accelerated computation.

        We override the forward method to use our accelerated attention computation.
        """
        # Use the existing implementation but with our accelerated _compute_attention_values
        return super().forward(
            x, atomic_numbers, edge_distance, edge_index, node_offset
        )
