"""
Fast Feed-Forward Network using Gaunt Tensor Products.
"""

# External
import torch.nn as nn
from fairchem.core.models.equiformer_v2.transformer_block import (
    FeedForwardNetwork,
    TransBlockV2,
)
from fairchem.core.models.equiformer_v2.so3 import SO3_Embedding

# Internal
from models.gaunt_equiformer_v2.gaunt_accelerator import GauntTensorProduct
from models.gaunt_equiformer_v2.fast_attention import FastSO2EquivariantGraphAttention


class FastFeedForwardNetwork(FeedForwardNetwork):
    """
    Fast implementation of FeedForwardNetwork using Gaunt Tensor Products.
    This class inherits from FeedForwardNetwork and overrides key methods
    to use the more efficient Gaunt Tensor Product approach.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with the same parameters as FeedForwardNetwork"""
        super().__init__(*args, **kwargs)

        # Create GauntTensorProduct module for each resolution
        self.gaunt_tps = nn.ModuleList()
        for lmax in self.lmax_list:
            self.gaunt_tps.append(
                GauntTensorProduct(
                    lmax_list=[lmax],
                    mmax_list=[min(lmax, max(self.mmax_list))],
                    channels=self.hidden_channels,
                    device=None,  # Will be set during forward pass
                )
            )

    def _apply_activation(self, x, activation_fn):
        """
        Apply activation function using Gaunt Tensor Products for tensor products.

        Args:
            x: Input tensor
            activation_fn: Activation function to apply

        Returns:
            SO3_Embedding: Output after applying activation
        """
        # This is where we accelerate the tensor product operations
        batch_size = x.embedding.shape[0]
        device = x.embedding.device

        # Apply activation based on the selected type
        if self.use_gate_act:
            # For gate activation, we need to split the input
            # Use the original implementation for simplicity
            return super()._apply_activation(x, activation_fn)

        elif self.use_grid_mlp:
            # For grid MLP, we need to convert to grid representation
            # Use the original implementation for simplicity
            return super()._apply_activation(x, activation_fn)

        else:
            # For regular S2 activation or separable S2 activation,
            # we can use our Gaunt Tensor Products

            # Create output embedding
            output = SO3_Embedding(
                batch_size,
                self.lmax_list,
                self.sphere_channels,
                device,
                x.embedding.dtype,
            )

            # Process each resolution (lmax, mmax pair)
            for i_res in range(len(self.lmax_list)):
                lmax = self.lmax_list[i_res]
                mmax = self.mmax_list[i_res]

                # Extract the relevant parts of input embedding
                x_res = x.embedding[:, self.res_mapping[i_res]]

                if self.use_sep_s2_act:
                    # Separable S2 activation - apply activation to each component
                    activated = activation_fn(x_res)
                    output.embedding[:, self.res_mapping[i_res]] = activated
                else:
                    # Regular S2 activation - apply tensor products
                    for c_out in range(self.sphere_channels):
                        for c_in in range(self.sphere_channels):
                            # Use the Gaunt Tensor Product for acceleration
                            result = (
                                self.gaunt_tps[i_res]
                                .tensor_product_channel(
                                    x_res[:, :, c_out].unsqueeze(0),
                                    x_res[:, :, c_in].unsqueeze(0),
                                    lmax,
                                )
                                .squeeze(0)
                            )

                            # Apply activation
                            result = activation_fn(result)

                            # Add to output embedding
                            output.embedding[
                                :, self.res_mapping[i_res], c_out
                            ] += result

            return output

    def forward(self, x):
        """
        Forward pass using Gaunt Tensor Products for accelerated computation.

        We can use the parent class implementation since we've overridden
        the _apply_activation method.
        """
        return super().forward(x)


class FastTransBlockV2(TransBlockV2):
    """
    Fast implementation of TransBlockV2 that uses our accelerated components.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with the same parameters but use our fast implementations"""
        # Initialize with parent class to set up all parameters
        super().__init__(*args, **kwargs)

        # Replace the attention module with our fast version
        self.attn = FastSO2EquivariantGraphAttention(
            self.sphere_channels,
            self.attn_hidden_channels,
            self.num_heads,
            self.attn_alpha_channels,
            self.attn_value_channels,
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.SO3_grid,
            self.max_num_elements,
            self.edge_channels_list,
            self.use_atom_edge_embedding,
            self.use_m_share_rad,
            self.attn_activation,
            self.use_s2_act_attn,
            self.use_attn_renorm,
            self.use_gate_act,
            self.use_sep_s2_act,
            self.alpha_drop,
        )

        # Replace the feed-forward network with our fast version
        self.ffn = FastFeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )
