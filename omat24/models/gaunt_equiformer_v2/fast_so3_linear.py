"""
Fast SO3 Linear module using Gaunt Tensor Products.
This module replaces the standard SO3_LinearV2 with a faster implementation.
"""

# External
import torch.nn as nn
from fairchem.core.models.equiformer_v2.so3 import SO3_LinearV2, SO3_Embedding

# Internal
from models.gaunt_equiformer_v2.gaunt_accelerator import GauntTensorProduct


class FastSO3Linear(SO3_LinearV2):
    """
    Fast implementation of SO3_LinearV2 using Gaunt Tensor Products.
    This class inherits from SO3_LinearV2 and overrides the forward method
    to use the more efficient Gaunt Tensor Product approach.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with the same parameters as SO3_LinearV2"""
        super().__init__(*args, **kwargs)

        # Create GauntTensorProduct module
        self.gaunt_tp = GauntTensorProduct(
            lmax_list=self.lmax_list,
            mmax_list=self.mmax_list,
            channels=self.sphere_channels,
            device=self.weight.device,
        )

    def forward(self, x: SO3_Embedding) -> SO3_Embedding:
        """
        Forward pass using Gaunt Tensor Products for accelerated computation.

        Args:
            x: Input SO3_Embedding

        Returns:
            SO3_Embedding: Output embedding after linear transformation
        """
        # Get device and input shape
        device = x.embedding.device
        batch_size = x.embedding.shape[0]

        # Create output embedding
        output = SO3_Embedding(
            batch_size, self.lmax_list_out, self.out_features, device, x.embedding.dtype
        )

        # Process each resolution (lmax, mmax pair)
        for i_res_out in range(len(self.lmax_list_out)):
            lmax_out = self.lmax_list_out[i_res_out]
            mmax_out = self.mmax_list_out[i_res_out]

            # Process each input resolution
            for i_res_in in range(len(self.lmax_list)):
                lmax_in = self.lmax_list[i_res_in]
                mmax_in = self.mmax_list[i_res_in]

                # Extract the relevant parts of input embedding
                x_subset = x.embedding[:, self.res_input_mapping[i_res_in]]

                # Extract the weight for this transformation
                weight_subset = self.weight[
                    self.res_output_mapping[i_res_out],
                    :,
                    self.res_input_mapping[i_res_in],
                ]

                # Use Gaunt Tensor Product for tensor product computation
                # This is where the speedup happens!
                for c_out in range(self.out_features):
                    for c_in in range(self.in_features):
                        # Extract weights for this channel pair
                        weight_channel = weight_subset[c_out, :, c_in]

                        # Extract input for this channel
                        x_channel = x_subset[:, :, c_in]

                        # Perform tensor product using Gaunt approach
                        # Select the appropriate lmax to use
                        lmax_to_use = min(lmax_in, lmax_out)
                        result = self.gaunt_tp.tensor_product_channel(
                            x_channel.unsqueeze(0),
                            weight_channel.unsqueeze(0),
                            lmax_to_use,
                        ).squeeze(0)

                        # Add to output embedding
                        output.embedding[
                            :, self.res_output_mapping[i_res_out], c_out
                        ] += result

        if self.bias is not None:
            # Add bias to output
            for i_res_out in range(len(self.lmax_list_out)):
                l_start_out = self.l_start_out[i_res_out]
                l_end_out = self.l_end_out[i_res_out]

                for l in range(l_start_out, l_end_out + 1):
                    idx_bias = self.get_bias_idx(i_res_out, l)

                    output.embedding[
                        :,
                        self.res_output_mapping[i_res_out][
                            self.irreps_mapping[l][0] : self.irreps_mapping[l][1]
                        ],
                        :,
                    ] += self.bias[idx_bias].unsqueeze(0)

        return output
