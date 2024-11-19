# import torch
# import torch.nn as nn
# from x_transformers import TransformerWrapper, Encoder


# class XTransformerModel(torch.nn.Module):
#     def __init__(self, num_elements, d_model, n_layers, n_heads, d_ff_mult):
#         super().__init__()
#         # Atomic embedding for atomic numbers
#         self.atomic_emb = nn.Embedding(num_elements, d_model)

#         # Total input dimension to transformer
#         transformer_input_dim = d_model + 3

#         # Transformer definition
#         self.transformer = TransformerWrapper(
#             use_token_emb=False,  # Disable internal token embedding
#             attn_layers=Encoder(
#                 dim=transformer_input_dim,
#                 depth=n_layers,
#                 heads=n_heads,
#                 ff_mult=d_ff_mult,  # Set feedforward dimension multiplier
#             ),
#             use_abs_pos_emb=False,  # Disable absolute positional embeddings
#             use_rotary_pos_emb=False,  # Disable rotary positional embeddings
#         )

#         # Predictors for Energy, Forces, and Stresses
#         self.energy_predictor = nn.Linear(transformer_input_dim, 1)  # Energy: [M, 1]
#         self.forces_predictor = nn.Linear(transformer_input_dim, 3)  # Forces: [M, A, 3]
#         self.stresses_predictor = nn.Linear(
#             transformer_input_dim, 3
#         )  # Stresses: [M, 3]

#     def forward(self, atomic_numbers, positions, src_key_padding_mask=None):
#         """
#         Args:
#             atomic_numbers: Tensor of shape [M, A] (atomic numbers for each atom in a batch)
#             positions: Tensor of shape [M, A, 3] (positions of atoms in 3D space)
#             src_key_padding_mask: Optional mask for padding

#         Returns:
#             energy: Tensor of shape [M, 1]
#             forces: Tensor of shape [M, A, 3]
#             stresses: Tensor of shape [M, 3]
#         """
#         # Embedding atomic numbers
#         atomic_emb = self.atomic_emb(atomic_numbers)  # Shape: [M, A, d_model]

#         # Concatenate embeddings and positions
#         emb = torch.cat([atomic_emb, positions], dim=-1)  # Shape: [M, A, d_model + 3]

#         # Pass concatenated data through the transformer
#         transformer_output = self.transformer(
#             emb, mask=src_key_padding_mask
#         )  # Shape: [M, A, transformer_input_dim]

#         # Energy: Global pooling and linear layer
#         pooled_output = transformer_output.mean(
#             dim=1
#         )  # Shape: [M, transformer_input_dim]
#         energy = self.energy_predictor(pooled_output)  # Shape: [M, 1]

#         # Forces: Per-atom output via linear layer
#         forces = self.forces_predictor(transformer_output)  # Shape: [M, A, 3]

#         # Stresses: Global pooling and linear layer
#         stresses = self.stresses_predictor(pooled_output)  # Shape: [M, 3]

#         return energy, forces, stresses


import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Encoder


class CombinedEmbedding(nn.Module):
    def __init__(self, num_tokens, d_model):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, d_model)
        self.position_proj = nn.Linear(3, d_model)  # Project 3D positions to d_model

    def forward(self, x, positions):
        """
        Args:
            x: Tensor of shape [M, A] containing atomic numbers.
            positions: Tensor of shape [M, A, 3] containing 3D atomic positions.
        Returns:
            combined_emb: Tensor of shape [M, A, d_model]
        """
        token_embeddings = self.token_emb(x)  # [M, A, d_model]
        pos_embeddings = self.position_proj(positions)  # [M, A, d_model]
        combined_emb = (
            token_embeddings + pos_embeddings
        )  # TODO: this should be concat such that dim is [M, A, d_model + 3]
        return combined_emb


class XTransformerModel(TransformerWrapper):
    def __init__(self, num_tokens, d_model, **kwargs):
        # Initialize the base TransformerWrapper without its own embedding
        super().__init__(
            num_tokens=num_tokens,
            max_seq_len=300,
            emb_dim=d_model,
            attn_layers=Encoder(
                dim=d_model,
                depth=kwargs.get("n_layers", 6),
                heads=kwargs.get("n_heads", 8),
                ff_mult=kwargs.get("d_ff_mult", 4),
            ),
            use_abs_pos_emb=False,  # Disable internal pos embeddings, ensure no additional embeddings are used
        )

        # Initialize the combined embedding
        self.emb = CombinedEmbedding(num_tokens, d_model)

        # Predictors for Energy, Forces, and Stresses
        self.energy_predictor = nn.Linear(d_model, 1)  # Energy: [M, 1]
        self.forces_predictor = nn.Linear(d_model, 3)  # Forces: [M, A, 3]
        self.stresses_predictor = nn.Linear(d_model, 3)  # Stresses: [M, 3]

    def forward(self, x, positions, mask=None):
        """
        Args:
            x: Tensor of shape [M, A] containing atomic numbers.
            positions: Tensor of shape [M, A, 3] containing 3D atomic positions.
            mask: Optional tensor of shape [M, A] indicating padding.
        Returns:
            Transformer output: Tensor of shape [M, A, d_model]
        """
        # Obtain combined embeddings
        combined_emb = self.emb(x, positions)  # [M, A, d_model]

        # Pass combined embeddings to the transformer
        output = self.attn_layers(x=combined_emb, mask=mask)  # [M, A, d_model]

        # Energy: Global pooling and linear layer
        pooled_output = output.mean(dim=1)  # Shape: [M, transformer_input_dim]
        energy = self.energy_predictor(pooled_output)  # Shape: [M, 1]

        # Forces: Per-atom output via linear layer
        forces = self.forces_predictor(output)  # Shape: [M, A, 3]

        # Stresses: Global pooling and linear layer
        stresses = self.stresses_predictor(pooled_output)  # Shape: [M, 3]

        return energy, forces, stresses
