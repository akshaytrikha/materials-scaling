import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder


class XTransformerModel(torch.nn.Module):
    def __init__(
        self, num_elements, d_model, n_layers, n_heads, d_ff, transformer_output_dim
    ):
        super().__init__()
        # Atomic embedding for atomic numbers
        self.atomic_emb = nn.Embedding(num_elements, d_model)

        # Transformer definition
        self.transformer = TransformerWrapper(
            num_tokens=119,  # No token-based sequence; we use embeddings directly
            max_seq_len=None,  # No explicit sequence length
            attn_layers=Decoder(
                dim=d_model,
                depth=n_layers,
                heads=n_heads,
                ff_mult=d_ff // d_model,
            ),
        )

        # Predictors for Energy, Forces, and Stresses
        self.energy_predictor = nn.Linear(
            transformer_output_dim, 1
        )  # Energy: [M, 1, 1]
        self.forces_predictor = nn.Linear(
            transformer_output_dim, 3
        )  # Forces: [M, A, 3]
        self.stresses_predictor = nn.Linear(
            transformer_output_dim, 3
        )  # Stresses: [M, 1, 3]

    def forward(self, atomic_numbers, positions, src_key_padding_mask=None):
        """
        Args:
            atomic_numbers: Tensor of shape [M, A, 1] (atomic numbers for each atom in a batch)
            positions: Tensor of shape [M, A, 3] (positions of atoms in 3D space)
            src_key_padding_mask: Optional mask for padding

        Returns:
            energy: Tensor of shape [M, 1, 1]
            forces: Tensor of shape [M, A, 3]
            stresses: Tensor of shape [M, 1, 3]
        """
        # Embedding atomic numbers
        atomic_emb = self.atomic_emb(atomic_numbers)  # Shape: [M, A, d_model]

        # Concatenate embeddings and positions
        emb = torch.cat([atomic_emb, positions], dim=-1)  # Shape: [M, A, d_model + 3]

        # Pass concatenated data through the transformer
        transformer_output = self.transformer(
            emb, mask=src_key_padding_mask
        )  # Shape: [M, A, transformer_output_dim]
        breakpoint()
        # Energy: Global pooling and linear layer
        energy = self.energy_predictor(
            transformer_output.mean(dim=1)
        )  # Shape: [M, 1, 1]

        # Forces: Per-atom output via linear layer
        forces = self.forces_predictor(transformer_output)  # Shape: [M, A, 3]

        # Stresses: Global pooling and linear layer
        stresses = self.stresses_predictor(
            transformer_output.mean(dim=1)
        )  # Shape: [M, 1, 3]

        return energy, forces, stresses
