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
        token_embeddings = self.token_emb(x)
        pos_embeddings = self.position_proj(positions)
        combined_emb = token_embeddings + pos_embeddings  # [M, A, d_model]

        return combined_emb


class XTransformerModel(TransformerWrapper):
    def __init__(self, num_tokens, d_model, depth, n_heads, d_ff_mult):
        self.embedding_dim = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.d_ff_mult = d_ff_mult

        # Init base TransformerWrapper without its own embedding
        super().__init__(
            num_tokens=num_tokens,
            max_seq_len=300,
            emb_dim=d_model,
            attn_layers=Encoder(
                dim=d_model,
                depth=depth,
                heads=n_heads,
                ff_mult=d_ff_mult,
            ),
            use_abs_pos_emb=False,  # Disable internal pos embeddings, ensure no additional embeddings are used
        )

        # Initialize the combined embedding
        self.emb = CombinedEmbedding(num_tokens, d_model)

        # Predictors for Energy, Forces, and Stresses
        self.energy_predictor = nn.Linear(d_model, 1)  # Energy: [M, 1]
        self.forces_predictor = nn.Linear(d_model, 3)  # Forces: [M, A, 3]
        self.stresses_predictor = nn.Linear(d_model, 6)  # Stresses: [M, 3]

        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())

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
        energy = self.energy_predictor(pooled_output).squeeze()  # Shape: [M]

        # Forces: Per-atom output via linear layer
        forces = self.forces_predictor(output)  # Shape: [M, A, 3]

        # Stresses: Global pooling and linear layer
        stresses = self.stresses_predictor(pooled_output)  # Shape: [M, 3]

        return forces, energy, stresses


class ConcatenatedEmbedding(nn.Module):
    def __init__(self, num_tokens, d_model):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, d_model)

    def forward(self, x, positions):
        """
        Args:
            x: Tensor of shape [M, A] containing atomic numbers.
            positions: Tensor of shape [M, A, 3] containing 3D atomic positions.
        Returns:
            combined_emb: Tensor of shape [M, A, d_model + 3]
        """
        token_embeddings = self.token_emb(x)
        combined_emb = torch.cat([token_embeddings, positions], dim=-1)  # [M, A, d_model + 3]
        return combined_emb

class XTransformerConcatenatedModel(TransformerWrapper):
    def __init__(self, num_tokens, d_model, depth, n_heads, d_ff_mult):
        self.embedding_dim = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.d_ff_mult = d_ff_mult

        # Init base TransformerWrapper without its own embedding
        super().__init__(
            num_tokens=num_tokens,
            max_seq_len=300,
            emb_dim=d_model,
            attn_layers=Encoder(
                dim=d_model + 3,
                depth=depth,
                heads=n_heads,
                ff_mult=d_ff_mult,
            ),
            use_abs_pos_emb=False,  # Disable internal pos embeddings, ensure no additional embeddings are used
        )

        # Initialize the combined embedding
        self.emb = ConcatenatedEmbedding(num_tokens, d_model)
        
        # Predictors for Energy, Forces, and Stresses
        self.energy_predictor = nn.Linear(d_model + 3, 1)  # Energy: [M, 1]
        self.forces_predictor = nn.Linear(d_model + 3, 3)  # Forces: [M, A, 3]
        self.stresses_predictor = nn.Linear(d_model + 3, 6)  # Stresses: [M, 3]

        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())

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
        combined_emb = self.emb(x, positions)  # [M, A, d_model + 3]

        # Pass combined embeddings to the transformer
        output = self.attn_layers(x=combined_emb, mask=mask)  # [M, A, d_model + 3]

        # Energy: Global pooling and linear layer
        pooled_output = output.mean(dim=1)  # Shape: [M, transformer_input_dim]

        energy = self.energy_predictor(pooled_output).squeeze()  # Shape: [M]
        # Forces: Per-atom output via linear layer
        forces = self.forces_predictor(output)  # Shape: [M, A, 3]

        # Stresses: Global pooling and linear layer
        stresses = self.stresses_predictor(pooled_output)  # Shape: [M, 3]
        
        return forces, energy, stresses