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
        combined_emb = torch.cat(
            [token_embeddings, positions], dim=-1
        )  # [M, A, d_model + 3]
        return combined_emb


class MetaTransformerModels:
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        concatenated=False,
    ):
        """Initializes TransformerModels with a list of configurations.

        Args:
            vocab_size (int): Number of unique tokens (atomic numbers).
            max_seq_len (int): Maximum sequence length for the transformer.
        """
        self.configurations = [
            {
                "d_model": 1,
                "depth": 1,
                "n_heads": 1,
                "d_ff_mult": 1,
                "concatenated": concatenated,
            },
            {
                "d_model": 2,
                "depth": 1,
                "n_heads": 1,
                "d_ff_mult": 2,
                "concatenated": concatenated,
            },
            {
                "d_model": 2,
                "depth": 1,
                "n_heads": 2,
                "d_ff_mult": 2,
                "concatenated": concatenated,
            },
            {
                "d_model": 2,
                "depth": 2,
                "n_heads": 2,
                "d_ff_mult": 2,
                "concatenated": concatenated,
            },
            {
                "d_model": 4,
                "depth": 2,
                "n_heads": 2,
                "d_ff_mult": 4,
                "concatenated": concatenated,
            },  # Medium
        ]

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        """Retrieves transformer model corresponding to the configuration at index `idx`.

        Args:
            idx (int): Index of the desired configuration.

        Returns:
            XTransformerModel: An instance of the transformer model with the specified configuration.

        Raises:
            IndexError: If the index is out of range.
        """
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range")
        config = self.configurations[idx]
        return XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=config["d_model"],
            depth=config["depth"],
            n_heads=config["n_heads"],
            d_ff_mult=config["d_ff_mult"],
            concatenated=config["concatenated"],
        )

    def __len__(self):
        """Returns the number of configurations."""
        return len(self.configurations)

    def __iter__(self):
        """Allows iteration over all transformer models."""
        for idx in range(len(self.configurations)):
            yield self[idx]


class XTransformerModel(TransformerWrapper):
    def __init__(self, num_tokens, d_model, depth, n_heads, d_ff_mult, concatenated):
        """Initializes XTransformerModel with specified configurations.

        Args:
            num_tokens (int): Number of unique tokens (atomic numbers).
            d_model (int): Dimension of the token embeddings.
            depth (int): Number of transformer layers.
            n_heads (int): Number of attention heads.
            d_ff_mult (int): Multiplier for the feed-forward network dimension.
            concatenated (bool): Whether to concatenate positional information.
        """
        self.embedding_dim = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.d_ff_mult = d_ff_mult
        self.additional_dim = 3 if concatenated else 0  # For concatenated positions

        # Initialize base TransformerWrapper without its own embedding
        super().__init__(
            num_tokens=num_tokens,
            max_seq_len=300,
            emb_dim=d_model,
            attn_layers=Encoder(
                dim=d_model + self.additional_dim,
                depth=depth,
                heads=n_heads,
                ff_mult=d_ff_mult,
            ),
            use_abs_pos_emb=False,  # Disable internal positional embeddings
        )

        # Initialize the combined or concatenated embedding
        if concatenated:
            self.emb = ConcatenatedEmbedding(num_tokens, d_model)
        else:
            self.emb = CombinedEmbedding(num_tokens, d_model)

        # Predictors for Energy, Forces, and Stresses
        self.energy_predictor = nn.Linear(
            d_model + self.additional_dim, 1
        )  # Energy: [M, 1]
        self.forces_predictor = nn.Linear(
            d_model + self.additional_dim, 3
        )  # Forces: [M, A, 3]
        self.stresses_predictor = nn.Linear(
            d_model + self.additional_dim, 6
        )  # Stresses: [M, 6]

        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x, positions, mask=None):
        """Forward pass of the transformer model.

        Args:
            x (Tensor): Tensor of shape [M, A] containing atomic numbers.
            positions (Tensor): Tensor of shape [M, A, 3] containing 3D atomic positions.
            mask (Tensor, optional): Tensor of shape [M, A] indicating padding.

        Returns:
            tuple: (forces, energy, stresses)
                - forces (Tensor): [M, A, 3]
                - energy (Tensor): [M]
                - stresses (Tensor): [M, 6]
        """
        breakpoint()
        print(f"x.shape is {x.shape}")
        print(f"positions.shape is {positions.shape}")
        # Obtain combined embeddings
        combined_emb = self.emb(x, positions)  # [M, A, d_model + self.additional_dim]
        print(f"combined_emb.shape is {combined_emb.shape}")

        # Pass combined embeddings to the transformer
        output = self.attn_layers(x=combined_emb, mask=mask)  # [M, A, d_model]
        print(f"output.shape is {output.shape}")

        # Energy: Global pooling and linear layer
        pooled_output = output.mean(dim=1)  # Shape: [M, d_model]
        print(f"pooled_output.shape is {pooled_output.shape}")
        energy = self.energy_predictor(pooled_output).squeeze()  # Shape: [M]
        print(f"energy.shape is {energy.shape}")
        # Forces: Per-atom output via linear layer
        forces = self.forces_predictor(output)  # Shape: [M, A, 3]
        print(f"forces.shape is {forces.shape}")

        # Stresses: Global pooling and linear layer
        stresses = self.stresses_predictor(pooled_output)  # Shape: [M, 6]
        print(f"stresses.shape is {stresses.shape}")

        return forces, energy, stresses
