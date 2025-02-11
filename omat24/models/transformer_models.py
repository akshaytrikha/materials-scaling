import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Encoder


class CombinedEmbedding(nn.Module):
    def __init__(self, num_tokens, d_model, additional_dim=3):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, d_model)
        self.position_proj = nn.Linear(
            additional_dim, d_model
        )  # Project 3D positions to d_model

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
        )  # [M, A, d_model + 3 or 5]
        return combined_emb


class MetaTransformerModels:
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        concatenated=False,
        use_factorized=False,
    ):
        """Initializes TransformerModels with a list of configurations.

        Args:
            vocab_size (int): Number of unique tokens (atomic numbers).
            max_seq_len (int): Maximum sequence length for the transformer.
        """
        self.configurations = [
            # 1729 params
            {
                "d_model": 1,
                "depth": 1,
                "n_heads": 1,
                "d_ff_mult": 1,
                "concatenated": concatenated,
            },
            # 9061 params
            {
                "d_model": 4,
                "depth": 2,
                "n_heads": 2,
                "d_ff_mult": 2,
                "concatenated": concatenated,
            },
            # 109059 params
            {
                "d_model": 8,
                "depth": 8,
                "n_heads": 4,
                "d_ff_mult": 8,
                "concatenated": concatenated,
            },
            # {
            #     "d_model": 2,
            #     "depth": 1,
            #     "n_heads": 2,
            #     "d_ff_mult": 2,
            #     "concatenated": concatenated,
            # },
            # {
            #     "d_model": 2,
            #     "depth": 2,
            #     "n_heads": 2,
            #     "d_ff_mult": 2,
            #     "concatenated": concatenated,
            # },
            # {
            #     "d_model": 4,
            #     "depth": 2,
            #     "n_heads": 2,
            #     "d_ff_mult": 4,
            #     "concatenated": concatenated,
            # },  # Medium
        ]

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_factorized = use_factorized

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
            use_factorized=self.use_factorized,
        )

    def __len__(self):
        """Returns the number of configurations."""
        return len(self.configurations)

    def __iter__(self):
        """Allows iteration over all transformer models."""
        for idx in range(len(self.configurations)):
            yield self[idx]


class XTransformerModel(TransformerWrapper):
    def __init__(
        self,
        num_tokens,
        d_model,
        depth,
        n_heads,
        d_ff_mult,
        concatenated,
        use_factorized,
    ):
        """Initializes XTransformerModel with specified configurations.

        Args:
            num_tokens (int): Number of unique tokens (atomic numbers).
            d_model (int): Dimension of the token embeddings.
            depth (int): Number of transformer layers.
            n_heads (int): Number of attention heads.
            d_ff_mult (int): Multiplier for the feed-forward network dimension.
            concatenated (bool): Whether to concatenate positional information.
            use_factorized (bool): Whether to use factorized distances instead of positions.
        """
        self.embedding_dim = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.d_ff_mult = d_ff_mult
        self.use_factorized = use_factorized
        self.additional_dim = (
            5 if use_factorized else 3 if concatenated else 0
        )  # For concatenated positions

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

        if concatenated:
            self.token_emb = ConcatenatedEmbedding(num_tokens, d_model)
        else:
            self.token_emb = CombinedEmbedding(num_tokens, d_model, self.additional_dim)

        # Predictors for Energy, Forces, and Stresses
        self.energy_output = nn.Linear(
            d_model + self.additional_dim, 1
        )  # Energy: [M, 1]
        self.force_output = nn.Linear(
            d_model + self.additional_dim, 3
        )  # Forces: [M, A, 3]
        self.stress_output = nn.Linear(
            d_model + self.additional_dim, 6
        )  # Stresses: [M, 6]

        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x, positions, distance_matrix=None, mask=None):
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
        # Obtain combined embeddings
        if self.use_factorized:
            combined_emb = self.token_emb(x, distance_matrix)  # [M, A, d_model]
        else:
            combined_emb = self.token_emb(x, positions)  # [M, A, d_model]

        # Pass combined embeddings to the transformer
        output = self.attn_layers(x=combined_emb, mask=mask)  # [M, A, d_model]

        # Predict forces
        forces = self.force_output(output)  # [M, A, 3]
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        forces = forces * expanded_mask.float()  # Mask padded atoms

        # Predict per-atom energy contributions and sum
        energy_contrib = self.energy_output(output).squeeze(-1)  # [M, A]
        energy_contrib = energy_contrib * mask.squeeze(-1).float()
        energy = energy_contrib.sum(dim=1)  # [batch_size]

        # Predict per-atom stress contributions and sum
        stress_contrib = self.stress_output(output)  # [M, A, 6]
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 6)
        stress_contrib = stress_contrib * expanded_mask.float()
        stress = stress_contrib.sum(dim=1)  # [batch_size, 6]

        return forces, energy, stress
