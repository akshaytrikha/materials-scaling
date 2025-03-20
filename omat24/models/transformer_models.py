# External
import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Encoder

# Internal
from models.model_utils import MLPOutput, initialize_output_heads, initialize_weights


class ConcatenatedEmbedding(nn.Module):
    def __init__(self, num_tokens, d_model):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, d_model)
        # Use standardized initialization
        initialize_weights(self.token_emb)

    def forward(self, x, positions):
        token_embeddings = self.token_emb(x)
        concatenated_emb = torch.cat([token_embeddings, positions], dim=-1)
        return concatenated_emb


class XTransformerModel(TransformerWrapper):
    def __init__(
        self,
        num_tokens,
        d_model,
        depth,
        n_heads,
        d_ff_mult,
        use_factorized,
    ):
        self.embedding_dim = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.d_ff_mult = d_ff_mult
        self.use_factorized = use_factorized
        self.additional_dim = 6  # For concatenated cartesian + fractional positions
        self.name = "Transformer"

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
                attn_flash=torch.cuda.is_available(),
            ),
            use_abs_pos_emb=False,  # Disable internal positional embeddings
        )

        self.token_emb = ConcatenatedEmbedding(num_tokens, d_model)

        # Predictors for Energy, Forces, and Stresses
        self.energy_head = MLPOutput(d_model + self.additional_dim, 1)
        self.force_head = MLPOutput(d_model + self.additional_dim, 3)
        self.stress_head = MLPOutput(d_model + self.additional_dim, 6)

        # Initialize output heads to predict dataset means
        initialize_output_heads(self.energy_head, self.force_head, self.stress_head)

        # Count parameters
        self.num_params = sum(
            p.numel() for name, p in self.named_parameters() if "token_emb" not in name
        )

    def forward(self, x, positions, distance_matrix=None, mask=None):
        # Obtain concatenated embeddings, positions now contains both coordinate types
        concatenated_emb = self.token_emb(x, positions)

        # Pass embeddings to the transformer
        output = self.attn_layers(x=concatenated_emb, mask=mask)

        # Predict forces
        forces = self.force_head(output)
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        forces = forces * expanded_mask.float()

        # Predict per-atom energy contributions and sum
        energy_contrib = self.energy_head(output).squeeze(-1)
        energy_contrib = energy_contrib * mask.float()
        energy = energy_contrib.sum(dim=1)

        # Predict per-atom stress contributions and sum
        stress_contrib = self.stress_head(output)
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 6)
        stress_contrib = stress_contrib * expanded_mask.float()
        stress = stress_contrib.sum(dim=1)

        return forces, energy, stress


class MetaTransformerModels:
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        use_factorized=False,
    ):
        """Initializes TransformerModels with a list of configurations.

        Args:
            vocab_size (int): Number of unique tokens (atomic numbers).
            max_seq_len (int): Maximum sequence length for the transformer.
            use_factorized (bool): Whether to use factorized distances instead of positions.
        """
        # fmt: off
        self.configurations = [
            {"d_model": 1, "depth": 1, "n_heads": 1, "d_ff_mult": 4}, # 1,778 params
            {"d_model": 8, "depth": 2, "n_heads": 1, "d_ff_mult": 4}, # 9,657 params
            {"d_model": 48, "depth": 3, "n_heads": 1, "d_ff_mult": 4}, # 119,758 params
            {"d_model": 160, "depth": 3, "n_heads": 2, "d_ff_mult": 4}, # 1,019,086 params
        ]
        # fmt: on

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
            use_factorized=self.use_factorized,
        )

    def __len__(self):
        """Returns the number of configurations."""
        return len(self.configurations)

    def __iter__(self):
        """Allows iteration over all transformer models."""
        for idx in range(len(self.configurations)):
            yield self[idx]
