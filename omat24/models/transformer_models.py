import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Encoder


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
            concatenated_emb: Tensor of shape [M, A, d_model + 3]
        """
        token_embeddings = self.token_emb(x)
        concatenated_emb = torch.cat(
            [token_embeddings, positions], dim=-1
        )  # [M, A, d_model + 3 or 5]
        return concatenated_emb


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
        """
        # fmt: off
        self.configurations = [
            {"d_model": 1, "depth": 1, "n_heads": 1, "d_ff_mult": 1},  # 1,670 params
            {"d_model": 4, "depth": 2, "n_heads": 2, "d_ff_mult": 2},  # 8,753 params
            {"d_model": 8, "depth": 2, "n_heads": 4, "d_ff_mult": 2},  # 25,541 params
            {"d_model": 8, "depth": 4, "n_heads": 4, "d_ff_mult": 4},  # 51,171 params
            {"d_model": 16, "depth": 4, "n_heads": 4, "d_ff_mult": 4},  # 93,851 params
            {"d_model": 16, "depth": 6, "n_heads": 4, "d_ff_mult": 8},  # 156,589 params
            {"d_model": 24, "depth": 6, "n_heads": 6, "d_ff_mult": 8},  # 327,061 params
            {"d_model": 32, "depth": 8, "n_heads": 8, "d_ff_mult": 8},  # 742,815 params
            {"d_model": 32, "depth": 12, "n_heads": 8, "d_ff_mult": 8},  # 1,109,475 params
            {"d_model": 64, "depth": 6, "n_heads": 8, "d_ff_mult": 16},  # 1,719,565 params
            {"d_model": 48, "depth": 12, "n_heads": 8, "d_ff_mult": 12},  # 2,028,739 params
            {"d_model": 64, "depth": 8, "n_heads": 8, "d_ff_mult": 16},  # 2,283,839 params
            {"d_model": 96, "depth": 8, "n_heads": 8, "d_ff_mult": 16},  # 4,198,303 params
            {"d_model": 128, "depth": 8, "n_heads": 8, "d_ff_mult": 16},  # 6,645,247 params
            {"d_model": 128, "depth": 10, "n_heads": 16, "d_ff_mult": 16},  # 10,967,985 params
            {"d_model": 192, "depth": 8, "n_heads": 12, "d_ff_mult": 24},  # 19,613,695 params
            {"d_model": 256, "depth": 6, "n_heads": 8, "d_ff_mult": 32},  # 29,298,349 params
            {"d_model": 256, "depth": 8, "n_heads": 16, "d_ff_mult": 32},  # 43,207,167 params
            {"d_model": 384, "depth": 8, "n_heads": 16, "d_ff_mult": 48},  # 128,511,487 params
            {"d_model": 512, "depth": 8, "n_heads": 16, "d_ff_mult": 32},  # 153,943,295 params
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
        """Initializes XTransformerModel with specified configurations.

        Args:
            num_tokens (int): Number of unique tokens (atomic numbers).
            d_model (int): Dimension of the token embeddings.
            depth (int): Number of transformer layers.
            n_heads (int): Number of attention heads.
            d_ff_mult (int): Multiplier for the feed-forward network dimension.
            use_factorized (bool): Whether to use factorized distances instead of positions.
        """
        self.embedding_dim = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.d_ff_mult = d_ff_mult
        self.use_factorized = use_factorized
        self.additional_dim = 5 if use_factorized else 3  # For concatenated positions
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
        self.energy_1 = nn.Linear(
            d_model + self.additional_dim, d_model + self.additional_dim
        )
        self.energy_2 = nn.Linear(d_model + self.additional_dim, 1)  # Energy: [M, 1]
        self.force_1 = nn.Linear(
            d_model + self.additional_dim, d_model + self.additional_dim
        )
        self.force_2 = nn.Linear(d_model + self.additional_dim, 3)  # Forces: [M, A, 3]
        self.stress_1 = nn.Linear(
            d_model + self.additional_dim, d_model + self.additional_dim
        )
        self.stress_2 = nn.Linear(d_model + self.additional_dim, 6)  # Stresses: [M, 6]

        # Initialize weights in the __init__ method
        # Initialize Linear 1 with Xavier initialization (normal distribution)
        nn.init.xavier_normal_(self.energy_1.weight)
        if self.energy_1.bias is not None:
            nn.init.zeros_(self.energy_1.bias)
        nn.init.xavier_normal_(self.force_1.weight)
        if self.force_1.bias is not None:
            nn.init.zeros_(self.force_1.bias)
        nn.init.xavier_normal_(self.stress_1.weight)
        if self.stress_1.bias is not None:
            nn.init.zeros_(self.stress_1.bias)
        # Initialize Linear 2 with Xavier initialization (normal distribution)
        nn.init.xavier_normal_(self.energy_2.weight)
        if self.energy_2.bias is not None:
            nn.init.zeros_(self.energy_2.bias)
        nn.init.xavier_normal_(self.force_2.weight)
        if self.force_2.bias is not None:
            nn.init.zeros_(self.force_2.bias)
        nn.init.xavier_normal_(self.stress_2.weight)
        if self.stress_2.bias is not None:
            nn.init.zeros_(self.stress_2.bias)

        # Count parameters
        self.num_params = sum(
            p.numel() for name, p in self.named_parameters() if "token_emb" not in name
        )

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
        # Obtain concatenated embeddings
        if self.use_factorized:
            concatenated_emb = self.token_emb(x, distance_matrix)  # [M, A, d_model]
        else:
            concatenated_emb = self.token_emb(x, positions)  # [M, A, d_model]

        # Pass embeddings to the transformer
        output = self.attn_layers(x=concatenated_emb, mask=mask)  # [M, A, d_model]

        # Predict forces
        forces = self.force_2(torch.tanh(self.force_1(output)))  # [M, A, 3]
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        forces = forces * expanded_mask.float()  # Mask padded atoms

        # Predict per-atom energy contributions and sum
        energy_contrib = self.energy_2(torch.tanh(self.energy_1(output))).squeeze(
            -1
        )  # [M, A]
        energy_contrib = energy_contrib * mask.squeeze(-1).float()
        energy = energy_contrib.sum(dim=1)  # [batch_size]

        # Predict per-atom stress contributions and sum
        stress_contrib = self.stress_2(torch.tanh(self.stress_1(output)))  # [M, A, 6]
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 6)
        stress_contrib = stress_contrib * expanded_mask.float()
        stress = stress_contrib.sum(dim=1)  # [batch_size, 6]

        return forces, energy, stress
