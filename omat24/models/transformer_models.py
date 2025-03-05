import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Encoder


class FixedGaussianRBF(nn.Module):
    """
    Fixed Gaussian Radial Basis Function layer with predefined centers
    """

    def __init__(self, num_centers=50, cutoff=12.0):
        super().__init__()
        self.num_centers = num_centers
        self.cutoff = cutoff
        self.num_basis = num_centers  # For compatibility with the interface

        # Fixed centers evenly spaced from 0 to cutoff
        # Note: We're not creating a parameter since these are fixed
        # Instead we'll create them in forward() to match the device

    def forward(self, distances):
        """
        Apply fixed Gaussian RBF to distances

        Args:
            distances: Tensor of shape [..., ] containing atomic distances

        Returns:
            Tensor of shape [..., num_centers] containing RBF-encoded distances
        """
        # Store original shape to reshape output later
        original_shape = distances.shape

        # Flatten input if needed
        distances_flat = distances.view(-1) if len(original_shape) > 1 else distances

        # Create centers on the same device as the input (evenly spaced from 0 to cutoff)
        centers = torch.linspace(
            0, self.cutoff, self.num_centers, device=distances.device
        )

        # Calculate width based on spacing of centers (rule of thumb: 2x the spacing)
        width = (centers[1] - centers[0]) * 2.0

        # Expand dimensions for broadcasting
        distances_expanded = distances_flat.unsqueeze(-1)  # [N, 1]

        # Apply RBF
        rbf_features = torch.exp(-((distances_expanded - centers) ** 2) / width**2)

        # Reshape to original dimensions with new feature dimension
        if len(original_shape) > 1:
            output_shape = original_shape + (self.num_centers,)
            rbf_features = rbf_features.view(output_shape)

        return rbf_features


# Compute pairwise distances between atoms
def compute_distance_matrix(positions):
    """
    Compute pairwise Euclidean distance matrix between atoms

    Args:
        positions: Tensor of shape [batch_size, num_atoms, 3]

    Returns:
        Tensor of shape [batch_size, num_atoms, num_atoms] containing pairwise distances
    """
    # Compute pairwise distances
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [batch, atom_i, atom_j, 3]
    dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8)  # [batch, atom_i, atom_j]
    return dist


class RBFEmbedding(nn.Module):
    """
    Embedding layer that combines atom type embeddings with RBF-encoded interatomic distances
    """

    def __init__(self, num_tokens, d_model, num_rbf_basis=50, cutoff=12.0):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, d_model)
        self.rbf = FixedGaussianRBF(num_centers=num_rbf_basis, cutoff=cutoff)

        # Project RBF features to embedding dimension
        self.rbf_projection = nn.Linear(num_rbf_basis, d_model)

    def forward(self, atomic_numbers, positions):
        """
        Args:
            atomic_numbers: Tensor of shape [batch_size, num_atoms] containing atomic numbers
            positions: Tensor of shape [batch_size, num_atoms, 3] containing 3D atomic positions
        Returns:
            Tensor of shape [batch_size, num_atoms, d_model] containing combined embeddings
        """
        # Get atom type embeddings
        token_embeddings = self.token_emb(atomic_numbers)  # [batch, atoms, d_model]

        # Compute distance matrix
        distances = compute_distance_matrix(positions)  # [batch, atoms, atoms]

        # Apply RBF encoding
        rbf_features = self.rbf(distances)  # [batch, atoms, atoms, num_basis]

        # Aggregate RBF features over neighboring atoms (mean pooling)
        rbf_features = rbf_features.mean(dim=2)  # [batch, atoms, num_basis]

        # Project to embedding dimension
        rbf_embeddings = self.rbf_projection(rbf_features)  # [batch, atoms, d_model]

        # Combine embeddings
        combined_emb = token_embeddings + rbf_embeddings
        # combined_emb = torch.cat([token_embeddings, rbf_embeddings], dim=-1)

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
            # {"d_model": 16, "depth": 1, "n_heads": 1, "d_ff_mult": 2, "num_rbf": 16},  # ~4k params
            # {"d_model": 16, "depth": 2, "n_heads": 2, "d_ff_mult": 2, "num_rbf": 32},  # ~15k params
            # {"d_model": 32, "depth": 2, "n_heads": 4, "d_ff_mult": 2, "num_rbf": 32},  # ~40k params 
            {"d_model": 32, "depth": 8, "n_heads": 2, "d_ff_mult": 8, "num_rbf": 128},  # ~100k params
        #     {"d_model": 64, "depth": 4, "n_heads": 4, "d_ff_mult": 4, "num_rbf": 48},  # ~300k params
        #     {"d_model": 64, "depth": 6, "n_heads": 8, "d_ff_mult": 8, "num_rbf": 64},  # ~500k params
        #     {"d_model": 96, "depth": 6, "n_heads": 8, "d_ff_mult": 8, "num_rbf": 64},  # ~1M params
        #     {"d_model": 128, "depth": 8, "n_heads": 8, "d_ff_mult": 8, "num_rbf": 96},  # ~3M params
        #     {"d_model": 128, "depth": 12, "n_heads": 8, "d_ff_mult": 8, "num_rbf": 96},  # ~5M params
        #     {"d_model": 192, "depth": 12, "n_heads": 12, "d_ff_mult": 12, "num_rbf": 128},  # ~15M params
        #     {"d_model": 256, "depth": 12, "n_heads": 16, "d_ff_mult": 16, "num_rbf": 128},  # ~30M params
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
        return XTransformerRBFModel(
            num_tokens=self.vocab_size,
            d_model=config["d_model"],
            depth=config["depth"],
            n_heads=config["n_heads"],
            d_ff_mult=config["d_ff_mult"],
            num_rbf=config["num_rbf"],
            use_factorized=self.use_factorized,
        )

    def __len__(self):
        """Returns the number of configurations."""
        return len(self.configurations)

    def __iter__(self):
        """Allows iteration over all transformer models."""
        for idx in range(len(self.configurations)):
            yield self[idx]


class XTransformerRBFModel(nn.Module):
    def __init__(
        self,
        num_tokens,
        d_model,
        depth,
        n_heads,
        d_ff_mult,
        num_rbf,
        use_factorized=False,
    ):
        """Initializes an improved XTransformerModel with RBF encodings.

        Args:
            num_tokens (int): Number of unique tokens (atomic numbers).
            d_model (int): Dimension of the token embeddings.
            depth (int): Number of transformer layers.
            n_heads (int): Number of attention heads.
            d_ff_mult (int): Multiplier for the feed-forward network dimension.
            num_rbf (int): Number of RBF basis functions.
            use_factorized (bool): Whether to use factorized distances instead of positions.
        """
        super().__init__()
        self.embedding_dim = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.d_ff_mult = d_ff_mult
        self.use_factorized = use_factorized
        self.name = "Transformer"
        self.num_rbf = num_rbf

        # Use RBF embedding
        self.token_emb = RBFEmbedding(
            num_tokens=num_tokens, d_model=d_model, num_rbf_basis=num_rbf, cutoff=12.0
        )

        # Create Transformer encoder
        self.transformer = Encoder(
            dim=d_model,
            depth=depth,
            heads=n_heads,
            ff_mult=d_ff_mult,
            attn_flash=torch.cuda.is_available(),
        )

        # Predictors for Energy, Forces, and Stresses
        self.energy_1 = nn.Linear(d_model, d_model)
        self.energy_2 = nn.Linear(d_model, 1)  # Energy: [M, 1]
        self.force_1 = nn.Linear(d_model, d_model)
        self.force_2 = nn.Linear(d_model, 3)  # Forces: [M, A, 3]
        self.stress_1 = nn.Linear(d_model, d_model)
        self.stress_2 = nn.Linear(d_model, 6)  # Stresses: [M, 6]

        # Initialize weights
        for m in [
            self.energy_1,
            self.force_1,
            self.stress_1,
            self.energy_2,
            self.force_2,
            self.stress_2,
        ]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Count parameters
        self.num_params = sum(
            p.numel()
            for name, p in self.named_parameters()
            if "token_emb.token_emb" not in name
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
        # Get combined RBF embeddings
        embeddings = self.token_emb(x, positions)  # [M, A, d_model]

        # Pass embeddings to the transformer
        output = self.transformer(x=embeddings, mask=mask)  # [M, A, d_model]

        # Predict forces
        forces = self.force_2(torch.tanh(self.force_1(output)))  # [M, A, 3]
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 3)
            forces = forces * expanded_mask.float()  # Mask padded atoms

        # Predict per-atom energy contributions and sum
        energy_contrib = self.energy_2(torch.tanh(self.energy_1(output))).squeeze(
            -1
        )  # [M, A]
        if mask is not None:
            energy_contrib = energy_contrib * mask.float()
        energy = energy_contrib.sum(dim=1)  # [batch_size]

        # Predict per-atom stress contributions and sum
        stress_contrib = self.stress_2(torch.tanh(self.stress_1(output)))  # [M, A, 6]
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 6)
            stress_contrib = stress_contrib * expanded_mask.float()
        stress = stress_contrib.sum(dim=1)  # [batch_size, 6]

        return forces, energy, stress


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
