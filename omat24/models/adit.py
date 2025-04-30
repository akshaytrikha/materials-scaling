# External
import torch
from torch import nn
from typing import Dict
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter

# Internal
from models.model_utils import MLPOutput, initialize_output_heads, initialize_weights


def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine positional embeddings from a prespecified indices.
    Args:
    indices: offsets of size [..., num_tokens] of type integer
    emb_dim: dimension of the embeddings to create
    max_len: maximum length
    Returns:
    positional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * torch.pi / (max_len ** (2 * K[None] / emb_dim))
    )
    pos_embedding_cos = torch.cos(
        indices[..., None] * torch.pi / (max_len ** (2 * K[None] / emb_dim))
    )
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class TransformerEncoder(nn.Module):
    """Transformer encoder from the All-Atom Diffusion Transformer paper.

    Args:
        max_num_elements: Maximum number of elements in the dataset
        d_model: Dimension of the model
        nhead: Number of attention heads
        dim_feedforward: Dimension of the feedforward network
        activation: Activation function to use
        dropout: Dropout rate
        norm_first: Whether to use pre-normalization in Transformer blocks
        bias: Whether to use bias
        num_layers: Number of layers
    """

    def __init__(
        self,
        max_num_elements: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: str,
        dropout: float,
        norm_first: bool,
        bias: bool,
        num_layers: int,
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers

        self.atom_type_embedder = nn.Embedding(max_num_elements, d_model)
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.frac_coords_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        activation = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
        }[activation]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Data object with the following attributes:
                atomic_numbers (torch.Tensor): Atomic numbers of atoms in the batch
                pos (torch.Tensor): Cartesian coordinates of atoms in the batch
                frac_coords (torch.Tensor): Fractional coordinates of atoms in the batch
                cell (torch.Tensor): Lattice vectors of the unit cell
                lattices (torch.Tensor): Lattice parameters of the unit cell (lengths and angles)
                lengths (torch.Tensor): Lengths of the lattice vectors
                angles (torch.Tensor): Angles between the lattice vectors
                natoms (torch.Tensor): Number of atoms in the batch
                batch (torch.Tensor): Batch index for each atom
        """
        x = self.atom_type_embedder(batch.atomic_numbers)  # (n, d)
        x += self.pos_embedder(batch.pos)
        x += self.frac_coords_embedder(batch.frac_coords)

        # Positional embedding
        x += get_index_embedding(batch.token_idx, self.d_model)

        # Convert from PyG batch to dense batch with padding
        x, token_mask = to_dense_batch(x, batch.batch)

        # Transformer forward pass
        x = self.transformer.forward(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]

        return {
            "x": x,
            "natoms": batch.natoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }


class ADiTS2EFSModel(nn.Module):
    """Model using ADiT transformer for force prediction."""

    def __init__(
        self,
        max_num_elements: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: str,
        dropout: float,
        norm_first: bool,
        bias: bool,
        num_layers: int,
    ):
        super().__init__()
        self.name = "ADiT"

        # Initialize the transformer
        self.transformer = TransformerEncoder(
            max_num_elements=max_num_elements,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            norm_first=norm_first,
            bias=bias,
            num_layers=num_layers,
        )

        # Predictors for Energy, Forces, and Stresses
        self.energy_head = MLPOutput(d_model, 1)
        self.force_head = MLPOutput(d_model, 3)
        self.stress_head = MLPOutput(d_model, 6)

        # Initialize weights
        initialize_weights(self.transformer)
        initialize_output_heads(self.energy_head)
        initialize_output_heads(self.force_head)
        initialize_output_heads(self.stress_head)

        # Count parameters
        self.num_params = sum(
            p.numel()
            for name, p in self.named_parameters()
            if "atom_type_embedder" not in name
            and "pos_embedder" not in name
            and "frac_coords_embedder" not in name
        )

    def forward(self, batch):
        """
        Args:
            batch: A PyG batch with required attributes
        Returns:
            Tuple of (forces, energy, stress) tensors
        """

    def forward(self, batch):
        # Forward through the transformer
        transformer_output = self.transformer(batch)

        # Get the atom embeddings
        atom_embeddings = transformer_output["x"]  # [num_atoms, d_model]

        # Predict forces (per-atom is correct)
        forces = self.force_head(atom_embeddings)  # [num_atoms, 3]

        # Predict initial per-atom values
        per_atom_energy = self.energy_head(atom_embeddings)  # [num_atoms, 1]
        per_atom_stress = self.stress_head(atom_embeddings)  # [num_atoms, 6]

        # Aggregate to per-structure values using batch index
        batch_idx = transformer_output["batch"]  # [num_atoms]

        # Sum per-atom energies for each structure
        energy = scatter(per_atom_energy, batch_idx, dim=0, reduce="sum").squeeze(
            -1
        )  # [num_graphs]

        # Average per-atom stress for each structure
        stress = scatter(
            per_atom_stress, batch_idx, dim=0, reduce="mean"
        )  # [num_graphs, 6]

        return forces, energy, stress


class MetaADiTModels:
    def __init__(
        self,
        max_num_elements=119,
    ):
        """Initializes ADiT Models with a list of configurations.

        Args:
            max_num_elements (int): Maximum number of elements in the dataset.
        """
        # fmt: off
        self.configurations = [
            {"d_model": 64, "nhead": 1, "dim_feedforward": 256, "num_layers": 1}, # 0.063M params
            {"d_model": 64, "nhead": 1, "dim_feedforward": 256, "num_layers": 2}, # 0.113M params
            {"d_model": 64, "nhead": 1, "dim_feedforward": 256, "num_layers": 3}, # 0.163M params
            {"d_model": 128, "nhead": 2, "dim_feedforward": 512, "num_layers": 3}, # 0.646M params
            {"d_model": 192, "nhead": 3, "dim_feedforward": 768, "num_layers": 5}, # 2.338M params
            {"d_model": 192, "nhead": 4, "dim_feedforward": 768, "num_layers": 6}, # 2.783M params
            {"d_model": 256, "nhead": 4, "dim_feedforward": 1024, "num_layers": 6}, # 2.338M params
            {"d_model": 320, "nhead": 5, "dim_feedforward": 1280, "num_layers": 7}, # 8.943M params
        ]
        # fmt: on

        self.max_num_elements = max_num_elements
        # Default values for parameters that remain constant across configurations
        self.default_activation = "gelu"
        self.default_dropout = 0.1
        self.default_norm_first = True
        self.default_bias = True

    def __getitem__(self, idx):
        """Retrieves ADiT model corresponding to the configuration at index `idx`.

        Args:
            idx (int): Index of the desired configuration.

        Returns:
            ADiTS2EFSModel: An instance of the ADiT model with the specified configuration.

        Raises:
            IndexError: If the index is out of range.
        """
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range")

        config = self.configurations[idx]

        return ADiTS2EFSModel(
            max_num_elements=self.max_num_elements,
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            activation=self.default_activation,
            dropout=self.default_dropout,
            norm_first=self.default_norm_first,
            bias=self.default_bias,
            num_layers=config["num_layers"],
        )

    def __len__(self):
        """Returns the number of configurations."""
        return len(self.configurations)

    def __iter__(self):
        """Allows iteration over all ADiT models."""
        for idx in range(len(self.configurations)):
            yield self[idx]
