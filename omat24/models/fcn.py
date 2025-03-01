# External
import torch
import torch.nn as nn

# Internal
from models.model_utils import MLPReadout


class MetaFCNModels:
    def __init__(self, vocab_size=119, use_factorized=False):
        self.configurations = [
            # 970 params (includes embedding layer)
            {"embedding_dim": 8, "hidden_dim": 16, "depth": 2},
            # 11.6k parameters (includes embedding layer)
            {"embedding_dim": 24, "hidden_dim": 48, "depth": 4},
            # 104k parameters (includes embedding layer)
            {"embedding_dim": 60, "hidden_dim": 108, "depth": 8},
            # 470k parameters (includes embedding layer)
            {"embedding_dim": 96, "hidden_dim": 192, "depth": 12},
            # 1M parameters (includes embedding layer)
            {"embedding_dim": 128, "hidden_dim": 256, "depth": 15},
            # # 4.3M parameters (includes embedding layer)
            # {"embedding_dim": 256, "hidden_dim": 512, "depth": 16},
            # # 10.9M parameters (includes embedding layer)
            # {"embedding_dim": 384, "hidden_dim": 768, "depth": 18},
        ]
        self.vocab_size = vocab_size
        self.use_factorized = use_factorized

    def __getitem__(self, idx):
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range")
        config = self.configurations[idx]
        return FCNModel(
            vocab_size=self.vocab_size,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            depth=config["depth"],
            use_factorized=self.use_factorized,
        )

    def __len__(self):
        return len(self.configurations)

    def __iter__(self):
        for idx in range(len(self.configurations)):
            yield self[idx]


class FCNModel(nn.Module):
    def __init__(
        self,
        vocab_size=119,
        embedding_dim=128,
        hidden_dim=256,
        depth=4,
        use_factorized=False,
    ):
        super(FCNModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_factorized = use_factorized
        self.name = "FCN"

        # Embedding for atomic numbers
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim
        )  # Assuming atomic numbers < 119

        # Initial layer
        if self.use_factorized:
            self.fc1 = nn.Linear(embedding_dim + 5, hidden_dim)
        else:
            self.fc1 = nn.Linear(embedding_dim + 3, hidden_dim)

        # Inner layers with residual connections
        self.inner_layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(),
            )
            self.inner_layers.append(layer)

        # Output layers using MLPReadout
        self.force_output = MLPReadout(hidden_dim, 3)  # Forces per atom
        self.energy_output = MLPReadout(hidden_dim, 1)  # Energy contributions per atom
        self.stress_output = MLPReadout(hidden_dim, 6)  # Stress contributions per atom

        # Calculate number of parameters
        self.num_params = sum(
            p.numel() for name, p in self.named_parameters() if "embedding" not in name
        )

    def forward(self, atomic_numbers, positions, distance_matrix=None, mask=None):
        """
        Args:
            atomic_numbers: Tensor of shape [batch_size, vocab_size]
            positions: Tensor of shape [batch_size, vocab_size, 3]
            distance_matrix: Factorized Inverse Distances [batch_size, vocab_size, 5]

        Returns:
            forces: Tensor of shape [batch_size, vocab_size, 3]
            energy: Tensor of shape [batch_size]
            stress: Tensor of shape [batch_size, 6]
        """
        # Create a mask for valid atoms (non-padded)
        new_mask = mask.unsqueeze(-1)  # Shape: [batch_size, vocab_size, 1]

        # Embed atomic numbers
        atomic_embeddings = self.embedding(
            atomic_numbers
        )  # [batch_size, vocab_size, embedding_dim]
        # Concatenate embeddings with positions or distances
        # print(atomic_embeddings.shape, positions.shape, distance_matrix.shape)
        if self.use_factorized:
            x = torch.cat(
                [atomic_embeddings, distance_matrix], dim=-1
            )  # [batch_size, vocab_size, embedding_dim + 5]
        else:
            x = torch.cat(
                [atomic_embeddings, positions], dim=-1
            )  # [batch_size, vocab_size, embedding_dim + 3]

        # Initial layer
        x = self.fc1(x)

        # Pass through inner layers with residual connections
        for layer in self.inner_layers:
            x_res = layer(x)
            x = x + x_res  # Residual connection

        # Predict forces
        forces = self.force_output(x)  # [batch_size, vocab_size, 3]
        forces = forces * new_mask.float()  # Mask padded atoms

        # Predict per-atom energy contributions and sum
        energy_contrib = self.energy_output(x).squeeze(-1)  # [batch_size, vocab_size]
        energy_contrib = energy_contrib * new_mask.squeeze(-1).float()
        energy = energy_contrib.sum(dim=1)  # [batch_size]

        # Predict per-atom stress contributions and sum
        stress_contrib = self.stress_output(x)  # [batch_size, vocab_size, 6]
        stress_contrib = stress_contrib * new_mask.float()
        stress = stress_contrib.sum(dim=1)  # [batch_size, 6]

        return forces, energy, stress
