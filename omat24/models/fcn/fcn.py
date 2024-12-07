# External
import torch
import torch.nn as nn
import yaml


class MetaFCNModels:
    def __init__(self, config_file="./models/fcn/fcn_config.yaml", vocab_size=119):
        self.vocab_size = vocab_size
        self.configurations = self.load_configurations(config_file)

    def load_configurations(self, config_file):
        try:
            with open(config_file, "r") as f:
                configurations = yaml.safe_load(f)
            return configurations
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_file} not found.")
        except yaml.YAMLError:
            raise ValueError(f"Error decoding YAML in {config_file}.")

    def __getitem__(self, idx):
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range")
        config = self.configurations[idx]
        return {
            "model": FCNModel(
                vocab_size=self.vocab_size,
                embedding_dim=config["embedding_dim"],
                hidden_dim=config["hidden_dim"],
                depth=config["depth"],
            ),
            "num_epochs": config["num_epochs"],
        }

    def __len__(self):
        return len(self.configurations)

    def __iter__(self):
        for idx in range(len(self.configurations)):
            yield self[idx]


class FCNModel(nn.Module):
    def __init__(
        self, vocab_size=119, embedding_dim=128, hidden_dim=256, depth=4, num_epochs=1
    ):
        super(FCNModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_epochs = num_epochs

        # Embedding for atomic numbers
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim
        )  # Assuming atomic numbers < 119

        # Initial layer
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

        # Output layers
        self.force_output = nn.Linear(hidden_dim, 3)  # Forces per atom
        self.energy_output = nn.Linear(hidden_dim, 1)  # Energy contributions per atom
        self.stress_output = nn.Linear(hidden_dim, 6)  # Stress contributions per atom

        # Calculate number of parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, atomic_numbers, positions):
        """
        Args:
            atomic_numbers: Tensor of shape [batch_size, vocab_size]
            positions: Tensor of shape [batch_size, vocab_size, 3]

        Returns:
            forces: Tensor of shape [batch_size, vocab_size, 3]
            energy: Tensor of shape [batch_size]
            stress: Tensor of shape [batch_size, 6]
        """
        # Create a mask for valid atoms (non-padded)
        mask = (atomic_numbers != 0).unsqueeze(-1)  # Shape: [batch_size, vocab_size, 1]

        # Embed atomic numbers
        atomic_embeddings = self.embedding(
            atomic_numbers
        )  # [batch_size, vocab_size, embedding_dim]
        # Concatenate embeddings with positions
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
        forces = forces * mask.float()  # Mask padded atoms

        # Predict per-atom energy contributions and sum
        energy_contrib = self.energy_output(x).squeeze(-1)  # [batch_size, vocab_size]
        energy_contrib = energy_contrib * mask.squeeze(-1).float()
        energy = energy_contrib.sum(dim=1)  # [batch_size]

        # Predict per-atom stress contributions and sum
        stress_contrib = self.stress_output(x)  # [batch_size, vocab_size, 6]
        stress_contrib = stress_contrib * mask.float()
        stress = stress_contrib.sum(dim=1)  # [batch_size, 6]

        return forces, energy, stress
