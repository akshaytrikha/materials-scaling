# fcn.py

import torch
import torch.nn as nn

class FCNModel(nn.Module):
    def __init__(self, max_atoms=300, embedding_dim=128, hidden_dim=256):
        super(FCNModel, self).__init__()
        self.max_atoms = max_atoms
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding for atomic numbers
        self.embedding = nn.Embedding(119, embedding_dim)  # Assuming atomic numbers < 119

        # Fully connected layers
        self.fc1 = nn.Linear(embedding_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers
        self.force_output = nn.Linear(hidden_dim, 3)  # Forces per atom
        self.energy_output = nn.Linear(hidden_dim, 1)  # Energy contributions per atom
        self.stress_output = nn.Linear(hidden_dim, 6)  # Stress contributions per atom

    def forward(self, atomic_numbers, positions):
        """
        Args:
            atomic_numbers: Tensor of shape [batch_size, max_atoms]
            positions: Tensor of shape [batch_size, max_atoms, 3]

        Returns:
            forces: Tensor of shape [batch_size, max_atoms, 3]
            energy: Tensor of shape [batch_size]
            stress: Tensor of shape [batch_size, 6]
        """
        # Create a mask for valid atoms (non-padded)
        mask = (atomic_numbers != 0).unsqueeze(-1)  # Shape: [batch_size, max_atoms, 1]

        # Embed atomic numbers
        atomic_embeddings = self.embedding(atomic_numbers)  # [batch_size, max_atoms, embedding_dim]

        # Concatenate embeddings with positions
        x = torch.cat([atomic_embeddings, positions], dim=-1)  # [batch_size, max_atoms, embedding_dim + 3]

        # Pass through fully connected layers
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))

        # Predict forces
        forces = self.force_output(x)  # [batch_size, max_atoms, 3]
        forces = forces * mask.float()  # Mask padded atoms

        # Predict per-atom energy contributions and sum
        energy_contrib = self.energy_output(x).squeeze(-1)  # [batch_size, max_atoms]
        energy_contrib = energy_contrib * mask.squeeze(-1).float()
        energy = energy_contrib.sum(dim=1)  # [batch_size]

        # Predict per-atom stress contributions and sum
        stress_contrib = self.stress_output(x)  # [batch_size, max_atoms, 6]
        stress_contrib = stress_contrib * mask.float()
        stress = stress_contrib.sum(dim=1)  # [batch_size, 6]

        return forces, energy, stress
