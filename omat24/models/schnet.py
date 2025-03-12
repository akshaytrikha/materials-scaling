# External
from math import pi as PI
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch_scatter import scatter

# Internal
from models.model_utils import MLPOutput, apply_initialization, initialize_weights


class InteractionBlock(nn.Module):
    """Interaction block for updating node embeddings based on pairwise distances."""

    def __init__(
        self, hidden_channels, num_filters, num_gaussians, cutoff, device="cpu"
    ):
        super().__init__()
        # mlp computes filter applied to node features
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters, device=device),
            ShiftedSoftplus(device=device),
            Linear(num_filters, num_filters, device=device),
        )
        self.conv = CFConv(
            hidden_channels,
            hidden_channels,
            num_filters,
            self.mlp,
            cutoff,
            device=device,
        )
        self.activation = ShiftedSoftplus(device=device)
        self.linear = Linear(hidden_channels, hidden_channels, device=device)

        # Use standardized initialization
        for module in [self.mlp[0], self.mlp[2], self.linear]:
            initialize_weights(module)

    def forward(self, h, edge_index, edge_weight, edge_attr):
        """Forward pass through the block"""
        h = self.conv(h, edge_index, edge_weight, edge_attr)
        h = self.activation(h)
        h = self.linear(h)
        return h


class CFConv(nn.Module):
    """Continuous-filter convolution layer"""

    def __init__(
        self, in_channels, out_channels, num_filters, nn, cutoff, device="cpu"
    ):
        super().__init__()
        self.lin1 = Linear(in_channels, num_filters, bias=False, device=device)
        self.lin2 = Linear(num_filters, out_channels, device=device)
        self.nn = nn.to(device)
        self.cutoff = cutoff
        self.device = device

        # Use standardized initialization
        initialize_weights(self.lin1)
        initialize_weights(self.lin2)

    def forward(self, h, edge_index, edge_weight, edge_attr):
        """Forward pass through the layer"""
        # first linear layer
        h = self.lin1(h)

        #  0.5 cos(x) + 1 shifts range from [-1, 1] to [0, 1]
        c = 0.5 * (torch.cos((edge_weight * PI) / self.cutoff) + 1)

        # apply nn to the edge attributes to compute filters
        W = c.unsqueeze(1) * self.nn(edge_attr)

        # apply filters to the node features
        h = scatter(
            src=h[edge_index[1]] * W,
            index=edge_index[0],
            dim=0,
        )

        # second linear layer
        h = self.lin2(h)

        return h


class GaussianSmearing(nn.Module):
    """Gaussian smearing module for converting pairwise distances"""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, device="cpu"):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians).to(device)
        self.gamma = 0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        difference = dist.view(-1, 1) - self.offset.view(1, -1)
        exponent = -self.gamma * difference.pow(2)
        return exponent.exp()


class ShiftedSoftplus(nn.Module):
    """Shifted softplus activation function"""

    def __init__(self, device="cpu"):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0, device=device)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class SchNet(nn.Module):
    """
    SchNet model for predicting energy and forces from atomistic systems.
    """

    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=5.0,
        max_num_neighbors=32,
        readout="add",
        dipole=False,
        mean=None,
        std=None,
        device="cpu",
    ):
        super(SchNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = "add" if dipole else readout
        self.dipole = dipole
        self.mean = mean
        self.std = std
        self.device = device
        self.name = "SchNet"

        # Embedding for atomic numbers (assumes atomic numbers < 100)
        self.embedding = nn.Embedding(100, hidden_channels)
        self.embedding.to(device)

        # Radial basis function expansion for distances
        self.distance_expansion = GaussianSmearing(
            start=0.0, stop=cutoff, num_gaussians=num_gaussians, device=device
        )

        # Build interaction blocks
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels, num_filters, num_gaussians, cutoff, device=device
                )
                for _ in range(num_interactions)
            ]
        )

        # Atom-wise readout layers
        self.energy_head = MLPOutput(hidden_channels, 1)
        self.stress_head = MLPOutput(hidden_channels, 6)

        # Apply standardized initialization
        apply_initialization(self)

        # Calculate number of parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, atomic_numbers, positions, edge_index, structure_index):
        """
        Forward pass for SchNet.
        """
        # --- Standard SchNet embedding & interactions ---
        positions.requires_grad = True
        h = self.embedding(atomic_numbers)
        edge_index = edge_index
        row, col = edge_index
        edge_weight = (positions[row] - positions[col]).norm(dim=1)
        edge_attr = self.distance_expansion(edge_weight)
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_weight, edge_attr)

        # --- Now we do separate readouts for E, stress, etc. ---
        # (1) Energy
        e_per_atom = self.energy_head(h)  # [N, 1]
        energy = scatter(e_per_atom, structure_index, dim=0, reduce="add").squeeze(
            -1
        )  # [B]

        # (2) Forces from autograd
        forces = -torch.autograd.grad(energy.sum(), positions, create_graph=True)[0]

        # (3) Stress
        stress_contrib = self.stress_head(h)  # e.g. [N, 6]
        stress = scatter(stress_contrib, structure_index, dim=0, reduce="add")  # [B, 6]

        return forces, energy, stress


class MetaSchNetModels:
    def __init__(self, device):
        """
        Initializes a list of SchNet model configurations.

        Args:
            device (str): Device to run the models on.
        """
        # Define a list of configurations with varying hyperparameters.
        # You can add or remove configurations as needed.
        self.configurations = [
            # 68,423 params
            {
                "hidden_channels": 64,
                "num_filters": 64,
                "num_interactions": 3,
                "num_gaussians": 50,
            },
            # 456,583 params
            {
                "hidden_channels": 128,
                "num_filters": 128,
                "num_interactions": 6,
                "num_gaussians": 50,
            },
            # 2,267,911
            {
                "hidden_channels": 256,
                "num_filters": 256,
                "num_interactions": 8,
                "num_gaussians": 50,
            },
        ]
        self.device = device

    def __getitem__(self, idx):
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range")
        config = self.configurations[idx]

        return SchNet(
            hidden_channels=config["hidden_channels"],
            num_filters=config["num_filters"],
            num_interactions=config["num_interactions"],
            num_gaussians=config["num_gaussians"],
            cutoff=5.0,
            max_num_neighbors=32,  # default maximum number of neighbors
            readout="add",
            dipole=False,
            device=self.device,
        )

    def __len__(self):
        return len(self.configurations)

    def __iter__(self):
        for idx in range(len(self.configurations)):
            yield self[idx]
