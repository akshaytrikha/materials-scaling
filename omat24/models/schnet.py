from math import pi as PI
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch_scatter import scatter
from torch_geometric.nn import radius_graph


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
            {
                "hidden_channels": 64,
                "num_filters": 64,
                "num_interactions": 3,
                "num_gaussians": 50,
            },
            {
                "hidden_channels": 128,
                "num_filters": 128,
                "num_interactions": 6,
                "num_gaussians": 50,
            },
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


class InteractionBlock(nn.Module):
    """Interaction block for updating node embeddings based on pairwise distances."""

    def __init__(
        self, hidden_channels, num_filters, num_gaussians, cutoff, device="cpu"
    ):
        """Args:

        hidden_channels (int, optional): Hidden embedding size - default is 128
        num_filters (int, optional): The number of filters to use - default is 128
        num_gaussians (int, optional): The number of gaussians :math:`\mu` - deault is 50
        num_filters (int): The number of filters to use.
        """
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

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, edge_attr):
        """Forward pass through the block

        Args:
            h (Tensor): Node features with shape [number of nodes, num node features]
            edge_index (LongTensor): edge indices with shape [2, num edges] as 1st row is source, 2nd row is target
            edge_weight (Tensor): Edge weights with shape [num edges] - represents importance of edges
            edge_attr (Tensor): Edge features with shape [num edges, num edge features] - contains richer info about edges

        Returns:
            Tensor: The updated node features.
        """
        h = self.conv(h, edge_index, edge_weight, edge_attr)  # TODO: fill in this line
        h = self.activation(h)
        h = self.linear(h)

        return h


class CFConv(nn.Module):
    """Continuous-filter convolution layer"""

    def __init__(
        self, in_channels, out_channels, num_filters, nn, cutoff, device="cpu"
    ):
        """Args:
        in_channels (int): The number of input channels
        out_channels (int): The number of output channels
        num_filters (int): The number of filters to use - default is 128
        nn (torch.nn.Module): The neural network to use
        cutoff (float): Cutoff distance in Angstroms for interatomic interactions
        device (str): device to load data on
        """
        super().__init__()
        self.lin1 = Linear(in_channels, num_filters, bias=False, device=device)
        # self.lin1.weight = self.lin1.weight.to(device)
        self.lin2 = Linear(num_filters, out_channels, device=device)
        self.nn = nn.to(device)
        self.cutoff = cutoff
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters"""
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, edge_attr):
        """Forward pass through the layer

        Args:
            h (Tensor): Node features with shape [number of nodes, num node features]
            edge_index (LongTensor): edge indices with shape [2, num edges]
                                    as 1st row is source, 2nd row is target
            edge_weight (Tensor): Edge weights with shape [num edges] -
                                represents importance of edges
            edge_attr (Tensor): Edge features with shape [num edges, num edge features]
                                contains richer info about edges

        Returns:
            Tensor: The updated node features
        """
        # first linear layer
        h = self.lin1(h)

        #  0.5 cos(x) + 1 shifts range from [-1, 1] to [0, 1]
        # convert edge_weight to radians and normalize values by dividing by self.cutoff
        c = 0.5 * (torch.cos((edge_weight * PI) / self.cutoff) + 1)  # TODO:

        # apply nn to the edge attributes to compute filters
        # has shape [num edges, num_filters]
        W = c.unsqueeze(1) * self.nn(edge_attr)

        # apply filters to the node features
        # h[edge_index[1]] is to get features of destination nodes for every edge
        # h = h[edge_index[1]] * W * c.unsqueeze(-1) # TODO: perform graph convolution
        h = scatter(
            src=h[edge_index[1]] * W,
            index=edge_index[0],
            dim=0,  # scattering along edge dimension (rows)
        )

        # second linear layer
        h = self.lin2(h)

        return h


class GaussianSmearing(nn.Module):
    """Gaussian smearing module for converting pairwise distances :math:`d_{ij}`"""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, device="cpu"):
        super().__init__()

        # self.device = device

        offset = torch.linspace(start, stop, num_gaussians).to(device)
        self.gamma = 0.5 / (offset[1] - offset[0]).item() ** 2

        # sets the attribute self.offset = offset
        self.register_buffer("offset", offset)

    def forward(self, dist):
        """Forward pass through the layer defined in the paper"""
        # TODO: implement this function
        # TODO: why does broadcasting work here?
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

    This version of SchNet accepts input data either as a dictionary (with keys
    'atomic_numbers' and 'positions') or as a PyG Data object (with attributes
    atomic_numbers, pos, and optionally edge_index). The forward pass computes the
    graph connectivity (if not provided), applies a series of interaction blocks,
    aggregates atom-wise contributions to yield a global energy, and computes forces
    as the negative gradient of the energy with respect to positions.
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
        # For dipole moment predictions, readout is fixed to "add"
        self.readout = "add" if dipole else readout
        self.dipole = dipole
        self.mean = mean
        self.std = std
        self.device = device

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
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus(device=device)
        self.lin2 = Linear(hidden_channels // 2, 1)
        self.reset_parameters()

        # Calculate number of parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, data):
        """
        Forward pass for SchNet.

        Args:
            data: Either a dictionary with keys "atomic_numbers" and "positions" or a PyG Data object.
                  For dict, keys should be:
                      "atomic_numbers": LongTensor of shape [N_atoms]
                      "positions": FloatTensor of shape [N_atoms, 3]
                  For PyG Data, attributes are assumed to be:
                      atomic_numbers, pos, (optionally edge_index and batch)

        Returns:
            energy: Tensor of shape [num_molecules] representing the predicted energy.
            forces: Tensor of shape [N_atoms, 3] representing the negative gradient of energy.
        """
        # Extract atomic numbers and positions
        if isinstance(data, dict):
            z = data["atomic_numbers"]
            pos = data["positions"]
            # When using dicts, assume a single molecule (or no batch)
            batch = torch.zeros(z.shape[0], dtype=torch.long, device=self.device)
        else:
            # Assume PyG Data (or Batch) object
            z = data.atomic_numbers
            pos = data.pos
            batch = (
                data.batch
                if hasattr(data, "batch")
                else torch.zeros(z.shape[0], dtype=torch.long, device=self.device)
            )

        # Ensure positions require gradients for force computation
        pos.requires_grad = True

        # Embed atomic numbers to get initial atom-wise features
        h = self.embedding(z)

        # Use precomputed edge_index if available; otherwise, compute it using radius_graph.
        if hasattr(data, "edge_index"):
            edge_index = data.edge_index
        else:
            if self.device != "cpu":
                edge_index = radius_graph(
                    pos.to("cpu"),
                    r=self.cutoff,
                    batch=batch,
                    max_num_neighbors=self.max_num_neighbors,
                ).to(self.device)
            else:
                edge_index = radius_graph(
                    pos,
                    r=self.cutoff,
                    batch=batch,
                    max_num_neighbors=self.max_num_neighbors,
                )

        # Compute pairwise distances for edges
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=1)
        # Expand the distances into a radial basis using Gaussian smearing
        edge_attr = self.distance_expansion(edge_weight)

        # Pass through a series of interaction blocks
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_weight, edge_attr)

        # Atom-wise energy contributions
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        # Aggregate per-atom energies into a global energy using scatter
        energy = scatter(h, batch, dim=0, reduce=self.readout).squeeze()

        # Compute forces as the negative gradient of energy with respect to positions
        forces = -torch.autograd.grad(energy.sum(), pos, create_graph=True)[0]

        # Dummy stress tensor
        stress = torch.zeros((6), device=self.device)

        return forces, energy.unsqueeze(0), stress
