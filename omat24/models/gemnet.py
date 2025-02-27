import torch
import torch.nn as nn
from torch_scatter import scatter
from fairchem.core.models.gemnet.gemnet import GemNetT


class MLPReadout(nn.Module):
    """A simple MLP readout: node_emb -> linear -> tanh -> linear -> out_dim."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, node_emb):
        return self.net(node_emb)


class GemNetS2EF(nn.Module):
    """Wraps a GemNetT backbone + readouts for stress.
    Returns forces, total_energy_per_structure, stress_per_structure."""

    def __init__(self, backbone_config):
        super().__init__()
        # 1) Create backbone
        self.backbone = GemNetT(**backbone_config)
        self.name = "GemNetT"

        # 2) Get embedding dimension from config
        self.hidden_channels = backbone_config.get("emb_size_atom", 64)

        # Create our own atom embedding layer for stress computation
        self.atom_embedding = nn.Embedding(
            backbone_config.get("num_atom", 119),
            self.hidden_channels // 2,  # Half size, will concatenate with position info
        )

        # Linear layer to combine atom embedding and position features
        self.feature_combiner = nn.Linear(
            self.hidden_channels // 2 + 3, self.hidden_channels  # atom_emb + pos (3D)
        )

        # 3) Define stress readout
        self.stress_head = MLPReadout(self.hidden_channels, 6)

        # Calculate number of parameters (excluding embeddings)
        self.num_params = sum(
            p.numel()
            for name, p in self.named_parameters()
            if "atom_embedding" not in name.lower()
        )

    def forward(self, batch):
        """
        Args:
          batch: A PyG batch with .atomic_numbers, .pos, etc.
        Returns:
          forces  -> [N_atoms, 3]
          energy  -> [N_structures]
          stress  -> [N_structures, 6]
        """
        # Extract the batch size and number of atoms
        structure_index = batch.batch  # Maps each atom to its structure

        # Run the backbone forward pass for energy and forces
        outputs = self.backbone(batch)
        energy = outputs["energy"].squeeze(-1)  # [N_structures]
        forces = outputs["forces"]  # [N_atoms, 3]

        # Create atom embeddings using our own layer
        atom_emb = self.atom_embedding(batch.atomic_numbers)

        # Normalize positions to prevent dominating the features
        pos_normalized = batch.pos * 0.1

        # Combine atom embeddings with position information
        combined_features = torch.cat([atom_emb, pos_normalized], dim=1)
        h = self.feature_combiner(combined_features)
        h = torch.relu(h)  # Add activation

        # Apply stress head
        s_per_node = self.stress_head(h)  # [N_atoms, 6]

        # Aggregate stress by structure
        stress = scatter(
            s_per_node, structure_index, dim=0, reduce="add"
        )  # [N_structures, 6]

        return forces, energy, stress


class MetaGemNetTModels:
    """
    Meta-class enumerating different GemNetT backbone configurations.
    """

    def __init__(self, device: torch.device):
        self.configurations = [
            # Small model (~1M params) with optimized settings for quick testing
            {
                "num_spherical": 3,
                "num_radial": 6,
                "num_blocks": 2,
                "emb_size_atom": 4,
                "emb_size_edge": 4,
                "emb_size_trip": 4,
                "emb_size_rbf": 4,
                "emb_size_cbf": 4,
                "emb_size_bil_trip": 4,
                "num_before_skip": 1,
                "num_after_skip": 1,
                "num_concat": 1,
                "num_atom": 119,
                "cutoff": 5.0,
                "max_neighbors": 50,
                "otf_graph": False,
                "regress_forces": True,
                "direct_forces": True,
                "use_pbc": False,
                "use_pbc_single": False,
                "extensive": True,
                "output_init": "HeOrthogonal",
                "activation": "swish",
            }
        ]
        self.device = device

    def __getitem__(self, idx: int) -> GemNetS2EF:
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range")
        config = self.configurations[idx]
        model = GemNetS2EF(config)
        model.to(self.device)
        return model

    def __len__(self) -> int:
        return len(self.configurations)

    def __iter__(self):
        for idx in range(len(self.configurations)):
            yield self[idx]
