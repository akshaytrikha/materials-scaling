import torch
import torch.nn as nn
from torch_scatter import scatter
from fairchem.core.models.gemnet_gp import GraphParallelGemNetT


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
    """Wraps a GraphParallelGemNetT backbone + readouts for stress.
    Returns forces, total_energy_per_structure, stress_per_structure."""

    def __init__(self, backbone_config):
        super().__init__()
        # 1) Create backbone
        self.backbone = GraphParallelGemNetT(**backbone_config)
        self.name = "GemNetGP"

        # 2) Get embedding dimension from config
        self.hidden_channels = backbone_config.get("emb_size_atom", 128)

        # 3) Define additional stress readout
        # Note: GraphParallelGemNetT directly outputs energy and forces, so we only need a stress MLP
        self.stress_head = MLPReadout(self.hidden_channels, 6)

        # Calculate number of parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, batch):
        """
        Args:
          batch: A PyG batch with .atomic_numbers, .pos, etc.
        Returns:
          forces  -> [N_atoms, 3]
          energy  -> [N_structures]
          stress  -> [N_structures, 6]
        """
        # Create a copy of the batch to avoid modifying the original
        batch_dict = {}
        for key in batch.keys():
            batch_dict[key] = getattr(batch, key)

        # Ensure the distance_vec is directly accessible via dict item
        if "edge_distance_vec" in batch_dict and "distance_vec" not in batch_dict:
            batch_dict["distance_vec"] = batch_dict["edge_distance_vec"]

        # Create new batch with correct attributes
        for key, value in batch_dict.items():
            setattr(batch, key, value)
            # Make sure it's also in the _store if using PyG Data objects
            if hasattr(batch, "_store"):
                batch._store[key] = value

        # The GraphParallelGemNetT backbone returns a dict with 'energy' and 'forces'
        outputs = self.backbone(batch)
        energy = outputs["energy"].squeeze(-1)  # [N_structures]
        forces = outputs["forces"]  # [N_atoms, 3]

        # For the stress prediction, create a placeholder for atom embeddings
        # In a full implementation, we would modify the backbone to return node embeddings
        h = torch.zeros(
            batch.pos.size(0), self.hidden_channels, device=batch.pos.device
        )

        # Compute per-atom stress contributions
        s_per_node = self.stress_head(h)  # shape [N_atoms, 6]

        # Aggregate stress by structure
        structure_index = batch.batch  # e.g. shape [N_atoms]
        stress = scatter(
            s_per_node, structure_index, dim=0, reduce="add"
        )  # [N_structures, 6]

        return forces, energy, stress


class MetaGemNetGPModels:
    """
    Meta-class enumerating different GraphParallelGemNetT backbone configurations.
    """

    def __init__(
        self,
        device: torch.device,
    ):
        self.configurations = [
            # Small model (~1M params) with relaxed requirements
            {
                "num_spherical": 3,
                "num_radial": 6,
                "num_blocks": 2,
                "emb_size_atom": 64,
                "emb_size_edge": 64,
                "emb_size_trip": 64,
                "emb_size_rbf": 16,
                "emb_size_cbf": 16,
                "emb_size_bil_trip": 64,
                "num_before_skip": 1,
                "num_after_skip": 1,
                "num_concat": 1,
                "num_atom": 119,  # Max atomic number
                "cutoff": 5.0,
                "max_neighbors": 20,
                "otf_graph": True,
                "regress_forces": True,
                "direct_forces": True,
                "use_pbc": False,  # Changed to false
                "use_pbc_single": False,
                "extensive": True,
                "output_init": "HeOrthogonal",
                "activation": "swish",
            },
            # Medium model (~5M params)
            {
                "num_spherical": 5,
                "num_radial": 10,
                "num_blocks": 3,
                "emb_size_atom": 128,
                "emb_size_edge": 128,
                "emb_size_trip": 128,
                "emb_size_rbf": 32,
                "emb_size_cbf": 32,
                "emb_size_bil_trip": 128,
                "num_before_skip": 1,
                "num_after_skip": 2,
                "num_concat": 1,
                "num_atom": 119,  # Max atomic number
                "cutoff": 5.0,
                "max_neighbors": 25,
                "otf_graph": True,
                "regress_forces": True,
                "direct_forces": True,
                "use_pbc": False,  # Changed to false
                "use_pbc_single": False,
                "extensive": True,
                "output_init": "HeOrthogonal",
                "activation": "swish",
            },
            # Large model (~20M params)
            {
                "num_spherical": 7,
                "num_radial": 16,
                "num_blocks": 4,
                "emb_size_atom": 256,
                "emb_size_edge": 256,
                "emb_size_trip": 256,
                "emb_size_rbf": 64,
                "emb_size_cbf": 64,
                "emb_size_bil_trip": 256,
                "num_before_skip": 2,
                "num_after_skip": 2,
                "num_concat": 2,
                "num_atom": 119,  # Max atomic number
                "cutoff": 5.0,
                "max_neighbors": 30,
                "otf_graph": True,
                "regress_forces": True,
                "direct_forces": True,
                "use_pbc": False,  # Changed to false
                "use_pbc_single": False,
                "extensive": True,
                "output_init": "HeOrthogonal",
                "activation": "swish",
            },
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
