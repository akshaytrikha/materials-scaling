import torch
import torch.nn as nn
from torch_scatter import scatter
from fairchem.core.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone


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


class EquiformerS2EF(nn.Module):
    """Wraps an EquiformerV2 backbone + readouts for per-atom energies, forces, and stress.
    Returns f_per_node, total_energy_per_structure, stress_per_structure."""

    def __init__(self, backbone_config):
        super().__init__()
        # 1) Create backbone
        self.backbone = EquiformerV2Backbone(**backbone_config)
        self.name = "EquiformerV2"

        # 2) Determine the embedding dimension based on the backbone configuration
        # For EquiformerV2, the node embedding dimension depends on the configuration
        # We'll calculate it based on lmax_list and sphere_channels
        self.lmax_list = backbone_config["lmax_list"]
        self.sphere_channels = backbone_config["sphere_channels"]

        # Calculate the total number of coefficients in the spherical harmonic representation
        total_coeffs = sum((lmax + 1) ** 2 for lmax in self.lmax_list)
        self.in_dim = total_coeffs * self.sphere_channels

        # 3) Define output heads (MLPs)
        self.energy_head = MLPReadout(self.in_dim, 1)
        self.force_head = MLPReadout(self.in_dim, 3)
        self.stress_head = MLPReadout(self.in_dim, 6)

        # Calculate number of parameters
        self.num_params = sum(p.numel() for name, p in self.named_parameters())

    def forward(self, batch):
        """
        Args:
          batch: A PyG batch with .atomic_numbers, .pos, etc.
        Returns:
          forces  -> [N_atoms, 3]
          energy  -> [N_structures]
          stress  -> [N_structures, 6]
        """
        out = self.backbone(batch)  # returns {"node_embedding", "graph"}
        node_emb = out["node_embedding"]
        graph = out["graph"]

        # Process the node embeddings to get a flat tensor
        # node_emb is an SO3_Embedding object with node_emb.embedding of shape [N_atoms, total_coeffs, sphere_channels]
        # We need to reshape this to [N_atoms, total_coeffs * sphere_channels]
        x_shape = node_emb.embedding.shape
        x = node_emb.embedding.reshape(x_shape[0], -1)  # Reshape to [N_atoms, in_dim]

        # Summation or scatter by structure index for energy
        e_per_node = self.energy_head(x).squeeze(-1)  # shape [N_atoms]
        structure_index = graph.batch_full  # e.g. shape [N_atoms]
        energy = scatter(
            e_per_node, structure_index, dim=0, reduce="add"
        )  # [N_structures]

        # Forces
        forces = self.force_head(x)  # shape [N_atoms, 3]

        # Stress
        s_per_node = self.stress_head(x)  # shape [N_atoms, 6]
        stress = scatter(
            s_per_node, structure_index, dim=0, reduce="add"
        )  # [N_structures, 6]

        return forces, energy, stress


class MetaEquiformerV2Models:
    """
    Meta-class enumerating different EquiformerV2Backbone configurations.
    We return the raw Equiformer backbone, so you can wrap it with EquiformerS2EF
    using your own chosen readout or the MLPReadout above.
    """

    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.configurations = [
            # # 1306 params
            # {
            #     "regress_forces": True,
            #     "num_layers": 2,
            #     "sphere_channels": 1,
            #     "attn_hidden_channels": 1,
            #     "num_heads": 1,
            #     "attn_alpha_channels": 1,
            #     "attn_value_channels": 1,
            #     "ffn_hidden_channels": 1,
            #     "norm_type": "rms_norm_sh",
            #     "lmax_list": [0],
            #     "mmax_list": [0],
            #     "grid_resolution": None,
            #     "num_sphere_samples": 1,
            #     "edge_channels": 1,
            #     "max_num_elements": 119,
            # },
            # Luis 31M params
            {
                "num_layers": 8,
                "sphere_channels": 32,
                "attn_hidden_channels": 16,
                "num_heads": 8,
                "attn_alpha_channels": 16,
                "attn_value_channels": 4,
                "ffn_hidden_channels": 32,
                "norm_type": "layer_norm_sh",
                "lmax_list": [4],
                "mmax_list": [2],
                "grid_resolution": 18,
                "num_sphere_samples": 32,
                "edge_channels": 32,
                "max_num_elements": 119,
                "activation_checkpoint": True,  # for memory optimization
            },
            # # Luis 31M params
            # {
            #     "num_layers": 8,
            #     "sphere_channels": 128,
            #     "attn_hidden_channels": 64,
            #     "num_heads": 8,
            #     "attn_alpha_channels": 64,
            #     "attn_value_channels": 16,
            #     "ffn_hidden_channels": 128,
            #     "norm_type": "layer_norm_sh",
            #     "lmax_list": [4],
            #     "mmax_list": [2],
            #     "grid_resolution": 18,
            #     "num_sphere_samples": 128,
            #     "edge_channels": 128,
            #     "max_num_elements": 119,
            #     "activation_checkpoint": True,  # for memory optimization
            # },
            # # Luis 86M params
            # {
            #     "num_layers": 10,
            #     "sphere_channels": 128,
            #     "attn_hidden_channels": 64,
            #     "num_heads": 8,
            #     "attn_alpha_channels": 64,
            #     "attn_value_channels": 16,
            #     "ffn_hidden_channels": 128,
            #     "norm_type": "layer_norm_sh",
            #     "lmax_list": [6],
            #     "mmax_list": [4],
            #     "grid_resolution": 18,
            #     "num_sphere_samples": 128,
            #     "edge_channels": 128,
            #     "max_num_elements": 119,
            #     "activation_checkpoint": True,  # for memory optimization
            # },
            # # Luis 153M params
            # {
            #     "num_layers": 20,
            #     "sphere_channels": 128,
            #     "attn_hidden_channels": 64,
            #     "num_heads": 8,
            #     "attn_alpha_channels": 64,
            #     "attn_value_channels": 16,
            #     "ffn_hidden_channels": 128,
            #     "norm_type": "layer_norm_sh",
            #     "lmax_list": [6],
            #     "mmax_list": [3],
            #     "grid_resolution": 18,
            #     "num_sphere_samples": 128,
            #     "edge_channels": 128,
            #     "max_num_elements": 119,
            #     "activation_checkpoint": True,  # for memory optimization
            # },
        ]
        self.device = device

    def __getitem__(self, idx: int) -> EquiformerS2EF:
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range")
        config = self.configurations[idx]
        model = EquiformerS2EF(config)
        model.to(self.device)
        return model

    def __len__(self) -> int:
        return len(self.configurations)

    def __iter__(self):
        for idx in range(len(self.configurations)):
            yield self[idx]
