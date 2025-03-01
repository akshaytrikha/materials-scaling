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
        self.num_params = sum(
            p.numel() for name, p in self.named_parameters() if "embedding" not in name
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
        # self.luis_base_config = {
        #     "regress_forces": True,
        #     "use_pbc": True,
        #     "use_pbc_single": True,
        #     "otf_graph": True,
        #     "enforce_max_neighbors_strictly": False,
        #     "max_neighbors": 20,
        #     "max_radius": 12.0,
        #     "max_num_elements": 119,
        #     "norm_type": "layer_norm_sh",
        #     "grid_resolution": 18,
        #     "num_sphere_samples": 128,
        #     "edge_channels": 128,
        #     "use_atom_edge_embedding": True,
        #     "share_atom_edge_embedding": False,
        #     "use_m_share_rad": False,
        #     "distance_function": "gaussian",
        #     "num_distance_basis": 512,
        #     "attn_activation": "silu",
        #     "use_s2_act_attn": False,
        #     "use_attn_renorm": True,
        #     "ffn_activation": "silu",
        #     "use_gate_act": False,
        #     "use_grid_mlp": True,
        #     "use_sep_s2_act": True,
        #     "alpha_drop": 0.1,
        #     "drop_path_rate": 0.1,
        #     "proj_drop": 0.0,
        #     "weight_init": "uniform",
        # }

        # # Create Luis configurations with varying parameters
        # self.luis_configurations = [
        #     # Luis 31M (ours 58M)
        #     {
        #         "num_layers": 8,
        #         "sphere_channels": 128,
        #         "attn_hidden_channels": 64,
        #         "num_heads": 8,
        #         "attn_alpha_channels": 64,
        #         "attn_value_channels": 16,
        #         "ffn_hidden_channels": 128,
        #         "lmax_list": [4],
        #         "mmax_list": [2],
        #     },
        #     # Luis 86M (ours 196M)
        #     {
        #         "num_layers": 10,
        #         "sphere_channels": 128,
        #         "attn_hidden_channels": 64,
        #         "num_heads": 8,
        #         "attn_alpha_channels": 64,
        #         "attn_value_channels": 16,
        #         "ffn_hidden_channels": 128,
        #         "lmax_list": [6],
        #         "mmax_list": [4],
        #     },
        #     # Luis 153M (ours 263M)
        #     {
        #         "num_layers": 20,
        #         "sphere_channels": 128,
        #         "attn_hidden_channels": 64,
        #         "num_heads": 8,
        #         "attn_alpha_channels": 64,
        #         "attn_value_channels": 16,
        #         "ffn_hidden_channels": 128,
        #         "lmax_list": [6],
        #         "mmax_list": [3],
        #     },
        # ]

        self.base_config = {
            "regress_forces": True,
            "use_pbc": True,
            "use_pbc_single": True,
            "otf_graph": True,
            "enforce_max_neighbors_strictly": False,
            "max_neighbors": 20,
            "max_radius": 12.0,
            "max_num_elements": 119,
            "attn_alpha_channels": 1,
            "attn_value_channels": 1,
            "norm_type": "layer_norm_sh",
            "lmax_list": [4],
            "mmax_list": [2],
            "grid_resolution": 18,
            "num_sphere_samples": 128,
            "use_atom_edge_embedding": True,
            "share_atom_edge_embedding": False,
            "use_m_share_rad": False,
            "distance_function": "gaussian",
            "num_distance_basis": 512,
            "attn_activation": "silu",
            "use_s2_act_attn": False,
            "use_attn_renorm": True,
            "ffn_activation": "silu",
            "use_gate_act": False,
            "use_grid_mlp": True,
            "use_sep_s2_act": True,
            "alpha_drop": 0.1,
            "drop_path_rate": 0.1,
            "proj_drop": 0.0,
            "weight_init": "uniform",
        }

        self.configurations = [
            # 3,168 params
            {
                "ffn_hidden_channels": 1,
                "edge_channels": 1,
                "sphere_channels": 1,
                "num_layers": 1,
                "attn_hidden_channels": 1,
                "num_heads": 1,
                "attn_alpha_channels": 1,
                "attn_value_channels": 1,
            },
            # 12,766 params
            {
                "ffn_hidden_channels": 2,
                "edge_channels": 2,
                "sphere_channels": 2,
                "num_layers": 2,
                "attn_hidden_channels": 2,
                "num_heads": 1,
                "attn_alpha_channels": 1,
                "attn_value_channels": 1,
            },
            # 94,545 params
            {
                "ffn_hidden_channels": 5,
                "edge_channels": 5,
                "sphere_channels": 5,
                "num_layers": 5,
                "attn_hidden_channels": 5,
                "num_heads": 2,
                "attn_alpha_channels": 3,
                "attn_value_channels": 2,
            },
            # 1,047,446 params
            {
                "ffn_hidden_channels": 16,
                "edge_channels": 16,
                "sphere_channels": 16,
                "num_layers": 10,
                "attn_hidden_channels": 12,
                "num_heads": 4,
                "attn_alpha_channels": 8,
                "attn_value_channels": 4,
            },
            # 10,158,806 params
            {
                "ffn_hidden_channels": 48,
                "edge_channels": 36,
                "sphere_channels": 36,
                "num_layers": 20,
                "attn_hidden_channels": 48,
                "num_heads": 6,
                "attn_alpha_channels": 24,
                "attn_value_channels": 12,
            },
            # 98,062,746 params
            {
                "ffn_hidden_channels": 96,
                "edge_channels": 96,
                "sphere_channels": 96,
                "num_layers": 40,
                "attn_hidden_channels": 96,
                "num_heads": 12,
                "attn_alpha_channels": 48,
                "attn_value_channels": 24,
            },
            # # 244,426,818 params
            # {
            #     "ffn_hidden_channels": 128,
            #     "edge_channels": 128,
            #     "sphere_channels": 128,
            #     "num_layers": 60,
            #     "attn_hidden_channels": 128,
            #     "num_heads": 16,
            #     "attn_alpha_channels": 64,
            #     "attn_value_channels": 32,
            # },
        ]

        self.device = device

    def __getitem__(self, idx: int) -> EquiformerS2EF:
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range")
        config = self.configurations[idx]
        config.update(self.base_config)
        model = EquiformerS2EF(config)
        model.to(self.device)
        return model

    def __len__(self) -> int:
        return len(self.configurations)

    def __iter__(self):
        for idx in range(len(self.configurations)):
            yield self[idx]
