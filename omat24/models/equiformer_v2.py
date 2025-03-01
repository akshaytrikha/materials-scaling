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
        # self.configurations = [
        #     # Luis 31M (ours 58M)
        #     {
        #         "regress_forces": True,
        #         "use_pbc": True,
        #         "use_pbc_single": True,
        #         "otf_graph": True,
        #         "enforce_max_neighbors_strictly": False,
        #         "max_neighbors": 20,
        #         "max_radius": 12.0,
        #         "max_num_elements": 119,
        #         "num_layers": 8,
        #         "sphere_channels": 128,
        #         "attn_hidden_channels": 64,
        #         "num_heads": 8,
        #         "attn_alpha_channels": 64,
        #         "attn_value_channels": 16,
        #         "ffn_hidden_channels": 128,
        #         "norm_type": "layer_norm_sh",
        #         "lmax_list": [4],
        #         "mmax_list": [2],
        #         "grid_resolution": 18,
        #         "num_sphere_samples": 128,
        #         "edge_channels": 128,
        #         "use_atom_edge_embedding": True,
        #         "share_atom_edge_embedding": False,
        #         "use_m_share_rad": False,
        #         "distance_function": "gaussian",
        #         "num_distance_basis": 512,
        #         "attn_activation": "silu",
        #         "use_s2_act_attn": False,
        #         "use_attn_renorm": True,
        #         "ffn_activation": "silu",
        #         "use_gate_act": False,
        #         "use_grid_mlp": True,
        #         "use_sep_s2_act": True,
        #         "alpha_drop": 0.1,
        #         "drop_path_rate": 0.1,
        #         "proj_drop": 0.0,
        #         "weight_init": "uniform",
        #     },
        #     # Luis 86M (ours 196M)
        #     {
        #         "regress_forces": True,
        #         "use_pbc": True,
        #         "use_pbc_single": True,
        #         "otf_graph": True,
        #         "enforce_max_neighbors_strictly": False,
        #         "max_neighbors": 20,
        #         "max_radius": 12.0,
        #         "max_num_elements": 119,
        #         "avg_num_nodes": 31.17,
        #         "avg_degree": 61.95,
        #         "num_layers": 10,  # Updated from 8 to 10
        #         "sphere_channels": 128,
        #         "attn_hidden_channels": 64,
        #         "num_heads": 8,
        #         "attn_alpha_channels": 64,
        #         "attn_value_channels": 16,
        #         "ffn_hidden_channels": 128,
        #         "norm_type": "layer_norm_sh",
        #         "lmax_list": [6],  # Updated from 4 to 6
        #         "mmax_list": [4],  # Updated from 2 to 4
        #         "grid_resolution": 18,
        #         "num_sphere_samples": 128,
        #         "edge_channels": 128,
        #         "use_atom_edge_embedding": True,
        #         "share_atom_edge_embedding": False,
        #         "use_m_share_rad": False,
        #         "distance_function": "gaussian",
        #         "num_distance_basis": 512,
        #         "attn_activation": "silu",
        #         "use_s2_act_attn": False,
        #         "use_attn_renorm": True,
        #         "ffn_activation": "silu",
        #         "use_gate_act": False,
        #         "use_grid_mlp": True,
        #         "use_sep_s2_act": True,
        #         "alpha_drop": 0.1,
        #         "drop_path_rate": 0.1,
        #         "proj_drop": 0.0,
        #         "weight_init": "uniform",
        #     },
        #     # Luis 153M (ours 263M)
        #     {
        #         "regress_forces": True,
        #         "use_pbc": True,
        #         "use_pbc_single": True,
        #         "otf_graph": True,
        #         "enforce_max_neighbors_strictly": False,
        #         "max_neighbors": 20,
        #         "max_radius": 12.0,
        #         "max_num_elements": 119,
        #         "avg_num_nodes": 31.17,
        #         "avg_degree": 61.95,
        #         "num_layers": 20,  # Increased to 20 from previous 10
        #         "sphere_channels": 128,
        #         "attn_hidden_channels": 64,
        #         "num_heads": 8,
        #         "attn_alpha_channels": 64,
        #         "attn_value_channels": 16,
        #         "ffn_hidden_channels": 128,
        #         "norm_type": "layer_norm_sh",
        #         "lmax_list": [6],  # Same as previous
        #         "mmax_list": [3],  # Changed from 4 to 3
        #         "grid_resolution": 18,
        #         "num_sphere_samples": 128,
        #         "edge_channels": 128,
        #         "use_atom_edge_embedding": True,
        #         "distance_function": "gaussian",
        #         "num_distance_basis": 512,
        #         "attn_activation": "silu",
        #         "use_s2_act_attn": False,
        #         "ffn_activation": "silu",
        #         "use_gate_act": False,
        #         "use_grid_mlp": True,
        #         "alpha_drop": 0.1,
        #         "drop_path_rate": 0.1,
        #         "proj_drop": 0.0,
        #         "weight_init": "uniform",
        #     },
        # ]

        self.configurations = [
            # 1,301 params
            {
                "regress_forces": True,
                "use_pbc": True,
                "use_pbc_single": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 119,
                "num_layers": 1,
                "sphere_channels": 1,
                "attn_hidden_channels": 1,
                "num_heads": 1,
                "attn_alpha_channels": 1,
                "attn_value_channels": 1,
                "ffn_hidden_channels": 2,
                "norm_type": "layer_norm_sh",
                "lmax_list": [0],
                "mmax_list": [0],
                "grid_resolution": 8,
                "num_sphere_samples": 16,
                "edge_channels": 2,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": True,
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 4,
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
            },
            # 5,514 params
            {
                "regress_forces": True,
                "use_pbc": True,
                "use_pbc_single": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 119,
                "num_layers": 1,
                "sphere_channels": 4,
                "attn_hidden_channels": 4,
                "num_heads": 1,
                "attn_alpha_channels": 4,
                "attn_value_channels": 4,
                "ffn_hidden_channels": 4,
                "norm_type": "layer_norm_sh",
                "lmax_list": [0],
                "mmax_list": [0],
                "grid_resolution": 8,
                "num_sphere_samples": 32,
                "edge_channels": 8,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": True,
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 8,
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
            },
            # 10,140 params
            {
                "regress_forces": True,
                "use_pbc": True,
                "use_pbc_single": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 119,
                "num_layers": 1,
                "sphere_channels": 6,
                "attn_hidden_channels": 6,
                "num_heads": 1,
                "attn_alpha_channels": 5,
                "attn_value_channels": 5,
                "ffn_hidden_channels": 10,
                "norm_type": "layer_norm_sh",
                "lmax_list": [1],
                "mmax_list": [0],
                "grid_resolution": 8,
                "num_sphere_samples": 64,
                "edge_channels": 10,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": True,
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 16,
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
            },
            # 51,324 params
            {
                "regress_forces": True,
                "use_pbc": True,
                "use_pbc_single": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 119,
                "num_layers": 2,
                "sphere_channels": 10,
                "attn_hidden_channels": 10,
                "num_heads": 1,
                "attn_alpha_channels": 10,
                "attn_value_channels": 10,
                "ffn_hidden_channels": 20,
                "norm_type": "layer_norm_sh",
                "lmax_list": [1],
                "mmax_list": [1],
                "grid_resolution": 10,
                "num_sphere_samples": 64,
                "edge_channels": 24,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": True,
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 48,
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
            },
            # 100,540 params
            {
                "regress_forces": True,
                "use_pbc": True,
                "use_pbc_single": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 119,
                "num_layers": 2,
                "sphere_channels": 18,
                "attn_hidden_channels": 18,
                "num_heads": 2,
                "attn_alpha_channels": 18,
                "attn_value_channels": 18,
                "ffn_hidden_channels": 18,
                "norm_type": "layer_norm_sh",
                "lmax_list": [1],
                "mmax_list": [1],
                "grid_resolution": 10,
                "num_sphere_samples": 64,
                "edge_channels": 32,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": True,
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 64,
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
            },
            # 513,250 params
            {
                "regress_forces": True,
                "use_pbc": True,
                "use_pbc_single": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 119,
                "num_layers": 3,
                "sphere_channels": 24,
                "attn_hidden_channels": 24,
                "num_heads": 4,
                "attn_alpha_channels": 24,
                "attn_value_channels": 12,
                "ffn_hidden_channels": 48,
                "norm_type": "layer_norm_sh",
                "lmax_list": [2],
                "mmax_list": [1],
                "grid_resolution": 12,
                "num_sphere_samples": 64,
                "edge_channels": 48,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": True,
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 96,
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
            },
            # 1,032,742 params
            {
                "regress_forces": True,
                "use_pbc": True,
                "use_pbc_single": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 119,
                "num_layers": 3,
                "sphere_channels": 36,
                "attn_hidden_channels": 36,
                "num_heads": 4,
                "attn_alpha_channels": 36,
                "attn_value_channels": 12,
                "ffn_hidden_channels": 72,
                "norm_type": "layer_norm_sh",
                "lmax_list": [2],
                "mmax_list": [1],
                "grid_resolution": 14,
                "num_sphere_samples": 64,
                "edge_channels": 72,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": False,
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 144,
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
            },
            # 5,325,146 params
            {
                "regress_forces": True,
                "use_pbc": True,
                "use_pbc_single": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 119,
                "num_layers": 5,
                "sphere_channels": 48,
                "attn_hidden_channels": 48,
                "num_heads": 4,
                "attn_alpha_channels": 48,
                "attn_value_channels": 32,
                "ffn_hidden_channels": 48,
                "norm_type": "layer_norm_sh",
                "lmax_list": [3],
                "mmax_list": [1],
                "grid_resolution": 32,
                "num_sphere_samples": 96,
                "edge_channels": 120,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": False,
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 240,
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
            },
            # 10,526,602 params
            {
                "regress_forces": True,
                "use_pbc": True,
                "use_pbc_single": True,
                "otf_graph": True,
                "enforce_max_neighbors_strictly": False,
                "max_neighbors": 20,
                "max_radius": 12.0,
                "max_num_elements": 119,
                "num_layers": 6,
                "sphere_channels": 64,
                "attn_hidden_channels": 64,
                "num_heads": 5,
                "attn_alpha_channels": 48,
                "attn_value_channels": 48,
                "ffn_hidden_channels": 48,
                "norm_type": "layer_norm_sh",
                "lmax_list": [3],
                "mmax_list": [1],
                "grid_resolution": 18,
                "num_sphere_samples": 128,
                "edge_channels": 96,
                "use_atom_edge_embedding": True,
                "share_atom_edge_embedding": False,
                "use_m_share_rad": False,
                "distance_function": "gaussian",
                "num_distance_basis": 240,
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
