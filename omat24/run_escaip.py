# External
import torch
from torch_geometric.data import Data, Batch

# Internal
from models.escaip.EScAIP import EScAIPModel


def create_example_batch():
    """
    Create a dummy batch compatible with the model's expected input format.

    Returns:
        Batch: A PyTorch Geometric Batch object with dummy data.
    """
    num_nodes = 10
    num_edges = 20
    num_graphs = 2

    # Here, we randomly assign atomic numbers between 1 and 10
    atomic_numbers_list = [
        torch.randint(0, 10, (num_nodes,)) for _ in range(num_graphs)
    ]

    data_list = [
        Data(
            x=torch.randn(num_nodes, 3),  # Example node features
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),  # Edge list
            edge_attr=torch.randn(num_edges, 3),  # Edge features
            pos=torch.randn(num_nodes, 3),  # Positions (if needed)
            atomic_numbers=atomic_numbers_list[i],  # Atomic numbers
            natoms=torch.tensor([num_nodes]),  # Number of atoms (nodes)
        )
        for i in range(num_graphs)
    ]

    return Batch.from_data_list(data_list)


# Model initialization
def initialize_model(device: str):
    """
    Initialize the EScAIPBackbone model with default or custom configurations.

    Returns:
        EScAIPBackbone: The instantiated model.
    """
    config = {
        "global_cfg": {
            "regress_forces": True,
            "direct_force": True,
            "use_fp16_backbone": False,
            "hidden_size": 128,  # Minimal value, must be divisible by num_heads
            "batch_size": 2,  # Minimal batch size
            "activation": "gelu",
        },
        "molecular_graph_cfg": {
            "use_pbc": False,
            "use_pbc_single": False,
            "otf_graph": False,
            "max_neighbors": 10,  # Minimal value
            "max_radius": 6.0,  # Minimal value
            "max_num_elements": 10,  # Minimal value
            "max_num_nodes_per_batch": 20,  # Minimal practical value
            "enforce_max_neighbors_strictly": False,
            "distance_function": "gaussian",
        },
        "gnn_cfg": {
            "num_layers": 2,  # Minimal number of layers
            "atom_embedding_size": 32,  # Minimal embedding size
            "node_direction_embedding_size": 16,  # Minimal size
            "node_direction_expansion_size": 4,  # Minimal size
            "edge_distance_expansion_size": 32,  # Minimal size
            "edge_distance_embedding_size": 64,  # Minimal size
            "atten_name": "math",
            "atten_num_heads": 4,  # Reduced number of attention heads
            "readout_hidden_layer_multiplier": 1,  # Simplified multiplier
            "output_hidden_layer_multiplier": 1,
            "ffn_hidden_layer_multiplier": 1,
        },
        "reg_cfg": {
            "mlp_dropout": 0.1,
            "atten_dropout": 0.1,
            "stochastic_depth_prob": 0.0,  # No stochastic depth for simplicity
            "normalization": "layernorm",  # Minimal normalization
        },
    }

    # Initialize the Combined EScAIP Model
    model = EScAIPModel(config=config).to(device)

    return model


# Main script
if __name__ == "__main__":
    # Create a dummy batch of data
    batch = create_example_batch()

    # Initialize the model
    model = initialize_model(device="cpu")

    # Run a forward pass
    output = model(batch)
    energy = output["energy"]
    forces = output["forces"]
    stress_isotropic = output["stress_isotropic"]
    stress_anisotropic = output["stress_anisotropic"]
