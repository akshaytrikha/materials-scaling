# External
import torch
from torch_geometric.data import Data, Batch
from pathlib import Path

# Internal
from models.escaip.EScAIP import EScAIPModel
from data import OMat24Dataset, get_pyg_dataloaders
from data_utils import download_dataset


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
            "max_neighbors": 125,  # Minimal value
            "max_radius": 6.0,  # Minimal value
            "max_num_elements": 125,  # TODO: Minimal value
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
    # Load dataset
    split_name = "val"
    dataset_name = "rattled-300-subsampled"

    dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
    if not dataset_path.exists():
        download_dataset(dataset_name, split_name)

    # Initialize PyG dataset and DataLoaders
    dataset = OMat24Dataset(
        dataset_path=dataset_path,
        config_kwargs={},
        augment=False,
        graph=True,
    )
    train_loader, val_loader = get_pyg_dataloaders(
        dataset=dataset,
        data_fraction=0.01,
        batch_size=2,
    )

    # Initialize model
    model = initialize_model(device="cpu")

    for i, batch in enumerate(train_loader):
        # Run a forward pass
        forces, energy, stress = model(batch)

        break
