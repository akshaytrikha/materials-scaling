# External
from pathlib import Path
import torch
import tarfile
import gdown
import os
from typing import Dict
from torch_geometric.nn import radius_graph


DATASETS = {
    "val": {
        "rattled-300-subsampled": {
            "url": "https://drive.google.com/uc?export=download&id=1ycJ0uMTaVS42T-A57h9_CkBx-ifwyfvL",
            "max_n_atoms": 104,
            "means": {
                "energy": -61.11903078327683,
                "forces": [
                    2.260326148106596e-12,
                    5.641472828198231e-12,
                    6.426049926738911e-12,
                ],
                "stress": [
                    -0.04294834421283328,
                    -0.04285861961557121,
                    -0.04230156679251802,
                    -0.00018619362332657064,
                    -2.250523665894865e-05,
                    -7.547306737387094e-05,
                ],
            },
        },
        "rattled-1000": {
            "url": "https://drive.google.com/uc?export=download&id=1nZh6j2G9SNtYXnAWsjYGju1QLAqMvIQ-",
            "max_n_atoms": 152,
        },
    },
    "train": {
        "rattled-500-subsampled": {
            "url": "https://drive.google.com/uc?id=1gP_p2uIgNGFpfm-eAR-FoRMh-Sf-wrAd",
            "max_n_atoms": 160,
        },
    },
}


def download_dataset(dataset_name: str, split_name: str):
    """Downloads a compressed dataset from a predefined URL and extracts it to the specified directory.

    Args:
        dataset_name (str): The key corresponding to the dataset in the DATASETS dictionary.

    Raises:
        KeyError: If the dataset_name is not found in the DATASETS dictionary.
        Exception: If there is an error during the extraction or deletion of the compressed file.
    """
    os.makedirs("./datasets", exist_ok=True)
    os.makedirs(f"./datasets/{split_name}", exist_ok=True)

    try:
        url = DATASETS[split_name][dataset_name]["url"]
    except KeyError:
        raise KeyError(f"Dataset '{dataset_name}' not found in DATASETS dictionary.")

    dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
    compressed_path = dataset_path.with_suffix(".tar.gz")
    print(f"Starting download from {url}...")
    gdown.download(url, str(compressed_path), quiet=False)

    # Extract the dataset
    extract_and_clenaup(compressed_path, dataset_path)


def extract_and_clenaup(compressed_path, dataset_path):
    print(f"Extracting {compressed_path}...")
    with tarfile.open(compressed_path, "r:gz") as tar:
        tar.extractall(path=dataset_path.parent)
    print(f"Extraction completed. Files are available at {dataset_path}.")
    try:
        compressed_path.unlink()
        print(f"Deleted the compressed file {compressed_path}.")
    except Exception as e:
        print(f"An error occurred while deleting {compressed_path}: {e}")


def pad_tensor(
    tensor: torch.Tensor, pad_size: int, dim: int = 0, padding_value: float = 0.0
) -> torch.Tensor:
    """
    Pads a tensor to a specified size along a given dimension.

    Args:
        tensor (torch.Tensor): The tensor to pad.
        pad_size (int): The desired size after padding.
        dim (int, optional): The dimension to pad. Defaults to 0.
        padding_value (float, optional): The value to use for padding. Defaults to 0.0.

    Returns:
        torch.Tensor: The padded tensor.
    """
    pad_length = pad_size - tensor.size(dim)
    if pad_length <= 0:
        return tensor
    padding = [padding_value] * pad_length
    if dim == 0:
        padding = torch.full(
            (pad_length, *tensor.shape[1:]),
            padding_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, padding], dim=dim)
    elif dim == 1:
        padding = torch.full(
            (tensor.size(0), pad_length, *tensor.shape[2:]),
            padding_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, padding], dim=dim)
    else:
        raise ValueError("Only dimensions 0 and 1 are supported for padding.")


def pad_matrix(
    matrix: torch.Tensor, pad_size_x: int, pad_size_y: int, padding_value: float = 0.0
) -> torch.Tensor:
    """
    Pad a 2D tensor to the specified dimensions [pad_size_x, pad_size_y].

    Args:
        matrix (torch.Tensor): Input tensor of shape [rows, cols].
        pad_size_x (int): Target number of rows.
        pad_size_y (int): Target number of columns.
        padding_value (float, optional): Value to fill the padded entries. Defaults to 0.0.

    Returns:
        torch.Tensor: Padded tensor of shape [pad_size_x, pad_size_y].
    """
    # Calculate extra rows and columns needed for padding.
    pad_rows = pad_size_x - matrix.size(0)  # additional rows required
    pad_cols = pad_size_y - matrix.size(1)  # additional columns required

    # Ensure the input is not larger than the target dimensions.
    if pad_rows < 0 and pad_cols < 0:
        raise ValueError("Matrix dimensions exceed target pad sizes.")

    # If extra rows are needed, create a filler tensor and concatenate along rows.
    if pad_rows > 0:
        pad_row = torch.full(
            (pad_rows, matrix.size(1)),
            padding_value,
            dtype=matrix.dtype,
            device=matrix.device,
        )
        matrix = torch.cat([matrix, pad_row], dim=0)

    # If extra columns are needed, create a filler tensor and concatenate along columns.
    if pad_cols > 0:
        pad_col = torch.full(
            (matrix.size(0), pad_cols),
            padding_value,
            dtype=matrix.dtype,
            device=matrix.device,
        )
        matrix = torch.cat([matrix, pad_col], dim=1)

    return matrix


def custom_collate_fn_dataset_padded(
    batch: list, max_n_atoms: int
) -> Dict[str, torch.Tensor]:
    """Collate function that pads all samples to a fixed number of atoms (MAX_ATOMS).

    Args:
        batch (List[Dict[str, torch.Tensor]]): A list of samples.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of batched and padded tensors.
    """
    atomic_numbers = [sample["atomic_numbers"] for sample in batch]
    positions = [sample["positions"] for sample in batch]
    distance_matrices = [sample["distance_matrix"] for sample in batch]
    factorized_matrices = [sample["factorized_matrix"] for sample in batch]
    energies = torch.stack([sample["energy"] for sample in batch], dim=0)
    forces = [sample["forces"] for sample in batch]
    stresses = torch.stack([sample["stress"] for sample in batch], dim=0)

    # Pad atomic_numbers, positions, forces to MAX_ATOMS
    padded_atomic_numbers = torch.stack(
        [
            pad_tensor(tensor, max_n_atoms, dim=0, padding_value=0)
            for tensor in atomic_numbers
        ],
        dim=0,
    )  # Shape: [batch_size, MAX_ATOMS]

    padded_positions = torch.stack(
        [
            pad_tensor(tensor, max_n_atoms, dim=0, padding_value=0.0)
            for tensor in positions
        ],
        dim=0,
    )  # Shape: [batch_size, MAX_ATOMS, 3]

    padded_forces = torch.stack(
        [
            pad_tensor(tensor, max_n_atoms, dim=0, padding_value=0.0)
            for tensor in forces
        ],
        dim=0,
    )  # Shape: [batch_size, MAX_ATOMS, 3]

    # Pad distance matrices to [MAX_ATOMS, MAX_ATOMS]
    padded_distance_matrices = torch.stack(
        [
            pad_matrix(tensor, max_n_atoms, max_n_atoms, padding_value=0.0)
            for tensor in distance_matrices
        ],
        dim=0,
    )  # Shape: [batch_size, MAX_ATOMS, MAX_ATOMS]

    padded_factorized_matrices = torch.stack(
        [
            pad_matrix(tensor, max_n_atoms, 0, padding_value=0.0)
            for tensor in factorized_matrices
        ],
        dim=0,
    )  # Shape: [batch_size, MAX_ATOMS, k]

    return_dict = {
        "atomic_numbers": padded_atomic_numbers,  # [batch_size, MAX_ATOMS]
        "positions": padded_positions,  # [batch_size, MAX_ATOMS, 3]
        "distance_matrix": padded_distance_matrices,  # [batch_size, MAX_ATOMS, MAX_ATOMS]
        "factorized_matrix": padded_factorized_matrices,  # [batch_size, MAX_ATOMS, k]
        "energy": energies,  # [batch_size]
        "forces": padded_forces,  # [batch_size, MAX_ATOMS, 3]
        "stress": stresses,  # [batch_size, 6]
    }

    return return_dict


def custom_collate_fn_batch_padded(batch: list) -> Dict[str, torch.Tensor]:
    """Collate function that pads variable-sized tensors to the maximum size within the batch.

    This function pads the `atomic_numbers`, `positions`, `forces`, and `distance_matrix` tensors to ensure
    that all samples in the batch have the same number of atoms (`max_atoms`). Padding is
    applied with a default value of 0 for atomic numbers and 0.0 for positions, forces, and distance matrices.

    Args:
        batch (List[Dict[str, torch.Tensor]]): A list of samples.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of batched and padded tensors.
    """
    atomic_numbers = [sample["atomic_numbers"] for sample in batch]
    positions = [sample["positions"] for sample in batch]
    distance_matrices = [sample["distance_matrix"] for sample in batch]
    factorized_matrices = [sample["factorized_matrix"] for sample in batch]
    energies = torch.stack([sample["energy"] for sample in batch], dim=0)
    forces = [sample["forces"] for sample in batch]
    stresses = torch.stack([sample["stress"] for sample in batch], dim=0)

    # Determine the maximum number of atoms in the batch
    max_atoms = max(sample.size(0) for sample in atomic_numbers)

    # Pad atomic_numbers, positions, forces to max_atoms
    padded_atomic_numbers = torch.stack(
        [
            pad_tensor(tensor, max_atoms, dim=0, padding_value=0)
            for tensor in atomic_numbers
        ],
        dim=0,
    )  # Shape: [batch_size, max_atoms]

    padded_positions = torch.stack(
        [
            pad_tensor(tensor, max_atoms, dim=0, padding_value=0.0)
            for tensor in positions
        ],
        dim=0,
    )  # Shape: [batch_size, max_atoms, 3]

    padded_forces = torch.stack(
        [pad_tensor(tensor, max_atoms, dim=0, padding_value=0.0) for tensor in forces],
        dim=0,
    )  # Shape: [batch_size, max_atoms, 3]

    # Pad distance matrices to [max_atoms, max_atoms]
    padded_distance_matrices = torch.stack(
        [
            pad_matrix(tensor, max_atoms, max_atoms, padding_value=0.0)
            for tensor in distance_matrices
        ],
        dim=0,
    )  # Shape: [batch_size, max_atoms, max_atoms]
    padded_factorized_matrices = torch.stack(
        [
            pad_matrix(tensor, max_atoms, 0, padding_value=0.0)
            for tensor in factorized_matrices
        ],
        dim=0,
    )  # Shape: [batch_size, MAX_ATOMS, k]

    return_dict = {
        "atomic_numbers": padded_atomic_numbers,  # [batch_size, max_atoms]
        "positions": padded_positions,  # [batch_size, max_atoms, 3]
        "distance_matrix": padded_distance_matrices,  # [batch_size, max_atoms, max_atoms]
        "factorized_matrix": padded_factorized_matrices,  # [batch_size, MAX_ATOMS, k]
        "energy": energies,  # [batch_size]
        "forces": padded_forces,  # [batch_size, max_atoms, 3]
        "stress": stresses,  # [batch_size, 6]
    }

    return return_dict


def generate_graph(positions):
    """
    Generate graph connectivity and edge attributes based on positions or distance matrix.
    Customize this method based on how you want to define edges.

    Args:
        positions (torch.Tensor): [N_atoms, 3]
        distance_matrix (torch.Tensor): [N_atoms, N_atoms]

    Returns:
        edge_index (torch.LongTensor): [2, num_edges]
        edge_attr (torch.Tensor): [num_edges, feature_dim]
    """
    # Example: Using radius graph with cutoff 6.0
    cutoff = 6.0
    edge_index = radius_graph(positions, r=cutoff, loop=False)

    # Compute edge attributes based on distance or other features
    # Example: Compute distance for each edge
    row, col = edge_index
    edge_distances = torch.norm(positions[row] - positions[col], dim=1).unsqueeze(
        1
    )  # [num_edges, 1]

    # Example: Include distance as edge attribute
    edge_attr = edge_distances  # You can add more features as needed

    return edge_index, edge_attr
