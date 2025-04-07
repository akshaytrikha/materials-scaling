# External
from pathlib import Path
import torch
import tarfile
import requests
import os
from tqdm.auto import tqdm
import json
import random
from torch.utils.data import Subset
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data

BASE_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/"
TRAIN_BASE_URL = BASE_URL + "241018/omat/train"
VAL_BASE_URL = BASE_URL + "241220/omat/val"


VALID_DATASETS = [
    "rattled-1000",
    "rattled-1000-subsampled",
    "rattled-500",
    "rattled-500-subsampled",
    "rattled-300",
    "rattled-300-subsampled",
    "aimd-from-PBE-3000-npt",
    "aimd-from-PBE-3000-nvt",
    "aimd-from-PBE-1000-npt",
    "aimd-from-PBE-1000-nvt",
    "rattled-relax",
]

with open("dataset_stats.json", "r") as f:
    DATASET_INFO = json.load(f)


def get_dataset_url(dataset_name: str, split_name: str):
    if split_name == "train":
        return f"{TRAIN_BASE_URL}/{dataset_name}.tar.gz"
    elif split_name == "val":
        return f"{VAL_BASE_URL}/{dataset_name}.tar.gz"
    else:
        raise ValueError(f"Invalid split name: {split_name}")


def download_dataset(
    dataset_name: str, split_name: str, base_path: str = "./datasets"
) -> None:
    """Download and extract a dataset.

    Args:
        dataset_name (str): Name of the dataset to download
        split_name (str): Split type ("train" or "val")
        base_path (str): Base path for dataset storage
    """
    if split_name not in ["train", "val"]:
        raise ValueError(f"Invalid split name: {split_name}")

    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Get the URL for the dataset
    url = get_dataset_url(dataset_name, split_name)

    # Create the necessary directories
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, split_name), exist_ok=True)

    # Construct paths
    dataset_path = Path(base_path) / split_name / dataset_name
    compressed_path = Path(base_path) / f"{dataset_name}.tar.gz"

    # Download the dataset
    print(f"Starting download from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get total file size for progress bar
    total_size = int(response.headers.get("content-length", 0))

    # Download with progress bar
    with open(str(compressed_path), "wb") as f:
        with tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {dataset_name}",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

    # Extract the dataset
    print(f"Extracting {compressed_path}...")
    with tarfile.open(compressed_path, "r:gz") as tar:
        tar.extractall(path=dataset_path.parent)
    print(f"Extraction completed. Files are available at {dataset_path}.")

    # Delete the compressed file
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


def split_dataset(dataset, train_data_fraction, val_data_fraction, seed):
    """Splits a dataset into training and validation subsets."""
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_data_fraction)
    remaining_size = dataset_size - val_size
    train_size = max(1, int(remaining_size * train_data_fraction))

    random.seed(seed)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size : val_size + train_size]

    train_subset = Subset(dataset, indices=train_indices)
    val_subset = Subset(dataset, indices=val_indices)

    return train_subset, val_subset, train_indices, val_indices


class PyGData(Data):
    """Custom PyG Data class with proper batching behavior for periodic systems."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        """Define how tensors should be concatenated during batching"""
        if key == "cell":
            return 0  # Concatenate cells along first dimension
        elif key == "pbc":
            return 0  # Concatenate PBC flags along first dimension
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        """Define how indices should be incremented during batching"""
        if key == "cell" or key == "pbc":
            return 0  # No increment for these attributes
        return super().__inc__(key, value, *args, **kwargs)
