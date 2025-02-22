# External
from pathlib import Path
import torch
import tarfile
import requests
import os
from typing import Dict
from tqdm.auto import tqdm
from torch_geometric.nn import radius_graph

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


DATASET_INFO = {
    "val": {
        "rattled-300-subsampled": {
            "max_n_atoms": 104,
            "means": {
                "energy": -61.174073438773426,
                "forces": [
                    1.981526641118292e-12,
                    6.042673230329075e-12,
                    6.376201392628392e-12,
                ],
                "stress": [
                    -0.04290581360985851,
                    -0.04287786547423149,
                    -0.04228498222277636,
                    -0.00017707018946872184,
                    -4.857258167117505e-06,
                    -4.462893072596567e-05,
                ],
            },
        },
        "rattled-1000-subsampled": {
            "max_n_atoms": 136,
            "means": {
                "energy": -59.54666023293311,
                "forces": [
                    -1.2772232047488638e-12,
                    -2.064552573689217e-12,
                    -1.784609924015228e-12,
                ],
                "stress": [
                    -0.04808255458885085,
                    -0.04774851593038242,
                    -0.047350432134002356,
                    -3.7177838596686274e-05,
                    -6.237952780940073e-05,
                    -0.00011369407850584661,
                ],
            },
        },
        "rattled-500": {
            "max_n_atoms": 148,
            "means": {
                "energy": -56.59780698167886,
                "forces": [
                    2.373055162870178e-12,
                    9.277368050540146e-13,
                    -1.6015695977643331e-12,
                ],
                "stress": [
                    -0.046081876006950184,
                    -0.045556711948197,
                    -0.044682139798658885,
                    -0.0001422450231372467,
                    -7.844288884680548e-05,
                    -0.00024032503394611605,
                ],
            },
        },
        "rattled-500-subsampled": {
            "max_n_atoms": 106,
            "means": {
                "energy": -60.11217606518476,
                "forces": [
                    3.497619135684947e-12,
                    3.804726969650204e-12,
                    1.330802064993335e-12,
                ],
                "stress": [
                    -0.0440671270123336,
                    -0.04354227003553679,
                    -0.04331436100937788,
                    1.882685793052701e-05,
                    -5.60002114806231e-05,
                    -0.00013769568377300145,
                ],
            },
        },
        "rattled-300": {
            "max_n_atoms": 106,
            "means": {
                "energy": -55.888231137271944,
                "forces": [
                    -1.1273485919347675e-12,
                    1.821101401585356e-12,
                    9.972695594898916e-13,
                ],
                "stress": [
                    -0.040591446813526534,
                    -0.040391684453479626,
                    -0.039826091629161085,
                    -9.693354604665101e-05,
                    7.979895326333171e-05,
                    -0.00031508096994254173,
                ],
            },
        },
        "aimd-from-PBE-3000-npt": {
            "max_n_atoms": 128,
            "means": {
                "energy": -294.39547996666244,
                "forces": [
                    2.77868377131136e-05,
                    -2.7957457766573115e-05,
                    1.4155804813723163e-05,
                ],
                "stress": [
                    -0.02455519312226668,
                    -0.02438445204649411,
                    -0.024090456286676453,
                    4.286747156947961e-05,
                    -3.0367453154403313e-05,
                    -1.2155024362093644e-05,
                ],
            },
        },
        "aimd-from-PBE-3000-nvt": {
            "max_n_atoms": 160,
            "means": {
                "energy": -304.36573226458,
                "forces": [
                    6.058496933454014e-06,
                    2.565589278629701e-06,
                    3.2978085276303616e-06,
                ],
                "stress": [
                    -0.024771605727282747,
                    -0.024539652190160893,
                    -0.02420130110234829,
                    -2.065511902957722e-06,
                    1.0381290401939535e-05,
                    -4.519458526776932e-05,
                ],
            },
        },
        "aimd-from-PBE-1000-npt": {
            "max_n_atoms": 168,
            "means": {
                "energy": -41.77778972552595,
                "forces": [
                    9.522272617586164e-06,
                    -5.518497286229726e-06,
                    5.8326972241464835e-06,
                ],
                "stress": [
                    -0.008698043004380972,
                    -0.008430431847583837,
                    -0.008740891679497471,
                    -5.508375715967929e-05,
                    5.099638890490089e-05,
                    1.566235159924876e-05,
                ],
            },
        },
        "aimd-from-PBE-1000-nvt": {
            "max_n_atoms": 140,
            "means": {
                "energy": -42.41992591550112,
                "forces": [
                    1.191149565533897e-05,
                    -6.632577418889336e-06,
                    7.112136166138489e-06,
                ],
                "stress": [
                    -0.008621120340396074,
                    -0.008321754424218038,
                    -0.008407662593357243,
                    2.3970542986706687e-05,
                    6.967859303579877e-06,
                    2.3026656458743446e-05,
                ],
            },
        },
        "rattled-relax": {
            "max_n_atoms": 120,
            "means": {
                "energy": -42.14154816220651,
                "forces": [
                    -1.8502639128872923e-12,
                    -1.1376622384039623e-12,
                    3.763036616127851e-12,
                ],
                "stress": [
                    -0.0009406117556594279,
                    -0.0011602565402192536,
                    -0.0010417804533152518,
                    3.259700219691146e-05,
                    -2.7666312760414842e-05,
                    6.0944138152337285e-05,
                ],
            },
        },
    },
    "train": {
        "rattled-500-subsampled": {
            "max_n_atoms": 160,
        },
    },
}


def get_dataset_url(dataset_name: str, split_name: str):
    if split_name == "train":
        return f"{TRAIN_BASE_URL}/{dataset_name}.tar.gz"
    elif split_name == "val":
        return f"{VAL_BASE_URL}/{dataset_name}.tar.gz"
    else:
        raise ValueError(f"Invalid split name: {split_name}")


def download_dataset(dataset_name: str, split_name: str):
    """Downloads a compressed dataset from a predefined URL and extracts it to the specified directory.

    Args:
        dataset_name (str): The key corresponding to the dataset in the DATASETS dictionary.
        split_name (str): The split to download ("train" or "val")

    Raises:
        ValueError: If the dataset_name is not valid or the split_name is not valid.
        Exception: If there is an error during the extraction or deletion of the compressed file.
    """
    if split_name not in ["train", "val"]:
        raise ValueError(f"Invalid split name: {split_name}")

    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Get the URL for the dataset
    url = get_dataset_url(dataset_name, split_name)

    # Create the necessary directories
    os.makedirs("./datasets", exist_ok=True)
    os.makedirs(f"./datasets/{split_name}", exist_ok=True)
    dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
    compressed_path = dataset_path.with_suffix(".tar.gz")

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

    # Pad atomic_numbers, positions, and forces to MAX_ATOMS
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

    # Determine the second dimension (k) for the factorized matrices.
    k = factorized_matrices[0].size(1) if factorized_matrices[0].dim() > 1 else 1

    padded_factorized_matrices = torch.stack(
        [
            pad_matrix(tensor, max_n_atoms, k, padding_value=0.0)
            for tensor in factorized_matrices
        ],
        dim=0,
    )  # Shape: [batch_size, MAX_ATOMS, k]

    return {
        "atomic_numbers": padded_atomic_numbers,  # [batch_size, MAX_ATOMS]
        "positions": padded_positions,  # [batch_size, MAX_ATOMS, 3]
        "distance_matrix": padded_distance_matrices,  # [batch_size, MAX_ATOMS, MAX_ATOMS]
        "factorized_matrix": padded_factorized_matrices,  # [batch_size, MAX_ATOMS, k]
        "energy": energies,  # [batch_size]
        "forces": padded_forces,  # [batch_size, MAX_ATOMS, 3]
        "stress": stresses,  # [batch_size, 6]
    }


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
