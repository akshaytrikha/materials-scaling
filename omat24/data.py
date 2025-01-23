# External
from torch.utils.data import DataLoader, Subset, Dataset
from pathlib import Path
import torch
from fairchem.core.datasets import AseDBDataset
import ase
import tarfile
import gdown
import os
import random

# Internal
from matrix import compute_distance_matrix, random_rotate_atoms
from data_utils import (
    custom_collate_fn_batch_padded,
    custom_collate_fn_dataset_padded,
    DATASETS,
)


def download_dataset(dataset_name: str):
    """Downloads a .tar.gz file from the specified URL and extracts it to the given directory."""
    os.makedirs("./datasets", exist_ok=True)
    url = DATASETS[dataset_name]
    dataset_path = Path(f"datasets/{dataset_name}")
    compressed_path = dataset_path.with_suffix(".tar.gz")
    print(f"Starting download from {url}...")
    gdown.download(url, str(compressed_path), quiet=False)
    # Extract the dataset
    print(f"Extracting {compressed_path}...")
    with tarfile.open(compressed_path, "r:gz") as tar:
        tar.extractall(path=dataset_path.parent)
    print(f"Extraction completed. Files are available at {dataset_path}.")
    # Clean up
    try:
        compressed_path.unlink()
        print(f"Deleted the compressed file {compressed_path}.")
    except Exception as e:
        print(f"An error occurred while deleting {compressed_path}: {e}")


def get_dataloaders(
    dataset: Dataset,
    train_data_fraction: float,
    batch_size: int,
    seed: int,
    batch_padded: bool = True,
    return_indices: bool = False,
):
    """Creates training and validation DataLoaders from a given dataset.

    This function splits the dataset into training and validation subsets based on the
    specified `data_fraction`. It then creates DataLoaders for each subset, using either
    a custom collate function that keeps variable-length tensors as lists or one that
    pads them to uniform sizes.

    Args:
        dataset (Dataset): The dataset to create DataLoaders from.
        data_fraction (float): Fraction of the dataset to use (e.g., 0.9 for 90%).
        batch_size (int): Number of samples per batch.
        seed (int): Seed for random number generators to ensure reproducibility
        batch_padded (bool, optional): Whether to pad variable-length tensors. Defaults to True.

    Returns:
        tuple:
            - train_loader (DataLoader): DataLoader for the training subset.
            - val_loader (DataLoader): DataLoader for the validation subset.
    """
    dataset_size = len(dataset)
    val_size = int(dataset_size * 0.1)
    remaining_size = dataset_size - val_size
    train_size = int(remaining_size * train_data_fraction)

    random.seed(seed)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:val_size + train_size]

    train_subset = Subset(dataset, indices=train_indices)
    val_subset = Subset(dataset, indices=val_indices)

    # Select the appropriate collate function
    if batch_padded:
        collate_fn = custom_collate_fn_batch_padded
    else:
        collate_fn = lambda batch: custom_collate_fn_dataset_padded(
            batch, dataset.max_n_atoms
        )

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,  # Typically, shuffle=False for validation
        collate_fn=collate_fn,
    )

    if return_indices:
        # For debugging
        return train_loader, val_loader, train_indices, val_indices
    else:
        return train_loader, val_loader


class OMat24Dataset(Dataset):
    """Dataset class for the OMat24 dataset with data augmentation via random rotations.

    Args:
        dataset_path (Path): Path to the extracted dataset directory.
        config_kwargs (dict, optional): Additional configuration parameters for AseDBDataset. Defaults to {}.
        augment (bool, optional): Whether to apply data augmentation (random rotations). Defaults to True.
    """

    def __init__(self, dataset_path: Path, config_kwargs={}, augment: bool = False):
        self.dataset = AseDBDataset(config=dict(src=str(dataset_path), **config_kwargs))
        self.augment = augment
        split_name = dataset_path.parent.name  # Parent directory's name
        dataset_name = dataset_path.name
        self.max_n_atoms = DATASETS[split_name][dataset_name]["max_n_atoms"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve atoms object for the given index
        atoms: ase.atoms.Atoms = self.dataset.get_atoms(idx)

        # Extract atomic numbers and positions
        atomic_numbers = atoms.get_atomic_numbers()  # Shape: (N_atoms,)
        positions = atoms.get_positions()  # Shape: (N_atoms, 3)

        # Convert to tensors
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.float)

        # Extract target properties (e.g., energy, forces, stress)
        energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float)
        forces = torch.tensor(
            atoms.get_forces(), dtype=torch.float
        )  # Shape: (N_atoms, 3)
        stress = torch.tensor(
            atoms.get_stress(), dtype=torch.float
        )  # Shape: (6,) if stress tensor

        if self.augment:
            # Apply random rotation to positions and forces
            positions, R = random_rotate_atoms(positions)
            forces = forces @ R.T

        # Compute the distance matrix from (possibly rotated) positions
        distance_matrix = compute_distance_matrix(
            positions
        )  # Shape: [N_atoms, N_atoms]

        # Package the input and labels into a dictionary for model processing
        sample = {
            "atomic_numbers": atomic_numbers,  # Element types
            "positions": positions,  # 3D atomic coordinates
            "distance_matrix": distance_matrix,  # [N_atoms, N_atoms]
            "energy": energy,  # Target energy
            "forces": forces,  # Target forces on each atom
            "stress": stress,  # Target stress tensor
        }

        return sample
