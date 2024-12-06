# External
from torch.utils.data import DataLoader, Subset, Dataset
from pathlib import Path
import torch
from fairchem.core.datasets import AseDBDataset
import ase
from typing import Tuple
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PyGDataLoader

# Internal
from matrix import compute_distance_matrix, random_rotate_atoms
from data_utils import (
    custom_collate_fn_batch_padded,
    custom_collate_fn_dataset_padded,
    generate_graph,
)


def get_dataloaders(
    dataset: Dataset,
    data_fraction: float,
    batch_size: int,
    batch_padded: bool = True,
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
        batch_padded (bool, optional): Whether to pad variable-length tensors. Defaults to True.

    Returns:
        tuple:
            - train_loader (DataLoader): DataLoader for the training subset.
            - val_loader (DataLoader): DataLoader for the validation subset.
    """
    # Determine the number of samples based on the data fraction
    dataset_size = int(len(dataset) * data_fraction)
    train_size = int(dataset_size * 0.8)

    train_subset = Subset(dataset, indices=range(train_size))
    val_subset = Subset(dataset, indices=range(train_size, dataset_size))

    # Select the appropriate collate function
    if batch_padded:
        collate_fn = custom_collate_fn_batch_padded
    else:
        collate_fn = custom_collate_fn_dataset_padded
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,  # Typically, shuffle=False for validation
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def get_pyg_dataloaders(
    dataset_path: Path,
    config_kwargs={},
    data_fraction: float = 0.9,
    batch_size: int = 32,
    augment: bool = False,
) -> Tuple[PyGDataLoader, PyGDataLoader]:
    """
    Creates training and validation PyG DataLoaders from a given dataset.

    Args:
        dataset_path (Path): Path to the dataset directory.
        config_kwargs (dict, optional): Additional configuration for the dataset. Defaults to {}.
        data_fraction (float, optional): Fraction of the dataset to use (e.g., 0.9 for 90%). Defaults to 0.9.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        augment (bool, optional): Whether to apply data augmentation. Defaults to False.

    Returns:
        Tuple[PyGDataLoader, PyGDataLoader]: Training and validation DataLoaders.
    """
    # Initialize the PyG dataset
    dataset = OMat24Dataset(
        dataset_path=dataset_path,
        config_kwargs=config_kwargs,
        augment=augment,
        graph=True,
    )

    # Determine the number of samples based on the data fraction
    dataset_size = int(len(dataset) * data_fraction)
    train_size = int(dataset_size * 0.8)

    train_subset = Subset(dataset, indices=range(train_size))
    val_subset = Subset(dataset, indices=range(train_size, dataset_size))

    # Create PyG DataLoaders
    train_loader = PyGDataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class OMat24Dataset(Dataset):
    """Dataset class for the OMat24 dataset with data augmentation via random rotations.

    Args:
        dataset_path (Path): Path to the extracted dataset directory.
        config_kwargs (dict, optional): Additional configuration parameters for AseDBDataset. Defaults to {}.
        augment (bool, optional): Whether to apply data augmentation (random rotations). Defaults to True.
    """

    def __init__(
        self,
        dataset_path: Path,
        config_kwargs={},
        augment: bool = False,
        graph: bool = False,
    ):
        self.dataset = AseDBDataset(config=dict(src=str(dataset_path), **config_kwargs))
        self.augment = augment
        self.graph = graph

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

        if self.graph:
            # Generate graph connectivity (edge_index) and edge attributes (edge_attr)
            edge_index, edge_attr = generate_graph(positions, distance_matrix)

            # Create PyG Data object
            data = Data(
                pos=positions,
                atomic_numbers=atomic_numbers,
                edge_index=edge_index,
                edge_attr=edge_attr,
                energy=energy,
                forces=forces,
                stress=stress,
            )
            data.natoms = torch.tensor([len(atoms)])

            return data
        else:
            # Package the input and labels into a dictionary
            sample = {
                "atomic_numbers": atomic_numbers,  # Element types
                "positions": positions,  # 3D atomic coordinates
                "distance_matrix": distance_matrix,  # [N_atoms, N_atoms]
                "energy": energy,  # Target energy
                "forces": forces,  # Target forces on each atom
                "stress": stress,  # Target stress tensor
            }

            return sample
