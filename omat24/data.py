# External
from torch.utils.data import DataLoader, Subset, Dataset
from pathlib import Path
import torch
from fairchem.core.datasets import AseDBDataset
import ase
import random
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PyGDataLoader

# Internal
from matrix import compute_distance_matrix, factorize_matrix, random_rotate_atoms
from data_utils import (
    custom_collate_fn_batch_padded,
    custom_collate_fn_dataset_padded,
    generate_graph,
    DATASETS,
)


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


def get_dataloaders(
    dataset: Dataset,
    train_data_fraction: float,
    batch_size: int,
    seed: int,
    batch_padded: bool = False,
    return_indices: bool = False,
    val_data_fraction: float = 0.1,
    train_workers: int = 0,
    val_workers: int = 0,
    graph: bool = False,
):
    """Creates training and validation DataLoaders from a given dataset.

    This function splits the dataset into training and validation subsets based on the
    specified fractions. It then creates DataLoaders for each subset, using either a custom
    collate function that keeps variable-length tensors as lists or one that pads them to uniform sizes.

    Args:
        dataset (Dataset): The dataset to create DataLoaders from.
        train_data_fraction (float): Fraction of the remaining data (after validation split) to use for training.
        batch_size (int): Number of samples per batch.
        seed (int): Seed for random number generators to ensure reproducibility.
        batch_padded (bool, optional): Whether to pad variable-length tensors.
        return_indices (bool, optional): Whether to return dataset indices for debugging.
        val_data_fraction (float, optional): Fraction of the dataset to use for validation.
        train_workers (int, optional): Number of worker processes for the training DataLoader.
        val_workers (int, optional): Number of worker processes for the validation DataLoader.
        graph (bool, optional): Whether to create PyG DataLoaders for graph datasets.

    Returns:
        tuple:
            - train_loader (DataLoader): DataLoader for the training subset.
            - val_loader (DataLoader): DataLoader for the validation subset.
    """
    train_subset, val_subset, train_indices, val_indices = split_dataset(
        dataset, train_data_fraction, val_data_fraction, seed
    )

    if graph:
        # Create PyG DataLoaders
        train_loader = PyGDataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=train_workers
        )
        val_loader = PyGDataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=val_workers
        )
    else:
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
            num_workers=train_workers,
            persistent_workers=train_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,  # Typically, shuffle=False for validation
            collate_fn=collate_fn,
            num_workers=val_workers,
            persistent_workers=val_workers > 0,
            pin_memory=torch.cuda.is_available(),
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
        split_name = dataset_path.parent.name  # Parent directory's name
        dataset_name = dataset_path.name
        self.max_n_atoms = DATASETS[split_name][dataset_name]["max_n_atoms"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve atoms object for the given index
        atoms: ase.atoms.Atoms = self.dataset.get_atoms(idx)

        # Extract atomic numbers, positions, and symbols
        atomic_numbers = atoms.get_atomic_numbers()  # Shape: (N_atoms,)
        positions = atoms.get_positions()  # Shape: (N_atoms, 3)
        symbols = (
            atoms.symbols.get_chemical_formula()
        )  # Keep as string, no tensor conversion needed

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
            sample = Data(
                pos=positions,
                atomic_numbers=atomic_numbers,
                edge_index=edge_index,
                edge_attr=edge_attr,
                energy=energy,
                forces=forces,
                stress=stress,
            )
            sample.natoms = torch.tensor(len(atoms))
            sample.postiions = positions
            sample.idx = idx
            sample.symbols = symbols
            return sample
        else:
            factorized_matrix = factorize_matrix(
                distance_matrix
            )  # Left matrix: U * sqrt(Sigma) - Shape: [N_atoms, k=5]

            # Package the input and labels into a dictionary
            sample = {
                "idx": idx,
                "symbols": symbols,
                "atomic_numbers": atomic_numbers,  # Element types
                "positions": positions,  # 3D atomic coordinates
                "distance_matrix": distance_matrix,  # [N_atoms, N_atoms]
                "factorized_matrix": factorized_matrix,  # [N_atoms, k=5]
                "energy": energy,  # Target energy
                "forces": forces,  # Target forces on each atom
                "stress": stress,  # Target stress tensor
            }
            return sample
