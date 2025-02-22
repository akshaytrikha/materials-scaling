# External
from typing import List
import torch
import ase
import random
from pathlib import Path
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from fairchem.core.datasets import AseDBDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PyGDataLoader

# Internal
from matrix import compute_distance_matrix, factorize_matrix, random_rotate_atoms
from data_utils import (
    custom_collate_fn_batch_padded,
    custom_collate_fn_dataset_padded,
    generate_graph,
    DATASET_INFO,
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
    dataset_paths: List[str],
    train_data_fraction: float,
    batch_size: int,
    seed: int,
    batch_padded: bool = False,
    val_data_fraction: float = 0.1,
    train_workers: int = 0,
    val_workers: int = 0,
    graph: bool = False,
):
    """Creates training and validation DataLoaders from a list of dataset paths.
    Each dataset is loaded, split into training and validation subsets, and then
    the splits are concatenated. This works for a single dataset path as well as multiple paths.

    Args:
        dataset_paths (List[str]): List of dataset paths.
        train_data_fraction (float): Fraction of each dataset (after validation split) to use for training.
        batch_size (int): Number of samples per batch.
        seed (int): Seed for reproducibility.
        batch_padded (bool, optional): Whether to pad variable-length tensors.
        val_data_fraction (float, optional): Fraction of each dataset to use for validation.
        train_workers (int, optional): Number of worker processes for the training DataLoader.
        val_workers (int, optional): Number of worker processes for the validation DataLoader.
        graph (bool, optional): Whether to create PyG DataLoaders for graph datasets.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_subsets = []
    val_subsets = []

    # Load each dataset from its path and split it individually
    for path in dataset_paths:
        dataset = OMat24Dataset(dataset_paths=[path], graph=graph)
        train_subset, val_subset, _, _ = split_dataset(
            dataset, train_data_fraction, val_data_fraction, seed
        )
        train_subsets.append(train_subset)
        val_subsets.append(val_subset)

    train_dataset = ConcatDataset(train_subsets)
    val_dataset = ConcatDataset(val_subsets)

    if graph:
        # Create PyG DataLoaders
        train_loader = PyGDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=train_workers,
        )
        val_loader = PyGDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_workers,
        )
    else:
        # Set maximum number of atoms for padding
        if len(dataset_paths) > 1:
            max_n_atoms = 180
        else:
            max_n_atoms = dataset.max_n_atoms

        # Select the appropriate collate function
        if batch_padded:
            collate_fn = custom_collate_fn_batch_padded
        else:
            collate_fn = lambda batch: custom_collate_fn_dataset_padded(
                batch, dataset.max_n_atoms
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=train_workers,
            persistent_workers=train_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=val_workers,
            persistent_workers=val_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

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
        dataset_paths: List[Path],
        config_kwargs={},
        augment: bool = False,
        graph: bool = False,
        debug: bool = False,
    ):
        self.dataset = AseDBDataset(config=dict(src=dataset_paths, **config_kwargs))
        self.augment = augment
        self.graph = graph
        self.debug = debug

        if len(dataset_paths) > 1:
            self.max_n_atoms = 180
        else:
            split_name = dataset_paths[0].parent.name  # Parent directory's name
            dataset_name = dataset_paths[0].name
            self.max_n_atoms = DATASET_INFO[split_name][dataset_name]["max_n_atoms"]

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

        if self.graph:
            # Generate graph connectivity (edge_index) and edge attributes (edge_attr)
            edge_index, edge_attr = generate_graph(positions)

            # Create PyG Data object
            sample = Data(
                pos=positions,
                atomic_numbers=atomic_numbers,
                edge_index=edge_index,
                edge_attr=edge_attr,
                energy=energy,
                forces=forces,
                stress=stress.unsqueeze(0),
            )
            sample.natoms = torch.tensor(len(atoms))
            sample.postiions = positions
            sample.idx = idx
            sample.symbols = symbols

        else:
            # Compute the distance matrix from (possibly rotated) positions
            distance_matrix = compute_distance_matrix(
                positions
            )  # Shape: [N_atoms, N_atoms]

            factorized_matrix = factorize_matrix(
                distance_matrix
            )  # Left matrix: U * sqrt(Sigma) - Shape: [N_atoms, k=5]

            # Package into a dictionary
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

        # Add source information for verifying mutli-dataset usage
        if self.debug:
            sample["source"] = atoms.info["calc_id"]

        return sample
