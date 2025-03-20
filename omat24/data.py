# External
from typing import List
import torch
import ase
import random
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from fairchem.core.datasets import AseDBDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data.distributed import DistributedSampler

# Internal
from matrix import (
    compute_distance_matrix,
    factorize_matrix,
    random_rotate_atom_positions,
    rotate_stress,
)
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


class BatchPaddedCollate:
    def __init__(self, factorize):
        self.factorize = factorize

    def __call__(self, batch):
        return custom_collate_fn_batch_padded(batch, self.factorize)


class DatasetPaddedCollate:
    def __init__(self, max_n_atoms, factorize):
        self.max_n_atoms = max_n_atoms
        self.factorize = factorize

    def __call__(self, batch):
        return custom_collate_fn_dataset_padded(batch, self.max_n_atoms, self.factorize)


def get_dataloaders(
    dataset_paths: List[str],
    train_data_fraction: float,
    batch_size: int,
    seed: int,
    architecture: str,
    batch_padded: bool = False,
    val_data_fraction: float = 0.1,
    train_workers: int = 0,
    val_workers: int = 0,
    graph: bool = False,
    factorize: bool = False,
    distributed: bool = False,
    augment: bool = False,
):
    """Creates training and validation DataLoaders from a list of dataset paths.
    Each dataset is loaded, split into training and validation subsets, and then
    the splits are concatenated. This works for a single dataset path as well as multiple paths.

    Args:
        dataset_paths (List[str]): List of dataset paths.
        train_data_fraction (float): Fraction of each dataset (after validation split) to use for training.
        batch_size (int): Number of samples per batch.
        seed (int): Seed for reproducibility.
        architecture (str): Model architecture name (e.g., "FCN", "Transformer", "SchNet", "EquiformerV2").
        batch_padded (bool, optional): Whether to pad variable-length tensors.
        val_data_fraction (float, optional): Fraction of each dataset to use for validation.
        train_workers (int, optional): Number of worker processes for the training DataLoader.
        val_workers (int, optional): Number of worker processes for the validation DataLoader.
        graph (bool, optional): Whether to create PyG DataLoaders for graph datasets.
        factorize (bool, optional): Whether to factorize the distance matrix into a low-rank matrix.
        distributed (bool, optional): Whether to use distributed training.
        augment (bool, optional): Whether to apply data augmentation (random rotations). Defaults to False.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_subsets = []
    val_subsets = []

    # Set max number of atoms per sample based on dataset split
    split_name = dataset_paths[0].parent.name
    max_n_atoms = DATASET_INFO[split_name]["all"]["max_n_atoms"]

    # Load each dataset from its path and split it individually
    for path in dataset_paths:
        dataset = OMat24Dataset(
            dataset_paths=[path],
            graph=graph,
            architecture=architecture,
            augment=augment,
        )
        train_subset, val_subset, _, _ = split_dataset(
            dataset, train_data_fraction, val_data_fraction, seed
        )
        train_subsets.append(train_subset)
        val_subsets.append(val_subset)

    train_dataset = ConcatAseDBDataset(train_subsets)
    val_dataset = ConcatAseDBDataset(val_subsets)

    # Configure samplers for DDP
    if distributed:
        train_sampler = DistributedSampler(train_dataset, seed=seed)
        val_sampler = DistributedSampler(val_dataset, seed=seed, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    if graph:
        # Create PyG DataLoaders
        train_loader = PyGDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle and not distributed,
            sampler=train_sampler,
            num_workers=train_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = PyGDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=val_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        # Create collate function instances
        if batch_padded:
            collate_fn = BatchPaddedCollate(factorize)
        else:
            collate_fn = DatasetPaddedCollate(max_n_atoms, factorize)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle and not distributed,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=train_workers,
            persistent_workers=train_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=val_workers,
            persistent_workers=val_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader


class PyGData(Data):
    """Custom PyG Data class with additional attributes for running EquiformerV2 on OMat24 dataset."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "cell":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # Also ensure 'cell' does not get incremented.
        if key == "cell":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class OMat24Dataset(Dataset):
    """Dataset class for the OMat24 dataset with data augmentation via random rotations.

    Args:
        dataset_path (Path): Path to the extracted dataset directory.
        architecture (str): Model architecture name (e.g., "FCN", "Transformer", "SchNet", "EquiformerV2").
        config_kwargs (dict, optional): Additional configuration parameters for AseDBDataset. Defaults to {}.
        augment (bool, optional): Whether to apply data augmentation (random rotations). Defaults to False.
        graph (bool, optional): Whether to generate graph data for PyG. Defaults to False.
    """

    def __init__(
        self,
        dataset_paths: List[Path],
        architecture: str,
        config_kwargs={},
        augment: bool = False,
        graph: bool = False,
        debug: bool = False,
        rank: int = None,
        world_size: int = None,
    ):
        self.dataset_paths = dataset_paths
        self.config_kwargs = config_kwargs
        self.architecture = architecture
        self.augment = augment
        self.graph = graph
        self.debug = debug

        # Shard the dataset if using DDP
        self.rank = rank
        self.world_size = world_size

        # Initialize the dataset
        self._init_dataset()

    def _init_dataset(self):
        """Initialize the ASE dataset with optional sharding"""
        if self.rank is not None and self.world_size is not None:
            # Create a sharded dataset config
            shard_config = dict(
                src=self.dataset_paths,
                shard_id=self.rank,
                num_shards=self.world_size,
                **self.config_kwargs
            )
            self.dataset = AseDBDataset(config=shard_config)
        else:
            self.dataset = AseDBDataset(
                config=dict(src=self.dataset_paths, **self.config_kwargs)
            )

    def __getstate__(self):
        """Custom pickling method"""
        state = self.__dict__.copy()
        # Don't pickle the ASE dataset
        state["dataset"] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling method"""
        self.__dict__.update(state)
        # Reinitialize the dataset when unpickling
        self._init_dataset()

    def get_atoms(self, idx):
        return self.dataset.get_atoms(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve atoms object for the given index
        atoms: ase.atoms.Atoms = self.dataset.get_atoms(idx)

        # Extract atomic numbers, positions, symbols, and cell parameters
        atomic_numbers = atoms.get_atomic_numbers()  # Shape: (N_atoms,)
        positions = atoms.get_positions()  # Shape: (N_atoms, 3)
        symbols = (
            atoms.symbols.get_chemical_formula()
        )  # Keep as string, no tensor conversion needed
        cell = atoms.get_cell()

        # Convert to tensors
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.float)
        cell = torch.tensor(np.array(cell), dtype=torch.float)

        # Extract target properties (e.g., energy, forces, stress)
        energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float)
        forces = torch.tensor(
            atoms.get_forces(), dtype=torch.float
        )  # Shape: (N_atoms, 3)
        stress = torch.tensor(
            atoms.get_stress(), dtype=torch.float
        )  # Shape: (6,) if stress tensor

        if self.augment:
            # Apply random rotation to positions, forces & stress
            positions, R = random_rotate_atom_positions(positions)
            forces = forces @ R.T
            stress = rotate_stress(stress, R)

        if self.graph:
            pyg_args = {
                "pos": positions,
                "atomic_numbers": atomic_numbers,
                "energy": energy,
                "forces": forces,
                "stress": stress.unsqueeze(0),
            }

            if self.architecture == "SchNet":
                # Generate graph connectivity (edge_index) and edge attributes (edge_attr)
                edge_index, edge_attr = generate_graph(positions)
                pyg_args["edge_index"] = edge_index
                pyg_args["edge_attr"] = edge_attr
            elif self.architecture == "EquiformerV2":
                pyg_args["cell"] = cell.unsqueeze(0)  # Shape: [1, 3, 3]
                pyg_args["pbc"] = torch.tensor(atoms.get_pbc(), dtype=torch.float)

            # Create PyG Data object
            sample = PyGData(**pyg_args)
            sample.natoms = torch.tensor(len(atoms))
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


class ConcatAseDBDataset(ConcatDataset):
    """A thin wrapper around ConcatDataset that provides a get_atoms() method
    which gracefully handles Subset objects as well.

    Used for training naive baseline models."""

    def get_atoms(self, idx):
        # Iterate over each sub-dataset in self.datasets:
        for ds in self.datasets:
            ds_len = len(ds)
            if idx < ds_len:
                if isinstance(ds, Subset):
                    return ds.dataset.get_atoms(ds.indices[idx])
                else:
                    return ds.get_atoms(idx)
            idx -= ds_len
        raise IndexError("Index out of range in ConcatAseDBDataset")
