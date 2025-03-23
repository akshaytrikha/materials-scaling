# External
from typing import List
import torch
import ase
import random
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset, random_split
from fairchem.core.datasets import AseDBDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import pickle
from bisect import bisect_right


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


# Define top-level wrapper functions for collate_fn
def collate_fn_batch_padded_wrapper(batch):
    """Top-level wrapper to make the function picklable for multiprocessing."""
    # You'll need to have factorize as a global variable or pass it during initialization
    global FACTORIZE  # This should be set at the module level before creating DataLoader
    return custom_collate_fn_batch_padded(batch, FACTORIZE)


def collate_fn_dataset_padded_wrapper(batch):
    """Top-level wrapper to make the function picklable for multiprocessing."""
    # You'll need to have these variables at the module level
    global MAX_N_ATOMS, FACTORIZE  # These should be set at the module level
    return custom_collate_fn_dataset_padded(batch, MAX_N_ATOMS, FACTORIZE)


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
    train_datasets = []
    val_datasets = []

    # Set max number of atoms based on dataset split
    split_name = dataset_paths[0].parent.name
    max_n_atoms = DATASET_INFO[split_name]["all"]["max_n_atoms"]

    # Load and split each pickled dataset
    for path in dataset_paths:
        pickle_path = path / f"{path.name}.pkl"
        if not pickle_path.exists():
            raise FileNotFoundError(
                f"\n\n{pickle_path.name} not found. Please run python3 scripts/pickle_dataset.py {pickle_path.stem} --split {split_name}"
            )

        print(f"Loading dataset from {pickle_path}")
        with open(pickle_path, "rb") as f:
            dataset = pickle.load(f)

        # Split the dataset
        train_subset, val_subset, _, _ = split_dataset(
            dataset, train_data_fraction, val_data_fraction, seed
        )

        train_datasets.append(train_subset)
        val_datasets.append(val_subset)

    # Create concatenated datasets
    if len(train_datasets) > 1:
        train_dataset = ConcatPickledDataset(train_datasets)
        val_dataset = ConcatPickledDataset(val_datasets)
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]

    # Create dataloaders
    train_loader = OMat24DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = OMat24DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
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


class OMat24Dataset(torch.utils.data.Dataset):
    def __init__(self, symbols, positions, atomic_numbers, forces, energy, stress):
        self.symbols = symbols
        self.positions = positions
        self.atomic_numbers = atomic_numbers
        self.forces = forces
        self.energy = energy
        self.stress = stress

    def __len__(self):
        return len(self.atomic_numbers)

    def __getitem__(self, idx):
        atomic_numbers = torch.tensor(self.atomic_numbers[idx], dtype=torch.long)
        positions = torch.tensor(self.positions[idx], dtype=torch.float)
        forces = torch.tensor(self.forces[idx], dtype=torch.float)
        energy = torch.tensor(self.energy[idx], dtype=torch.float)
        stress = torch.tensor(self.stress[idx], dtype=torch.float)

        sample = {
            "idx": idx,
            "symbols": self.symbols[idx],
            "atomic_numbers": atomic_numbers,  # Element types
            "positions": positions,  # Cartesian & fractional coordinates
            # "distance_matrix": distance_matrix,  # [N_atoms, N_atoms]
            # "factorized_matrix": factorized_matrix,  # [N_atoms, k=5]
            "energy": energy,  # Target energy
            "forces": forces,  # Target forces on each atom
            "stress": stress,  # Target stress tensor
        }

        return sample


class ConcatPickledDataset(torch.utils.data.Dataset):
    """Concatenates multiple pickled datasets into a single dataset.
    __getitem__() returns the sample from the appropriate dataset."""

    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = [0] + list(np.cumsum(self.lengths))

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        # Find which dataset the index belongs to
        dataset_idx = bisect_right(self.cumulative_lengths, idx) - 1
        sample_idx = idx - self.cumulative_lengths[dataset_idx]

        # Get the sample from the appropriate dataset
        return self.datasets[dataset_idx][sample_idx]


class OMat24DataLoader(torch.utils.data.DataLoader):
    # Sequences are different
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def collate_fn(batch):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    # Extract items from the dataset
    positions_list = [
        torch.tensor(item["positions"], dtype=torch.float32) for item in batch
    ]
    numbers_list = [
        torch.tensor(item["atomic_numbers"], dtype=torch.int64) for item in batch
    ]
    energy_list = [
        torch.tensor(item["energy"], dtype=torch.float32).unsqueeze(0) for item in batch
    ]
    forces_list = [torch.tensor(item["forces"], dtype=torch.float32) for item in batch]
    stress_list = [torch.tensor(item["stress"], dtype=torch.float32) for item in batch]

    # Create a mask of ones for each sample
    mask_list = [
        torch.ones(len(item["atomic_numbers"]), dtype=torch.int64) for item in batch
    ]

    # Pad variable-length sequences
    positions_t = pad_sequence(positions_list, batch_first=True)
    numbers_t = pad_sequence(numbers_list, batch_first=True)
    forces_t = pad_sequence(forces_list, batch_first=True)
    mask_t = pad_sequence(mask_list, batch_first=True).to(torch.bool)

    # Stack energy and stress
    energy_t = torch.cat(energy_list, dim=0)  # shape: [batch_size]
    stress_t = torch.stack(stress_list, dim=0)  # shape: [batch_size, 6] if stress is 6D

    return positions_t, numbers_t, forces_t, energy_t, stress_t, mask_t


# Old dataset class
# class OMat24Dataset(Dataset):
#     """Dataset class for the OMat24 dataset with data augmentation via random rotations.

#     Args:
#         dataset_path (Path): Path to the extracted dataset directory.
#         architecture (str): Model architecture name (e.g., "FCN", "Transformer", "SchNet", "EquiformerV2").
#         config_kwargs (dict, optional): Additional configuration parameters for AseDBDataset. Defaults to {}.
#         augment (bool, optional): Whether to apply data augmentation (random rotations). Defaults to False.
#         graph (bool, optional): Whether to generate graph data for PyG. Defaults to False.
#     """

#     def __init__(
#         self,
#         dataset_paths: List[Path],
#         architecture: str,
#         config_kwargs={},
#         augment: bool = False,
#         graph: bool = False,
#         debug: bool = False,
#         rank: int = None,
#         world_size: int = None,
#     ):
#         self.dataset_paths = dataset_paths
#         self.config_kwargs = config_kwargs
#         self.architecture = architecture
#         self.augment = augment
#         self.graph = graph
#         self.debug = debug

#         # Shard the dataset if using DDP
#         self.rank = rank
#         self.world_size = world_size

#         # Initialize the dataset
#         self._init_dataset()

#     def _init_dataset(self):
#         """Initialize the ASE dataset with optional sharding"""
#         if self.rank is not None and self.world_size is not None:
#             # Create a sharded dataset config
#             shard_config = dict(
#                 src=self.dataset_paths,
#                 shard_id=self.rank,
#                 num_shards=self.world_size,
#                 **self.config_kwargs
#             )
#             self.dataset = AseDBDataset(config=shard_config)
#         else:
#             self.dataset = AseDBDataset(
#                 config=dict(src=self.dataset_paths, **self.config_kwargs)
#             )

#     def __getstate__(self):
#         """Custom pickling method"""
#         state = self.__dict__.copy()
#         # Don't pickle the ASE dataset
#         state["dataset"] = None
#         return state

#     def __setstate__(self, state):
#         """Custom unpickling method"""
#         self.__dict__.update(state)
#         # Reinitialize the dataset when unpickling
#         self._init_dataset()

#     def get_atoms(self, idx):
#         return self.dataset.get_atoms(idx)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         # Retrieve atoms object for the given index
#         atoms: ase.atoms.Atoms = self.dataset.get_atoms(idx)

#         # Extract atomic numbers, positions, symbols, and cell parameters
#         atomic_numbers = atoms.get_atomic_numbers()  # Shape: (N_atoms,)
#         positions = atoms.get_positions(wrap=True)  # Shape: (N_atoms, 3)
#         scaled_positions = atoms.get_scaled_positions(wrap=True)  # Shape: (N_atoms, 3)

#         symbols = (
#             atoms.symbols.get_chemical_formula()
#         )  # Keep as string, no tensor conversion needed
#         cell = atoms.get_cell()

#         # Convert to tensors
#         atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
#         positions = torch.tensor(positions, dtype=torch.float)
#         scaled_positions = torch.tensor(scaled_positions, dtype=torch.float)
#         combined_positions = torch.cat([positions, scaled_positions], dim=1)
#         # combined_positions = torch.tensor(combined_positions, dtype=torch.float)
#         cell = torch.tensor(np.array(cell), dtype=torch.float)

#         # Extract target properties (e.g., energy, forces, stress)
#         energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float)
#         forces = torch.tensor(
#             atoms.get_forces(), dtype=torch.float
#         )  # Shape: (N_atoms, 3)
#         stress = torch.tensor(
#             atoms.get_stress(), dtype=torch.float
#         )  # Shape: (6,) if stress tensor

#         if self.augment:
#             # Apply random rotation to positions, forces & stress
#             positions, R = random_rotate_atom_positions(positions)
#             # We only rotate the first 3 columns (Cartesian coordinates) of combined positions
#             combined_positions[:, :3] = positions
#             # TODO: Fractional coordinates need special handling for rotation
#             # For now, we leave them as is since they're unit cell relative
#             forces = forces @ R.T
#             stress = rotate_stress(stress, R)

#         if self.graph:
#             pyg_args = {
#                 "pos": positions,
#                 "atomic_numbers": atomic_numbers,
#                 "energy": energy,
#                 "forces": forces,
#                 "stress": stress.unsqueeze(0),
#             }

#             if self.architecture == "SchNet":
#                 # Generate graph connectivity (edge_index) and edge attributes (edge_attr)
#                 edge_index, edge_attr = generate_graph(positions)
#                 pyg_args["edge_index"] = edge_index
#                 pyg_args["edge_attr"] = edge_attr
#             elif self.architecture == "EquiformerV2":
#                 pyg_args["cell"] = cell.unsqueeze(0)  # Shape: [1, 3, 3]
#                 pyg_args["pbc"] = torch.tensor(atoms.get_pbc(), dtype=torch.float)

#             # Create PyG Data object
#             sample = PyGData(**pyg_args)
#             sample.natoms = torch.tensor(len(atoms))
#             sample.idx = idx
#             sample.symbols = symbols

#         else:
#             # Compute the distance matrix from (possibly rotated) positions
#             distance_matrix = compute_distance_matrix(
#                 positions
#             )  # Shape: [N_atoms, N_atoms]

#             factorized_matrix = factorize_matrix(
#                 distance_matrix
#             )  # Left matrix: U * sqrt(Sigma) - Shape: [N_atoms, k=5]

#             # Package into a dictionary
#             sample = {
#                 "idx": idx,
#                 "symbols": symbols,
#                 "atomic_numbers": atomic_numbers,  # Element types
#                 "positions": combined_positions,  # Cartesian & fractional coordinates
#                 "distance_matrix": distance_matrix,  # [N_atoms, N_atoms]
#                 "factorized_matrix": factorized_matrix,  # [N_atoms, k=5]
#                 "energy": energy,  # Target energy
#                 "forces": forces,  # Target forces on each atom
#                 "stress": stress,  # Target stress tensor
#             }

#         # Add source information for verifying mutli-dataset usage
#         if self.debug:
#             sample["source"] = atoms.info["calc_id"]

#         return sample


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
