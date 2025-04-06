# External
from typing import List
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from fairchem.core.datasets import AseDBDataset
from torch_geometric.data import Batch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Internal
from matrix import random_rotate_atom_positions, rotate_stress
from data_utils import split_dataset, PyGData


def get_in_memory_dataloaders(
    dataset_paths: List[str],
    train_data_fraction: float,
    batch_size: int,
    seed: int,
    architecture: str,
    val_data_fraction: float = 0.1,
    train_workers: int = 0,
    val_workers: int = 0,
    graph: bool = False,
    distributed: bool = False,
    augment: bool = False,
):
    """Creates in-memory training and validation DataLoaders.

    Args:
        dataset_paths (List[str]): List of dataset paths.
        train_data_fraction (float): Fraction of each dataset (after validation split) to use for training.
        batch_size (int): Number of samples per batch.
        seed (int): Seed for reproducibility.
        architecture (str): Model architecture name.
        val_data_fraction (float, optional): Fraction of each dataset to use for validation.
        train_workers (int, optional): Number of worker processes for the training DataLoader.
        val_workers (int, optional): Number of worker processes for the validation DataLoader.
        graph (bool, optional): Whether the model is graph-based.
        distributed (bool, optional): Whether to use distributed training.
        augment (bool, optional): Whether to apply data augmentation (random rotations).

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_subsets = []
    val_subsets = []

    # Process each dataset path
    for path in dataset_paths:
        print(f"Loading dataset from {path} into memory...")
        dataset = InMemoryOMat24Dataset(
            dataset_paths=[path], architecture=architecture, augment=augment
        )

        # Split the dataset
        train_subset, val_subset, _, _ = split_dataset(
            dataset, train_data_fraction, val_data_fraction, seed
        )

        train_subsets.append(train_subset)
        val_subsets.append(val_subset)

    # Concatenate the datasets
    train_dataset = ConcatDataset(train_subsets)
    val_dataset = ConcatDataset(val_subsets)

    # Configure samplers for DDP
    if distributed:
        train_sampler = DistributedSampler(train_dataset, seed=seed)
        val_sampler = DistributedSampler(val_dataset, seed=seed, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle and not distributed,
        sampler=train_sampler,
        num_workers=train_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=Batch.from_data_list,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=val_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=Batch.from_data_list,
    )

    return train_loader, val_loader


class InMemoryOMat24Dataset(Dataset):
    """In-memory dataset class for the OMat24 dataset.

    Pre-loads all data into memory for faster access during training.

    Args:
        dataset_paths (Path): Path to the extracted dataset directory.
        architecture (str): Model architecture name.
        augment (bool, optional): Whether to apply data augmentation (random rotations).
    """

    def __init__(
        self,
        dataset_paths: List[Path],
        architecture: str,
        augment: bool = False,
    ):
        self.dataset_path = dataset_paths
        self.architecture = architecture
        self.augment = augment

        # Load dataset
        ase_dataset = AseDBDataset(config=dict(src=dataset_paths))

        # Pre-load all data into memory
        self.data_list = []

        print(f"Pre-loading {len(ase_dataset)} structures into memory...")
        for idx in tqdm(range(len(ase_dataset))):
            # Get the atoms object
            atoms = ase_dataset.get_atoms(idx)

            # Extract data based on architecture requirements
            data_dict = self._extract_data(atoms, idx)

            # Store the pre-extracted data
            self.data_list.append(data_dict)

    def _extract_data(self, atoms, idx):
        """Extract necessary data from atoms object based on architecture."""
        # Basic data that all architectures need
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()
        symbols = atoms.symbols.get_chemical_formula()

        # Convert to tensors
        data_dict = {
            "pos": torch.tensor(positions, dtype=torch.float),
            "atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.long),
            "energy": torch.tensor(energy, dtype=torch.float),
            "forces": torch.tensor(forces, dtype=torch.float),
            "stress": torch.tensor(stress, dtype=torch.float).unsqueeze(0),
            "natoms": torch.tensor(len(atoms)),
            "idx": idx,
            "symbols": symbols,
        }

        # Architecture-specific data
        if self.architecture == "SchNet":
            # For SchNet, we'll generate graph connectivity during __getitem__
            # We don't pre-compute it since it's fast and depends on the cutoff parameter
            pass
        elif self.architecture == "EquiformerV2":
            data_dict["cell"] = torch.tensor(
                atoms.get_cell(), dtype=torch.float
            ).unsqueeze(0)
            data_dict["pbc"] = torch.tensor(atoms.get_pbc(), dtype=torch.float)
        elif self.architecture == "ADiT":
            data_dict["cell"] = torch.tensor(
                np.array(atoms.get_cell()), dtype=torch.float
            ).unsqueeze(0)
            data_dict["pbc"] = torch.tensor(atoms.get_pbc(), dtype=torch.bool)
            data_dict["frac_coords"] = torch.tensor(
                atoms.get_scaled_positions(), dtype=torch.float
            )
            data_dict["token_idx"] = torch.arange(len(atoms))

        return data_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Get pre-extracted data
        data_dict = self.data_list[idx]

        # Create a copy to avoid modifying the original data
        data_dict = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in data_dict.items()
        }

        # Generate edge_index and edge_attr for SchNet during runtime
        if self.architecture == "SchNet":
            from data_utils import generate_graph

            edge_index, edge_attr = generate_graph(data_dict["pos"])
            data_dict["edge_index"] = edge_index
            data_dict["edge_attr"] = edge_attr

        # Apply data augmentation if specified
        if self.augment:
            # Apply random rotation to positions, forces & stress
            positions, R = random_rotate_atom_positions(data_dict["pos"])
            data_dict["pos"] = positions
            data_dict["forces"] = data_dict["forces"] @ R.T
            data_dict["stress"] = rotate_stress(
                data_dict["stress"].squeeze(0), R
            ).unsqueeze(0)

        # Create PyG Data object
        sample = PyGData(**data_dict)
        return sample
