from torch.utils.data import DataLoader, Subset, Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import torch
from fairchem.core.datasets import AseDBDataset
import ase
import tarfile
import gdown
import os
from typing import Dict

MAX_ATOMS = 300

DATASETS = {
    "rattled-300-subsampled": f"https://drive.google.com/uc?id=1vZE0J9ccC-SkoBYn3K0H0P3PUlpPy_NC"
}


def download_dataset(dataset_name: str):
    """Downloads a compressed dataset from a predefined URL and extracts it to the specified directory.

    Args:
        dataset_name (str): The key corresponding to the dataset in the DATASETS dictionary.

    Raises:
        KeyError: If the dataset_name is not found in the DATASETS dictionary.
        Exception: If there is an error during the extraction or deletion of the compressed file.
    """
    os.makedirs("./datasets", exist_ok=True)

    try:
        url = DATASETS[dataset_name]
    except KeyError:
        raise KeyError(f"Dataset '{dataset_name}' not found in DATASETS dictionary.")

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
            (pad_length, *tensor.shape[1:]), padding_value, dtype=tensor.dtype
        )
        return torch.cat([tensor, padding], dim=dim)
    elif dim == 1:
        padding = torch.full(
            (tensor.size(0), pad_length, *tensor.shape[2:]),
            padding_value,
            dtype=tensor.dtype,
        )
        return torch.cat([tensor, padding], dim=dim)
    else:
        raise ValueError("Only dimensions 0 and 1 are supported for padding.")


def custom_collate_fn_dataset_padded(batch: list) -> Dict[str, torch.Tensor]:
    """Collate function that pads all samples to a fixed number of atoms (MAX_ATOMS).

    Args:
        batch (Dict[str]): A dictionary of a batch mapping to lists
    Returns:
        batch (Dict[str]): A dictionary of a batch mapping to lists
    """
    atomic_numbers = [sample["atomic_numbers"] for sample in batch]
    positions = [sample["positions"] for sample in batch]
    energies = torch.stack([sample["energy"] for sample in batch], dim=0)
    forces = [sample["forces"] for sample in batch]
    stresses = torch.stack([sample["stress"] for sample in batch], dim=0)

    # Pad atomic_numbers, positions, and forces to MAX_ATOMS
    padded_atomic_numbers = torch.stack(
        [
            pad_tensor(tensor, MAX_ATOMS, dim=0, padding_value=0)
            for tensor in atomic_numbers
        ],
        dim=0,
    )  # Shape: [batch_size, MAX_ATOMS]

    padded_positions = torch.stack(
        [
            pad_tensor(tensor, MAX_ATOMS, dim=0, padding_value=0.0)
            for tensor in positions
        ],
        dim=0,
    )  # Shape: [batch_size, MAX_ATOMS, 3]

    padded_forces = torch.stack(
        [pad_tensor(tensor, MAX_ATOMS, dim=0, padding_value=0.0) for tensor in forces],
        dim=0,
    )  # Shape: [batch_size, MAX_ATOMS, 3]

    return {
        "atomic_numbers": padded_atomic_numbers,  # [batch_size, MAX_ATOMS]
        "positions": padded_positions,  # [batch_size, MAX_ATOMS, 3]
        "energy": energies,  # [batch_size]
        "forces": padded_forces,  # [batch_size, MAX_ATOMS, 3]
        "stress": stresses,  # [batch_size, 6]
    }


def custom_collate_fn_batch_padded(batch):
    """Collate function that pads variable-sized tensors to the maximum size within the batch.

    This function pads the `atomic_numbers`, `positions`, and `forces` tensors to ensure
    that all samples in the batch have the same number of atoms (`max_atoms`). Padding is
    applied with a default value of 0 for atomic numbers and 0.0 for positions and forces.

    Args:
        batch (Dict[str]): A dictionary of a batch mapping to lists

    Returns:
        batch (Dict[str]): A dictionary of a batch mapping to lists
    """
    # Separate each field
    atomic_numbers = [sample["atomic_numbers"] for sample in batch]
    positions = [sample["positions"] for sample in batch]
    energies = torch.stack([sample["energy"] for sample in batch], dim=0)
    forces = [sample["forces"] for sample in batch]
    stresses = torch.stack([sample["stress"] for sample in batch], dim=0)

    # Determine the maximum number of atoms in the batch
    max_atoms = max(sample.size(0) for sample in atomic_numbers)

    # Pad atomic_numbers and positions
    padded_atomic_numbers = pad_sequence(
        atomic_numbers, batch_first=True, padding_value=0
    )
    padded_positions = pad_sequence(positions, batch_first=True, padding_value=0.0)
    padded_forces = pad_sequence(forces, batch_first=True, padding_value=0.0)

    return {
        "atomic_numbers": padded_atomic_numbers,  # Tensor of shape [batch_size, max_atoms]
        "positions": padded_positions,  # Tensor of shape [batch_size, max_atoms, 3]
        "energy": energies,  # Tensor of shape [batch_size]
        "forces": padded_forces,  # Tensor of shape [batch_size, max_atoms, 3]
        "stress": stresses,  # Tensor of shape [batch_size, 6]
    }


def get_dataloaders(
    dataset: Dataset, data_fraction: float, batch_size: int, batch_padded: bool = True
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


class OMat24Dataset(Dataset):
    """
    Args:
        dataset_path (Path): Path to the extracted dataset directory.
        config_kwargs (dict, optional): Additional configuration parameters for AseDBDataset. Defaults to {}.
    """

    def __init__(self, dataset_path: Path, config_kwargs={}):
        self.dataset = AseDBDataset(config=dict(src=str(dataset_path), **config_kwargs))

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

        # Package the input and labels into a dictionary for model processing
        sample = {
            "atomic_numbers": atomic_numbers,  # Element types
            "positions": positions,  # 3D atomic coordinates
            "energy": energy,  # Target energy
            "forces": forces,  # Target forces on each atom
            "stress": stress,
        }

        return sample
