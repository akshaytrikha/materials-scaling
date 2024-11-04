import datasets
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import torch
from torch.utils.data import Dataset
from fairchem.core.datasets import AseDBDataset
import ase
import tarfile
import gdown
import os

DATASETS = {
    "rattled-300-subsampled": f"https://drive.google.com/uc?id=1vZE0J9ccC-SkoBYn3K0H0P3PUlpPy_NC"
}


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
    dataset: datasets.DatasetDict, data_fraction: float, batch_size: int
):
    """Create train and validation dataloaders for a subset of the dataset.

    Args:
        dataset (datasets.DatasetDict): The dataset to create dataloaders from.
        data_fraction (float): Fraction of the dataset to use for training.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        tuple:
            - train_loader (torch.utils.data.DataLoader): Dataloader for the training subset.
            - val_loader (torch.utils.data.DataLoader): Dataloader for the validation subset.
    """
    # Determine the number of training samples based on the data fraction
    dataset_size = int(len(dataset) * data_fraction)
    train_size = int(dataset_size * 0.8)

    train_subset = Subset(dataset, indices=range(train_size))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    val_subset = Subset(dataset, indices=range(train_size, len(dataset)))
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


class OMat24Dataset(Dataset):
    def __init__(self, dataset_path: Path, config_kwargs={}):
        self.dataset = AseDBDataset(config=dict(src=str(dataset_path), **config_kwargs))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve atoms object for the given index
        atoms: ase.atoms.Atoms = self.dataset.get_atoms(idx)

        # Extract atomic numbers, positions
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
            "atomic_numbers": atomic_numbers,  # element types
            "positions": positions,  # 3D atomic coordinates
            "energy": energy,  # target energy
            "forces": forces,  # target forces on each atom
            "stress": stress,  # target stress tensor if available
        }

        return sample
