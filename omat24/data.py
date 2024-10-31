from pathlib import Path
import torch
from torch.utils.data import Dataset
from fairchem.core.datasets import AseDBDataset
import ase


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
