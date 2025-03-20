# dataset_stats.py
import json
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from fairchem.core.datasets import AseDBDataset

from data_utils import download_dataset, VALID_DATASETS


class AseAtomsWrapper(Dataset):
    """A simple wrapper around AseDBDataset that exposes __getitem__ and __len__
    in the format that PyTorch's DataLoader expects."""

    def __init__(self, ase_dataset):
        self.ase_dataset = ase_dataset

    def __len__(self):
        return len(self.ase_dataset)

    def __getitem__(self, idx):
        return self.ase_dataset.get_atoms(idx)


def collate_fn_return_list(batch):
    """
    Collate function that returns the entire list of Atoms objects.
    """
    return batch


def main():
    split_name = "val"
    output_data = {}

    NUM_WORKERS = 0
    BATCH_SIZE = 2048

    for dataset_name in VALID_DATASETS:
        print(f"Processing dataset: {dataset_name}")

        dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
        if not dataset_path.exists():
            download_dataset(dataset_name, split_name)

        ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))
        wrapped_dataset = AseAtomsWrapper(ase_dataset)
        dataloader = DataLoader(
            wrapped_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn_return_list,
        )

        # Initialize accumulators for means and second moments.
        sum_energy = 0.0
        sum_energy2 = 0.0
        n_configs = 0

        sum_forces = np.zeros(3)
        sum_forces2 = np.zeros(3)
        total_atoms = 0

        sum_stress = np.zeros(6)
        sum_stress2 = np.zeros(6)
        n_stress = 0

        max_n_atoms = 0

        for batch_atoms_list in tqdm(dataloader, desc=dataset_name):
            for atoms in batch_atoms_list:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()  # shape (n_atoms, 3)
                stress = atoms.get_stress()  # shape (6,)

                n_atoms = len(atoms.get_atomic_numbers())
                max_n_atoms = max(max_n_atoms, n_atoms)

                # Energy accumulation
                sum_energy += energy
                sum_energy2 += energy**2
                n_configs += 1

                # Forces accumulation (per-atom forces)
                sum_forces += forces.sum(axis=0)
                sum_forces2 += (forces**2).sum(axis=0)
                total_atoms += n_atoms

                # Stress accumulation (per configuration)
                sum_stress += stress
                sum_stress2 += stress**2
                n_stress += 1

        # Compute means and std deviations
        mean_energy = sum_energy / n_configs if n_configs > 0 else 0.0
        energy_std = (
            np.sqrt(sum_energy2 / n_configs - mean_energy**2) if n_configs > 0 else 0.0
        )

        mean_forces = (
            (sum_forces / total_atoms).tolist() if total_atoms > 0 else [0.0, 0.0, 0.0]
        )
        forces_std = (
            np.sqrt(
                sum_forces2 / total_atoms - np.square(sum_forces / total_atoms)
            ).tolist()
            if total_atoms > 0
            else [0.0, 0.0, 0.0]
        )

        mean_stress = (sum_stress / n_stress).tolist() if n_stress > 0 else [0.0] * 6
        stress_std = (
            np.sqrt(sum_stress2 / n_stress - np.square(sum_stress / n_stress)).tolist()
            if n_stress > 0
            else [0.0] * 6
        )

        output_data[dataset_name] = {
            "max_n_atoms": max_n_atoms,
            "means": {
                "energy": mean_energy,
                "forces": mean_forces,
                "stress": mean_stress,
            },
            "std": {
                "energy": energy_std,
                "forces": forces_std,
                "stress": stress_std,
            },
        }

        # Write a JSON file after each dataset
        out_file = Path(f"{split_name}_dataset_stats.json")
        with out_file.open("w") as f:
            json.dump(output_data, f, indent=4)

    # Also print to the console
    print(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    main()
