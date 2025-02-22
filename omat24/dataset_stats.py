import json
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from fairchem.core.datasets import AseDBDataset

from data_utils import download_dataset, VALID_DATASETS


class AseAtomsWrapper(Dataset):
    """
    A simple wrapper around AseDBDataset that exposes __getitem__ and __len__ in the format
    that PyTorch's DataLoader expects. This helps the DataLoader use num_workers properly.
    """

    def __init__(self, ase_dataset):
        self.ase_dataset = ase_dataset

    def __len__(self):
        return len(self.ase_dataset)

    def __getitem__(self, idx):
        return self.ase_dataset.get_atoms(idx)


def collate_fn_return_list(batch):
    """
    Collate function that simply returns the entire list of Atoms in the batch
    (rather than discarding all but the first). This allows us to process them in bulk.
    """
    return batch


def main():
    split_name = "val"
    output_data = {}

    NUM_WORKERS = 0
    BATCH_SIZE = 2048

    for dataset_name in VALID_DATASETS:
        print(f"Processing dataset: {dataset_name}")

        # Download if not present
        dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
        if not dataset_path.exists():
            download_dataset(dataset_name, split_name)

        # Wrap it
        ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))
        wrapped_dataset = AseAtomsWrapper(ase_dataset)

        # Create DataLoader
        dataloader = DataLoader(
            wrapped_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn_return_list,
        )

        total_energy = 0.0
        num_configs = 0

        total_forces = np.zeros(3)  # Sum of all force components
        total_atoms = 0  # Count of all atoms across all configs

        total_stress = np.zeros(6)  # Sum of stress components
        num_stress = 0  # Number of configs with stress data

        max_n_atoms = 0

        # Iterate over the batches
        for batch_atoms_list in tqdm(dataloader, desc=dataset_name):
            # Each 'batch_atoms_list' is a *list* of 'Atoms' objects, length up to BATCH_SIZE
            for atoms in batch_atoms_list:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()  # shape (n_atoms, 3)
                stress = atoms.get_stress()  # shape (6,)

                num_atoms = len(atoms.get_atomic_numbers())
                max_n_atoms = max(max_n_atoms, num_atoms)

                total_energy += energy
                num_configs += 1

                total_forces += forces.sum(axis=0)
                total_atoms += num_atoms

                total_stress += stress
                num_stress += 1

        # Compute means
        mean_energy = total_energy / num_configs if num_configs > 0 else 0.0
        mean_forces = (
            (total_forces / total_atoms).tolist()
            if total_atoms > 0
            else [0.0, 0.0, 0.0]
        )
        mean_stress = (
            (total_stress / num_stress).tolist() if num_stress > 0 else [0.0] * 6
        )

        # Save results
        output_data[dataset_name] = {
            "max_n_atoms": max_n_atoms,
            "means": {
                "energy": mean_energy,
                "forces": mean_forces,
                "stress": mean_stress,
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
