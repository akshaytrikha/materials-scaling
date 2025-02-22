# External
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from fairchem.core.datasets import AseDBDataset
import numpy as np
import json

# Internal
from data_utils import download_dataset, VALID_DATASETS


# Wrap AseDBDataset so DataLoader can use num_workers properly
class AseAtomsWrapper(Dataset):
    def __init__(self, ase_dataset):
        self.ase_dataset = ase_dataset

    def __len__(self):
        return len(self.ase_dataset)

    def __getitem__(self, idx):
        # Retrieve the Atoms object for the given index
        atoms = self.ase_dataset.get_atoms(idx)
        return atoms


def main():
    # Use the same split for all datasets
    split_name = "val"
    output_data = {}

    # Adjust as needed (e.g., num_workers=4 or more)
    NUM_WORKERS = 0
    BATCH_SIZE = 2048

    for dataset_name in VALID_DATASETS:
        print(f"Processing dataset: {dataset_name}")

        dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
        if not dataset_path.exists():
            download_dataset(dataset_name, split_name)

        # Load your ASE dataset
        ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))

        # Wrap it so we can use DataLoader
        wrapped_dataset = AseAtomsWrapper(ase_dataset)
        dataloader = DataLoader(
            wrapped_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=lambda x: x[
                0
            ],  # since batch_size=1, just return the single element
        )

        total_energy = 0.0
        num_configs = 0

        total_forces = np.zeros(3)  # Sum of all force components over all atoms
        total_atoms = 0  # Total number of atoms processed

        total_stress = np.zeros(6)  # Sum of stress components
        num_stress = 0  # Number of configurations (for stress averaging)

        max_n_atoms = 0

        # Iterate in parallel over the dataset
        for i, atoms in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=dataset_name
        ):
            # 'atoms' is an ase.atoms.Atoms object
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()  # shape: (n_atoms, 3)
            stress = atoms.get_stress()  # shape: (6,)

            num_atoms = len(atoms.get_atomic_numbers())
            max_n_atoms = max(max_n_atoms, num_atoms)

            total_energy += energy
            num_configs += 1

            total_forces += forces.sum(axis=0)
            total_atoms += num_atoms

            total_stress += stress
            num_stress += 1

        # Calculate means
        mean_energy = total_energy / num_configs if num_configs > 0 else 0.0
        mean_forces = (
            (total_forces / total_atoms).tolist()
            if total_atoms > 0
            else [0.0, 0.0, 0.0]
        )
        mean_stress = (
            (total_stress / num_stress).tolist() if num_stress > 0 else [0.0] * 6
        )

        # Store results in the desired structure
        output_data[dataset_name] = {
            "max_n_atoms": max_n_atoms,
            "means": {
                "energy": mean_energy,
                "forces": mean_forces,
                "stress": mean_stress,
            },
        }

        # Save the aggregated results to a JSON file immediately after processing each dataset
        with open(f"{split_name}_dataset_stats.json", "w") as f:
            json.dump(output_data, f, indent=4)

    # Print the JSON output to the console
    print(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    main()
