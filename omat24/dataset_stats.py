# dataset_stats.py
from pathlib import Path
from tqdm.auto import tqdm
import ase
from fairchem.core.datasets import AseDBDataset
import numpy as np
import json

from data_utils import download_dataset, VALID_DATASETS

# Use the same split for all datasets
split_name = "val"
output_data = {}

for dataset_name in VALID_DATASETS:
    print(f"Processing dataset: {dataset_name}")

    dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
    if not dataset_path.exists():
        download_dataset(dataset_name, split_name)

    ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))

    total_energy = 0.0
    num_configs = 0

    total_forces = np.zeros(3)  # Sum of all force components over all atoms
    total_atoms = 0  # Total number of atoms processed

    total_stress = np.zeros(6)  # Sum of stress components (per configuration)
    num_stress = 0  # Number of configurations (for stress averaging)

    max_n_atoms = 0

    # Process each configuration in the dataset
    for i, _ in tqdm(enumerate(ase_dataset), total=len(ase_dataset), desc=dataset_name):
        atoms: ase.atoms.Atoms = ase_dataset.get_atoms(i)

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
        (total_forces / total_atoms).tolist() if total_atoms > 0 else [0.0, 0.0, 0.0]
    )
    mean_stress = (total_stress / num_stress).tolist() if num_stress > 0 else [0.0] * 6

    # Rename dataset key if needed
    output_key = dataset_name
    if dataset_name == "rattled-1000":
        output_key = "rattled-300-subsampled"

    # Store results in the desired structure
    output_data[output_key] = {
        "max_n_atoms": max_n_atoms,
        "means": {
            "energy": mean_energy,
            "forces": mean_forces,
            "stress": mean_stress,
        },
    }

# Save the aggregated results to a JSON file with keys in double quotes
with open("all_datasets_stats.json", "w") as f:
    json.dump(output_data, f, indent=4)

# Print the JSON output to the console
print(json.dumps(output_data, indent=4))
