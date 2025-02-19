# External
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import ase
from fairchem.core.datasets import AseDBDataset
import numpy as np

# Internal
from data import download_dataset

# Setup dataset
split_name = "val"
dataset_name = "rattled-300-subsampled"

dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
if not dataset_path.exists():
    download_dataset(dataset_name, split_name)

ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))

# Initialize DataFrame to log atom info
df = pd.DataFrame(
    columns=["symbols", "num_atoms", "energy", "forces", "stress", "distance_matrix"]
)

# Initialize lists to collect statistics
energies = []
forces_x = []
forces_y = []
forces_z = []
stress_sxx = []
stress_syy = []
stress_szz = []
stress_sxy = []
stress_sxz = []
stress_syz = []

# Process dataset
for i, batch in tqdm(enumerate(ase_dataset), total=len(ase_dataset)):
    atoms: ase.atoms.Atoms = ase_dataset.get_atoms(i)

    symbols = atoms.symbols
    atomic_numbers = atoms.get_atomic_numbers()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()  # Shape: (num_atoms, 3)
    stress = atoms.get_stress()  # Shape: (6,)
    positions = atoms.get_positions()

    # Populate DataFrame
    df.at[i, "symbols"] = str(symbols)
    df.at[i, "num_atoms"] = len(atomic_numbers)
    df.at[i, "energy"] = energy
    df.at[i, "forces"] = forces.tolist()  # Convert to list for CSV compatibility
    df.at[i, "stress"] = stress.tolist()  # Convert to list for CSV compatibility
    df.at[i, "distance_matrix"] = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
    ).tolist()  # Convert to list for CSV compatibility

    # Collect statistics
    energies.append(energy)
    forces_x.extend(forces[:, 0])
    forces_y.extend(forces[:, 1])
    forces_z.extend(forces[:, 2])
    stress_sxx.append(stress[0])
    stress_syy.append(stress[1])
    stress_szz.append(stress[2])
    stress_sxy.append(stress[3])
    stress_sxz.append(stress[4])
    stress_syz.append(stress[5])

# Save atom info to CSV
df.to_csv(f"{dataset_name}_atom_info.csv", index=False)

# Compute statistics
stats = {
    "Energy": {
        "Mean": np.mean(energies),
        "Std Dev": np.std(energies),
    },
    "Forces_X": {
        "Mean": np.mean(forces_x),
        "Std Dev": np.std(forces_x),
    },
    "Forces_Y": {
        "Mean": np.mean(forces_y),
        "Std Dev": np.std(forces_y),
    },
    "Forces_Z": {
        "Mean": np.mean(forces_z),
        "Std Dev": np.std(forces_z),
    },
    "Stress_SXX": {
        "Mean": np.mean(stress_sxx),
        "Std Dev": np.std(stress_sxx),
    },
    "Stress_SYY": {
        "Mean": np.mean(stress_syy),
        "Std Dev": np.std(stress_syy),
    },
    "Stress_SZZ": {
        "Mean": np.mean(stress_szz),
        "Std Dev": np.std(stress_szz),
    },
    "Stress_SXY": {
        "Mean": np.mean(stress_sxy),
        "Std Dev": np.std(stress_sxy),
    },
    "Stress_SXZ": {
        "Mean": np.mean(stress_sxz),
        "Std Dev": np.std(stress_sxz),
    },
    "Stress_SYZ": {
        "Mean": np.mean(stress_syz),
        "Std Dev": np.std(stress_syz),
    },
}

# Convert statistics to DataFrame for better visualization
stats_df = pd.DataFrame(stats)  # Transpose for readability

# Save statistics to CSV
stats_df.to_csv(f"{dataset_name}_statistics.csv")

# Print statistics to console
print("\nDataset Statistics:")
print(stats_df)
