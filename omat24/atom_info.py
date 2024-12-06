# External
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import ase
from fairchem.core.datasets import AseDBDataset

# Internal
from data import download_dataset

# setup dataset
dataset_name = "rattled-300-subsampled"
dataset_path = Path(f"datasets/{dataset_name}")

if not dataset_path.exists():
    # Fetch and uncompress dataset
    download_dataset(dataset_name)

ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))

# log atom info
df = pd.DataFrame(columns=["symbols", "num_atoms"])

for i, batch in tqdm(enumerate(ase_dataset)):
    atoms: ase.atoms.Atoms = ase_dataset.get_atoms(i)

    symbols = atoms.symbols
    atomic_numbers = atoms.get_atomic_numbers()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    df.at[i, "symbols"] = str(symbols)
    df.at[i, "num_atoms"] = len(atomic_numbers)

df.to_csv(f"{dataset_name}_atom_info.csv", index=False)
