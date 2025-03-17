# External
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import ase
from fairchem.core.datasets import AseDBDataset
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn

# Internal
from data import download_dataset

# Setup dataset
dataset_name = "rattled-300-subsampled"
dataset_path = Path(f"datasets/{dataset_name}")
energy_df_path = Path("energy_regression.csv")

if not dataset_path.exists():
    # Fetch and uncompress dataset
    download_dataset(dataset_name)

ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))

if not energy_df_path.exists():
    # First time running script
    energy_df = pd.DataFrame(
        columns=["dataset_name", "intercept", "coefficient", "l1_loss"]
    )
else:
    # Subsequent runs
    energy_df = pd.read_csv(energy_df_path)

    # Check if dataset has already been processed
    if dataset_name in energy_df["dataset_name"].values:
        print(f"Dataset {dataset_name} already processed.")
        exit()

# ------------------------ Iterate and collect dataset info ------------------------
num_atoms = []
energies = []
for i in tqdm(range(len((ase_dataset)))):
    atoms: ase.atoms.Atoms = ase_dataset.get_atoms(i)
    num_atoms += [len(atoms.get_atomic_numbers())]
    energies += [atoms.get_potential_energy()]

num_atoms = torch.tensor(num_atoms, dtype=torch.float32)
energies = torch.tensor(energies, dtype=torch.float32)

# ------------------------ Calculate linear model ------------------------
# Create and fit the linear regression model
model = LinearRegression()
model.fit(num_atoms.reshape(-1, 1), energies)

# Output the model parameters
intercept = model.intercept_
coefficient = model.coef_[0]

# Write the model parameters to csv
# Concat the new model parameters to the energy_df
to_concat = pd.DataFrame(
    {
        "dataset_name": dataset_name,
        "intercept": [intercept],
        "coefficient": [coefficient],
        "l1_loss": None,
    }
)
energy_df = pd.concat([energy_df, to_concet], ignore_index=True)
energy_df.to_csv(f"energy_regression.csv", index=False)


# ------------------------ Calculate average loss using linear model ------------------------
# Calculate average loss over dataset
pred_energy = intercept + (coefficient * num_atoms)
loss = nn.L1Loss()(pred_energy, energies)

# Append avg_loss to energy_df at the correct column
energy_df.loc[energy_df["dataset_name"] == dataset_name, "l1_loss"] = loss.item()
energy_df.to_csv("energy_regression.csv", index=False)
