# External
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import ase
from fairchem.core.datasets import AseDBDataset
from sklearn.linear_model import LinearRegression
import numpy as np

# Internal
from data import download_dataset

# Setup dataset
dataset_name = "rattled-300-subsampled"
dataset_path = Path(f"datasets/{dataset_name}")

if not dataset_path.exists():
    # Fetch and uncompress dataset
    download_dataset(dataset_name)

ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))

if Path("energy_regression.csv").exists():
    energy_df = pd.read_csv("energy_regression.csv")
else:
    energy_df = pd.DataFrame(
        columns=["dataset_name", "intercept", "coefficient", "avg_loss"]
    )


# ------------------------ Calculate linear model ------------------------
if dataset_name not in energy_df["dataset_name"].to_list():
    atom_info_df = pd.read_csv(
        f"{dataset_name}_atom_info.csv", usecols=["num_atoms", "energy"]
    )
    # Features and target variable
    X_true = atom_info_df["num_atoms"].to_numpy()  # Feature: number of atoms
    y_true = atom_info_df["energy"].to_numpy()  # Target: energy

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X_true.reshape(-1, 1), y_true)

    # Output the model parameters
    intercept = model.intercept_
    coefficient = model.coef_[0]

    # Write the model parameters to csv
    energy_df = pd.DataFrame(
        {
            "dataset_name": dataset_name,
            "intercept": [intercept],
            "coefficient": [coefficient],
            "avg_loss": None,
        }
    )
    energy_df.to_csv(f"energy_regression.csv", index=False)
else:
    # Get model parameters
    intercept = energy_df[energy_df["dataset_name"] == dataset_name][
        "intercept"
    ].values[0]
    coefficient = energy_df[energy_df["dataset_name"] == dataset_name][
        "coefficient"
    ].values[0]


# ------------------------ Calculate average loss using linear model ------------------------
avg_loss = energy_df[energy_df["dataset_name"] == dataset_name].iloc[0]["avg_loss"]

if avg_loss is None or np.isnan(avg_loss):
    # Calculate average loss over dataset
    loss = 0

    atom_info_df = pd.read_csv(
        f"{dataset_name}_atom_info.csv", usecols=["num_atoms", "energy"]
    )

    X_true = atom_info_df["num_atoms"].to_numpy()  # Feature: number of atoms
    y_true = atom_info_df["energy"].to_numpy()  # Target: energy

    pred_energy = intercept + (coefficient * X_true)
    loss = np.sum(pred_energy - y_true)  # L1 loss

    # Append avg_loss to energy_df at the correct column
    avg_loss = loss / len(ase_dataset)
    energy_df.loc[energy_df["dataset_name"] == dataset_name, "avg_loss"] = avg_loss
    energy_df.to_csv("energy_regression.csv", index=False)
