"""
Script to generate a synthetic ASE database for two-atom systems with opposite forces.
The generated database will be saved in "datasets/vc_dim/ase.db" and can be loaded with FAIRChem's AseDBDataset.

Each sample is a hydrogen dimer with:
  - Positions: atoms at [-d/2, 0, 0] and [d/2, 0, 0] (with d chosen randomly)
  - Forces: random forces sampled from dataset statistics (and applied oppositely)
  - Energy: random energy sampled from dataset statistics
  - Stress: a 6-component tensor sampled from dataset statistics
"""

import os
import random
import numpy as np
import pandas as pd
from ase import Atoms
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator

SEED = 1024
np.random.seed(SEED)


def generate_random_stress():
    """
    Generate a random stress tensor using dataset statistics.
    It extracts the mean and standard deviation for each stress component from stats_df,
    and samples a 6-component stress vector from a normal distribution.
    """
    # Define the stress labels in the order: SXX, SYY, SZZ, SXY, SXZ, SYZ.
    stress_labels = [
        "Stress_SXX",
        "Stress_SYY",
        "Stress_SZZ",
        "Stress_SXY",
        "Stress_SXZ",
        "Stress_SYZ",
    ]
    means = stats_df.loc[stats_df["Unnamed: 0"].isin(stress_labels), "Mean"].to_numpy()
    stds = stats_df.loc[
        stats_df["Unnamed: 0"].isin(stress_labels), "Std Dev"
    ].to_numpy()

    stress = np.random.normal(loc=means, scale=stds)
    return stress


def generate_random_forces():
    """
    Generate random forces for a two-atom system using dataset statistics.
    The function extracts means and standard deviations for Forces_X, Forces_Y, and Forces_Z
    from the global stats_df, samples a force vector, and returns forces for two atoms with
    opposite directions.
    """
    labels = ["Forces_X", "Forces_Y", "Forces_Z"]
    means = stats_df.loc[stats_df["Unnamed: 0"].isin(labels), "Mean"].to_numpy()
    stds = stats_df.loc[stats_df["Unnamed: 0"].isin(labels), "Std Dev"].to_numpy()

    # Sample a 3-component force vector.
    force = np.random.normal(loc=means, scale=stds)
    # Apply opposite forces to the two atoms.
    forces = np.array([force, -force])
    return forces


def generate_random_energy():
    """
    Generate a random energy value using dataset statistics.
    The function extracts the mean and standard deviation for Energy from the global stats_df,
    and samples a scalar energy value from a normal distribution.
    """
    # Extract energy statistics (assuming the label is 'Energy')
    energy_row = stats_df[stats_df["Unnamed: 0"] == "Energy"]
    mean_energy = energy_row["Mean"].values[0]
    std_energy = energy_row["Std Dev"].values[0]

    energy = np.random.normal(loc=mean_energy, scale=std_energy)
    return energy


def generate_vc_dim_dataset(db_path, n_samples):
    """Generate a synthetic dataset with two-atom structures.

    Parameters:
        db_path (str): Path to the ASE database file to be created.
        n_samples (int): Number of synthetic samples to generate.
    """
    # Open (or create) the ASE database
    db = connect(db_path)

    for i in range(n_samples):
        # Random interatomic distance (in Å)
        d = random.uniform(0.8, 1.2)
        pos1 = [-d / 2, 0.0, 0.0]
        pos2 = [d / 2, 0.0, 0.0]
        atoms = Atoms("H2", positions=[pos1, pos2])

        # Define forces: random forces from dataset statistics.
        forces = generate_random_forces()

        # Define stress: random stress from dataset statistics.
        stress = generate_random_stress()

        # Define energy: random energy from dataset statistics.
        energy = generate_random_energy()

        # Attach arrays so that get_forces() and get_stress() return these values.
        atoms.set_array("forces", forces)
        atoms.info["stress"] = stress
        atoms.info["energy"] = energy

        atoms.set_calculator(
            SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
        )

        # Write the atom object to the database.
        # FAIRChem’s AseDBDataset will later load the atoms via ase.db.
        db.write(atoms)

    print(f"Generated {n_samples} samples and saved database to: {db_path}")


def main():
    n_atoms = 2
    n_samples = 1
    global stats_df
    stats_df = pd.read_csv("rattled-300-subsampled_statistics.csv")

    # Define dataset directory and file
    dataset_name = f"{n_atoms}_atoms_{n_samples}_samples"
    dataset_dir = os.path.join("datasets", "vc_dim", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    db_path = os.path.join(dataset_dir, f"{dataset_name}.db")

    # Generate dataset with desired parameters using random forces, stress, and energy.
    generate_vc_dim_dataset(db_path, n_samples=n_samples)


if __name__ == "__main__":
    main()
