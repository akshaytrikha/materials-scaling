"""
Script to generate a synthetic ASE database for two-atom systems with opposite forces.
The generated database will be saved in "datasets/vc_dim/ase.db" and can be loaded with FAIRChem's AseDBDataset.

Each sample is a hydrogen dimer with:
  - Positions: atoms at [-d/2,0,0] and [d/2,0,0] (with d chosen randomly)
  - Forces: [F, 0, 0] on the first atom and [-F, 0, 0] on the second atom
  - Energy: set to 0.0 (or you can change it)
  - Stress: a 6-component zero tensor
"""

import os
import random
import numpy as np
from ase import Atoms
from ase.db import connect


def generate_vc_dim_dataset(db_path, n_samples=10, force_magnitude=1.0):
    """Generate a synthetic dataset with two-atom structures.

    Parameters:
        db_path (str): Path to the ASE database file to be created.
        n_samples (int): Number of synthetic samples to generate.
        force_magnitude (float): Magnitude of the force (in eV/Å) applied oppositely.
    """
    # Open (or create) the ASE database
    db = connect(db_path)

    for i in range(n_samples):
        # Random interatomic distance (in Å)
        d = random.uniform(0.8, 1.2)
        pos1 = [-d / 2, 0.0, 0.0]
        pos2 = [d / 2, 0.0, 0.0]
        atoms = Atoms("H2", positions=[pos1, pos2])

        # Define forces: opposite forces on the two atoms.
        forces = np.array([[force_magnitude, 0.0, 0.0], [-force_magnitude, 0.0, 0.0]])

        # Define stress (6 components; here set to zero)
        stress = np.zeros(6)

        # Set a (dummy) energy (e.g., 0.0 eV)
        energy = 0.0

        # Attach arrays so that get_forces() and get_stress() return these values.
        atoms.set_array("forces", forces)
        atoms.info["stress"] = stress
        atoms.info["energy"] = energy

        # Write the atom object to the database.
        # FAIRChem’s AseDBDataset will later load the atoms via ase.db.
        db.write(atoms)

    print(f"Generated {n_samples} samples and saved database to: {db_path}")


def main():
    n_atoms = 2
    n_samples = 1

    # Define dataset directory and file
    dataset_name = f"{n_atoms}_atoms_{n_samples}_samples"
    dataset_dir = os.path.join("datasets", "vc_dim", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    db_path = os.path.join(dataset_dir, f"{dataset_name}.db")

    # Generate dataset with desired parameters
    generate_vc_dim_dataset(db_path, n_samples=n_samples, force_magnitude=1.0)


if __name__ == "__main__":
    main()
