import csv
import os
import numpy as np
from tqdm.auto import tqdm
from fairchem.core.datasets import AseDBDataset


split_name = "val"
dataset_names = os.listdir(f"./datasets/{split_name}")
dataset_paths = [f"datasets/{split_name}/{x}" for x in dataset_names]

for i, dataset_path in enumerate(dataset_paths):
    # Load dataset
    dataset = AseDBDataset(config=dict(src=dataset_path))

    # Open output CSV file
    os.makedirs(f"metadata/{split_name}", exist_ok=True)
    with open(
        f"metadata/{split_name}/{dataset_names[i]}_metadata.csv", "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(
            [
                "ID",
                "Composition",
                "NumAtoms",
                # "Lattice_a",
                # "Lattice_b",
                # "Lattice_c",
                # "Alpha",
                # "Beta",
                # "Gamma",
                # "Volume",
                # "DFT_Functional",
                # "Energy(eV)",
                # "MaxForce(eV/Å)",
                # "MaxStress(GPa)",
            ]
        )

        # Iterate over all entries (this may be 100M+ iterations)
        for idx in tqdm(range(len(dataset))):
            atoms = dataset.get_atoms(idx)  # fetch Atoms object for entry
            formula = atoms.get_chemical_formula(mode="hill")  # e.g., "Fe2O3"
            num_atoms = len(atoms)
            # Lattice cell parameters (Å and degrees)
            a, b, c = atoms.cell.lengths()  # cell vector lengths
            alpha, beta, gamma = atoms.cell.angles()  # cell angles
            volume = atoms.get_volume()
            # Metadata from atoms.info dictionary (if present)
            info = atoms.info
            # DFT details might be constant; assuming PBE for all, or store if provided
            dft_func = info.get("functional", "PBE")
            # # Targets (energy, forces, stress)
            # energy = atoms.get_potential_energy()  # total DFT energy in eV
            # forces = atoms.get_forces()  # array of forces on each atom
            # max_force = (
            #     (abs(forces) ** 2).sum(axis=1) ** (0.5).max() if forces is not None else ""
            # )
            # stress = atoms.get_stress()  # stress tensor components
            # max_stress = max(stress) if stress is not None else ""

            # Write the row
            writer.writerow(
                [
                    idx,
                    formula,
                    num_atoms,
                    # np.round(a, 5),
                    # np.round(b, 5),
                    # np.round(c, 5),
                    # np.round(alpha, 5),
                    # np.round(beta, 5),
                    # np.round(gamma, 5),
                    # np.round(volume, 5),
                    # dft_func,
                    # energy,
                    # max_force,
                    # max_stress,
                ]
            )
