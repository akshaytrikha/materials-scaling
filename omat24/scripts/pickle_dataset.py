# External
import argparse
from fairchem.core.datasets import AseDBDataset
import numpy as np
import pickle
from pathlib import Path
from tqdm.auto import tqdm

# Internal
from data import OMat24Dataset
from data_utils import VALID_DATASETS


def get_info(dataset_path):
    config_kwargs = {}
    ase_dataset = AseDBDataset(config=dict(src=str(dataset_path), **config_kwargs))
    symbols = []
    positions = []
    atomic_numbers = []
    forces = []
    energy = []
    stress = []

    for i in tqdm(range(len(ase_dataset.ids))):
        atoms = ase_dataset.get_atoms(i)
        symbols.append(atoms.get_chemical_formula())

        # inputs
        positions.append(
            np.concatenate(
                [
                    atoms.get_positions(wrap=True),
                    atoms.get_scaled_positions(wrap=True),
                ],
                axis=1,
            )
        )
        atomic_numbers.append(atoms.get_atomic_numbers())

        # labels
        forces.append(atoms.get_forces())
        energy.append(atoms.get_potential_energy())
        stress.append(atoms.get_stress())

    return symbols, positions, atomic_numbers, forces, energy, stress


def main():
    parser = argparse.ArgumentParser(description="Download OMat24 datasets")
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=VALID_DATASETS,
        help="Names of datasets to download",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Split types to download (default: both train and val)",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="./datasets",
        help="Base path for dataset storage",
    )

    args = parser.parse_args()

    for dataset in args.datasets:
        for split in args.splits:
            dataset_path = Path(f"{args.base_path}/{split}/{dataset}")
            symbols, positions, atomic_numbers, forces, energy, stress = get_info(
                dataset_path
            )

            dataset = OMat24Dataset(
                symbols, positions, atomic_numbers, forces, energy, stress
            )

            # save dataset to pickle
            with open(dataset_path / f"{dataset_path.name}.pkl", "wb") as f:
                pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
