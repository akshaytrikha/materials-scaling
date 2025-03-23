# External
import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
from fairchem.core.datasets import AseDBDataset

# Internal
from data import OMat24Dataset
from data_utils import download_dataset, VALID_DATASETS


def process_chunk(chunk_indices, dataset_path):
    """Process a chunk of the dataset."""
    config_kwargs = {}
    ase_dataset = AseDBDataset(config=dict(src=str(dataset_path), **config_kwargs))

    symbols = []
    positions = []
    atomic_numbers = []
    forces = []
    energy = []
    stress = []

    for i in chunk_indices:
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


def get_info_parallel(dataset_path, num_processes=None):
    """
    Parallel version of get_info function using multiprocessing.

    Args:
        dataset_path: Path to the dataset
        num_processes: Number of processes to use (default: CPU count)

    Returns:
        Tuple of (symbols, positions, atomic_numbers, forces, energy, stress)
    """
    config_kwargs = {}
    # Just to get the dataset size
    ase_dataset = AseDBDataset(config=dict(src=str(dataset_path), **config_kwargs))
    dataset_size = len(ase_dataset.ids)

    # Determine number of processes if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Ensure we don't create more processes than necessary
    num_processes = min(num_processes, dataset_size)

    # Divide the work evenly
    indices = list(range(dataset_size))
    chunks = np.array_split(indices, num_processes)
    chunks = [chunk.tolist() for chunk in chunks]  # Convert to list

    # Create a pool of workers
    pool = mp.Pool(processes=num_processes)

    # Create a partial function with the dataset_path
    process_func = partial(process_chunk, dataset_path=dataset_path)

    # Process all chunks in parallel with a progress bar
    print(f"Processing dataset with {num_processes} processes...")
    results = list(
        tqdm(
            pool.imap(process_func, chunks), total=len(chunks), desc="Processing chunks"
        )
    )

    # Close the pool
    pool.close()
    pool.join()

    # Combine results
    symbols = []
    positions = []
    atomic_numbers = []
    forces = []
    energy = []
    stress = []

    for r in results:
        symbols.extend(r[0])
        positions.extend(r[1])
        atomic_numbers.extend(r[2])
        forces.extend(r[3])
        energy.extend(r[4])
        stress.extend(r[5])

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
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of processes to use (default: CPU count)",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        for split in args.splits:
            dataset_path = Path(f"{args.base_path}/{split}/{dataset}")

            symbols, positions, atomic_numbers, forces, energy, stress = (
                get_info_parallel(dataset_path, num_processes=args.num_processes)
            )

            dataset = OMat24Dataset(
                symbols, positions, atomic_numbers, forces, energy, stress
            )

            # save dataset to pickle
            with open(dataset_path / f"{dataset_path.name}.pkl", "wb") as f:
                pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
