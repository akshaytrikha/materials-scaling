# External
from pathlib import Path
from tqdm.auto import tqdm
from fairchem.core.datasets import AseDBDataset

# Internal
from data_utils import download_dataset

split_name = "val"
set_of_dataset_names = {"rattled-300-subsampled", "rattled-1000"}

# Process each dataset and populate the dictionary
for dataset_name in set_of_dataset_names:
    dataset_path = Path(f"datasets/val/{dataset_name}")

    # Download dataset if not already present
    if not dataset_path.exists():
        download_dataset(dataset_name, split_name)

    # Load the dataset
    # dataset = OMat24Dataset(dataset_path=dataset_path, augment=False)
    dataset = AseDBDataset(config=dict(src=str(dataset_path)))

    # Calculate the max number of atoms
    max_n_atoms = 0
    for sample in tqdm(dataset):
        num_n_atoms = sample.natoms
        max_n_atoms = max(max_n_atoms, num_n_atoms)
    print(f"{dataset_name}: {max_n_atoms}")
