# External
from pathlib import Path

# Internal
from data import OMat24Dataset, download_dataset


dataset_name = "rattled-300-subsampled"
dataset_path = Path(f"datasets/{dataset_name}")

if not dataset_path.exists():
    # Fetch and uncompress dataset
    download_dataset(dataset_name)

dataset = OMat24Dataset(dataset_path=dataset_path)
sample = dataset[0]
print(sample["atomic_numbers"], sample["positions"], sample["energy"])
