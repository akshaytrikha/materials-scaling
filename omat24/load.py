# External
from pathlib import Path

# Internal
from data import OMat24Dataset, download_dataset, get_dataloaders


dataset_name = "rattled-300-subsampled"
dataset_path = Path(f"datasets/{dataset_name}")

if not dataset_path.exists():
    # Fetch and uncompress dataset
    download_dataset(dataset_name)

dataset = OMat24Dataset(dataset_path=dataset_path)
sample = dataset[0]
print(sample["atomic_numbers"], sample["positions"], sample["energy"])


# dataloading
train_loader, val_loader = get_dataloaders(dataset, data_fraction=0.1, batch_size=32)

for batch in train_loader:
    print(len(batch))

breakpoint()
