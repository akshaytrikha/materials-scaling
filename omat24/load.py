# External
from pathlib import Path

# Internal
from data import OMat24Dataset, get_dataloaders
from data_utils import download_dataset
from tqdm.auto import tqdm

dataset_name = "rattled-300-subsampled"
dataset_path = Path(f"datasets/val/{dataset_name}")

if not dataset_path.exists():
    # Fetch and uncompress dataset
    download_dataset(dataset_name, "val")

dataset = OMat24Dataset(dataset_path=dataset_path, augment=False)
sample = dataset[0]
print(
    sample["atomic_numbers"],
    sample["positions"],
    sample["energy"],
    sample["forces"],
    sample["stress"],
)

# dataloading
train_loader, val_loader = get_dataloaders(
    dataset, train_data_fraction=0.1, batch_size=32, batch_padded=False, seed=100
)

for i, batch in tqdm(enumerate(train_loader)):
    pass
