import os
from pathlib import Path
import requests
import tarfile
import argparse
from tqdm import tqdm

train_base_url = (
    "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train"
)
val_base_url = (
    "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val"
)


valid_datasets = [
    "rattled-1000",
    "rattled-1000-subsampled",
    "rattled-500",
    "rattled-500-subsampled",
    "rattled-300",
    "rattled-300-subsampled",
    "aimd-from-PBE-3000-npt",
    "aimd-from-PBE-3000-nvt",
    "aimd-from-PBE-1000-npt",
    "aimd-from-PBE-1000-nvt",
    "rattled-relax",
]


def get_dataset_url(dataset_name: str, split_name: str):
    if split_name == "train":
        return f"{train_base_url}/{dataset_name}.tar.gz"
    elif split_name == "val":
        return f"{val_base_url}/{dataset_name}.tar.gz"
    else:
        raise ValueError(f"Invalid split name: {split_name}")


def download_dataset(dataset_name: str, split_name: str):
    """Downloads a compressed dataset from a predefined URL and extracts it to the specified directory.

    Args:
        dataset_name (str): The key corresponding to the dataset in the DATASETS dictionary.
        split_name (str): The split to download ("train" or "val")

    Raises:
        ValueError: If the dataset_name is not valid or the split_name is not valid.
        Exception: If there is an error during the extraction or deletion of the compressed file.
    """
    if split_name not in ["train", "val"]:
        raise ValueError(f"Invalid split name: {split_name}")

    if dataset_name not in valid_datasets:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Get the URL for the dataset
    url = get_dataset_url(dataset_name, split_name)

    # Create the necessary directories
    os.makedirs("./datasets", exist_ok=True)
    os.makedirs(f"./datasets/{split_name}", exist_ok=True)
    dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
    compressed_path = dataset_path.with_suffix(".tar.gz")

    # Download the dataset
    print(f"Starting download from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get total file size for progress bar
    total_size = int(response.headers.get("content-length", 0))

    # Download with progress bar
    with open(str(compressed_path), "wb") as f:
        with tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {dataset_name}",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

    # Extract the dataset
    print(f"Extracting {compressed_path}...")
    with tarfile.open(compressed_path, "r:gz") as tar:
        tar.extractall(path=dataset_path.parent)
    print(f"Extraction completed. Files are available at {dataset_path}.")

    # Delete the compressed file
    try:
        compressed_path.unlink()
        print(f"Deleted the compressed file {compressed_path}.")
    except Exception as e:
        print(f"An error occurred while deleting {compressed_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download OMAT datasets")
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=valid_datasets,
        help="Names of datasets to download",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Split types to download (default: both train and val)",
    )

    args = parser.parse_args()

    for dataset in args.datasets:
        for split in args.splits:
            download_dataset(dataset, split)


if __name__ == "__main__":
    main()
