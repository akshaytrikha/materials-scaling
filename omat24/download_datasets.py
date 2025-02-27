# External
import argparse

# Internal
from data_utils import download_dataset, VALID_DATASETS


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
            download_dataset(dataset, split, args.base_path)


if __name__ == "__main__":
    main()
