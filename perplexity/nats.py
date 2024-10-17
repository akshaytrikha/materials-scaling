import os
import shutil
import math
from datasets import load_dataset


def save_and_zip_wikitext(dataset_name, local_dir):
    """Save and zip wikitext dataset"""
    dataset = load_dataset("wikitext", dataset_name)
    os.makedirs(local_dir, exist_ok=True)

    # Save train, validation, and test splits in JSONL format
    dataset["train"].to_json(f"{local_dir}/train.jsonl")
    dataset["validation"].to_json(f"{local_dir}/validation.jsonl")
    dataset["test"].to_json(f"{local_dir}/test.jsonl")

    # Zip entire directory
    zip_file_path = shutil.make_archive(local_dir, "zip", local_dir)
    print(f"Dataset {dataset_name} zipped at {zip_file_path}")

    return zip_file_path


def calculate_nats_from_zip(zip_file_path):
    """Calculate the number of nats based on the size of the zipped file"""
    # Get the size of the zipped file in bytes
    file_size_bytes = os.path.getsize(zip_file_path)

    # Convert size to bits and then to nats
    nats = (file_size_bytes * 8) / math.log(2)

    print(
        f"Estimated information content of the zipped file {zip_file_path}: {nats:.2f} nats\n"
    )
    return nats


# Download, save, zip, and calculate nats
for dataset_name in ["wikitext-2-raw-v1", "wikitext-103-raw-v1"]:
    zip_file = save_and_zip_wikitext(dataset_name, f"./{dataset_name}")
    calculate_nats_from_zip(zip_file)
