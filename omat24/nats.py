import os
import shutil
import math


def zip_dataset(dataset_name, local_dir):
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

    print(f"Estimated information content of the zipped file: {nats:.3f} nats\n")
    return nats


if __name__ == "__main__":
    dataset_name = "val-rattled-1000"
    # zip & calculate nats
    zip_file = zip_dataset(dataset_name, f"./datasets/{dataset_name}")
    calculate_nats_from_zip(zip_file)

    # delete zipped file
    os.remove(zip_file)
