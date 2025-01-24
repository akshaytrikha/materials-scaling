import sys
import json
import os
from pathlib import Path
from collections import defaultdict


def merge_json_files(file_1_name, file_2_name, output_name):
    """Merges two JSON experiment files into a single JSON file with sorted keys."""
    file_1_path = Path(f"results/{file_1_name}")
    file_2_path = Path(f"results/{file_2_name}")
    output_path = Path(f"results/{output_name}")

    # Check if input files exist
    if not os.path.isfile(file_1_path):
        print(f"Error: File '{file_1_path}' does not exist.")
        sys.exit(1)
    if not os.path.isfile(file_2_path):
        print(f"Error: File '{file_2_path}' does not exist.")
        sys.exit(1)

    try:
        # Load the first JSON file
        with open(file_1_path, "r") as f1:
            data1 = json.load(f1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file '{file_1_path}': {e}")
        sys.exit(1)

    try:
        # Load the second JSON file
        with open(file_2_path, "r") as f2:
            data2 = json.load(f2)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file '{file_2_path}': {e}")
        sys.exit(1)

    # Initialize a defaultdict to handle merging lists easily
    merged_data = defaultdict(list)

    # Add data from the first file
    for dataset_size, experiments in data1.items():
        merged_data[dataset_size].extend(experiments)

    # Add data from the second file
    for dataset_size, experiments in data2.items():
        merged_data[dataset_size].extend(experiments)

    # Convert defaultdict back to a regular dict
    merged_dict = dict(merged_data)

    # Sort the keys numerically
    sorted_merged_dict = {k: merged_dict[k] for k in sorted(merged_dict, key=int)}

    try:
        # Write the merged data to the output file with pretty formatting
        with open(output_path, "w") as outfile:
            json.dump(sorted_merged_dict, outfile, indent=4, allow_nan=True)
        print(f"Merged JSON saved to '{output_path}' with sorted keys.")
    except IOError as e:
        print(f"Error writing to file '{output_path}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    file_2_name = ""
    file_1_name = ""
    output_name = "merged.json"

    merge_json_files(file_1_name, file_2_name, output_name)
