import pickle
import torch
import os
import argparse
from pathlib import Path
from omat24.ddp import MinimalOMat24Dataset

def test_pickle(dataset_path):
    """Test if the dataset can be pickled and unpickled successfully."""
    print(f"Testing pickling for dataset at: {dataset_path}")
    
    # Initialize the dataset
    dataset = MinimalOMat24Dataset(dataset_paths=[dataset_path], debug=True)
    print(f"Original dataset length: {len(dataset)}")
    
    # Get a sample before pickling
    try:
        sample_before = dataset[0]
        print(f"Sample 0 before pickling - atomic numbers shape: {sample_before['atomic_numbers'].shape}")
    except Exception as e:
        print(f"Error accessing sample before pickling: {str(e)}")
    
    # Pickle the dataset
    print("Pickling dataset...")
    pickled_dataset = pickle.dumps(dataset)
    print(f"Pickle size: {len(pickled_dataset) / (1024 * 1024):.2f} MB")
    
    # Unpickle the dataset
    print("Unpickling dataset...")
    unpickled_dataset = pickle.loads(pickled_dataset)
    print(f"Unpickled dataset length: {len(unpickled_dataset)}")
    
    # Get a sample after unpickling
    try:
        sample_after = unpickled_dataset[0]
        print(f"Sample 0 after unpickling - atomic numbers shape: {sample_after['atomic_numbers'].shape}")
        print("Successfully accessed sample after unpickling!")
    except Exception as e:
        print(f"Error accessing sample after unpickling: {str(e)}")
    
    # Validate that samples are the same before and after
    try:
        atoms_equal = torch.equal(sample_before['atomic_numbers'], sample_after['atomic_numbers'])
        positions_equal = torch.equal(sample_before['positions'], sample_after['positions'])
        energy_equal = torch.equal(sample_before['energy'], sample_after['energy'])
        
        print(f"Validation - Atomic numbers match: {atoms_equal}")
        print(f"Validation - Positions match: {positions_equal}")
        print(f"Validation - Energy match: {energy_equal}")
        
        if atoms_equal and positions_equal and energy_equal:
            print("SUCCESS: Dataset can be properly pickled and unpickled!")
        else:
            print("WARNING: Dataset pickled but data doesn't match!")
    except Exception as e:
        print(f"Error validating samples: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dataset pickling")
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        required=True,
        help="Dataset name (e.g., rattled-300-subsampled)"
    )
    parser.add_argument(
        "--datasets-base-path",
        type=str,
        default="./datasets",
        help="Base path for datasets"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split"
    )
    
    args = parser.parse_args()
    
    # Construct the full dataset path
    dataset_path = Path(f"{args.datasets_base_path}/{args.split}/{args.dataset_name}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist.")
        exit(1)
        
    test_pickle(dataset_path)