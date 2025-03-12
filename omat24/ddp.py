import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
from pathlib import Path

# Import existing data functions
from data import get_dataloaders
from data_utils import download_dataset, VALID_DATASETS

# Set seed & device
SEED = 1024
torch.manual_seed(SEED)


def setup_ddp(rank, world_size):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP process group."""
    dist.destroy_process_group()


def test_dataloader(rank, world_size, args):
    """Test dataloader functionality."""
    # Set up DDP
    setup_ddp(rank, world_size)
    print(f"Rank {rank} setup complete")

    # Create dataset path
    dataset_names = args.datasets
    dataset_paths = [
        Path(f"{args.datasets_base_path}/{args.split}/{dataset_name}")
        for dataset_name in dataset_names
    ]

    if rank == 0:
        print(f"Testing data loading for datasets: {dataset_names}")
        print(f"Using {world_size} GPUs")

    try:
        # Get dataloaders
        train_loader, val_loader = get_dataloaders(
            dataset_paths=dataset_paths,
            train_data_fraction=args.data_fraction,
            batch_size=args.batch_size,
            seed=SEED,
            architecture="FCN",  # Use FCN format for testing
            batch_padded=False,
            val_data_fraction=0.1,
            train_workers=args.workers,
            val_workers=args.workers,
            graph=False,
            factorize=False,
            distributed=True,  # Enable distributed sampler
        )

        # Print info about dataloaders
        if rank == 0:
            print(f"Successfully created dataloaders!")
            print(f"Training samples: {len(train_loader.dataset)}")
            print(f"Validation samples: {len(val_loader.dataset)}")
            print(f"Training batches: {len(train_loader)}")
            print(f"Validation batches: {len(val_loader)}")

        # Test fetching a batch
        if rank == 0:
            print("Fetching first batch from train loader...")
        for batch_idx, batch in enumerate(train_loader):
            if rank == 0:
                print(f"Successfully loaded batch {batch_idx+1}/{len(train_loader)}")
                print(f"Batch keys: {batch.keys()}")
                print(f"Atomic numbers shape: {batch['atomic_numbers'].shape}")
            
            # Only test the first batch
            if batch_idx == 0:
                break

        if rank == 0:
            print("Dataloader test completed successfully!")

    except Exception as e:
        print(f"Rank {rank} encountered error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="DDP dataloader testing")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["rattled-300-subsampled"],
        choices=VALID_DATASETS + ["all"],
        help="Dataset name(s) or 'all' for all datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split",
    )
    parser.add_argument(
        "--datasets-base-path",
        type=str,
        default="./datasets",
        help="Base path for datasets",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--data-fraction", type=float, default=0.1, help="Fraction of dataset to use"
    )

    args = parser.parse_args()
    if args.datasets == ["all"]:
        args.datasets = VALID_DATASETS
    
    print("Datasets to test:")
    for ds in args.datasets:
        print(f"  - {ds}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        return

    # Get world size
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("No GPUs available. Exiting...")
        return

    print(f"Using {world_size} GPUs")

    # Launch processes
    mp.spawn(test_dataloader, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    # Need spawn method for CUDA initialization
    mp.set_start_method("spawn", force=True)
    main()
