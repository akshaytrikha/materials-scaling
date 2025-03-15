import os
from typing import List
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
from pathlib import Path
import random
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from fairchem.core.datasets import AseDBDataset
import bisect

# Set sharing strategy before any other multiprocessing operations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
SEED = 1024
torch.manual_seed(SEED)

# Simplified list of valid datasets
VALID_DATASETS = [
    "rattled-300-subsampled",
    "rattled-500-subsampled",
    "rattled-1000-subsampled",
    "rattled-300",
    "rattled-500",
    "rattled-1000",
    "aimd-from-PBE-1000-npt",
    "aimd-from-PBE-1000-nvt",
    "aimd-from-PBE-3000-npt",
    "aimd-from-PBE-3000-nvt",
    "rattled-relax",
]


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


# Simplified helper functions
def pad_tensor(tensor, pad_size, dim=0, padding_value=0.0):
    """Pad a tensor to a specified size along a given dimension."""
    pad_length = pad_size - tensor.size(dim)
    if pad_length <= 0:
        return tensor

    padding_shape = list(tensor.shape)
    padding_shape[dim] = pad_length
    padding = torch.full(
        padding_shape, padding_value, dtype=tensor.dtype, device=tensor.device
    )
    return torch.cat([tensor, padding], dim=dim)


def pad_matrix(matrix, pad_size_x, pad_size_y, padding_value=0.0):
    """Pad a 2D tensor to the specified dimensions."""
    # Pad rows if needed
    if matrix.size(0) < pad_size_x:
        pad_row = torch.full(
            (pad_size_x - matrix.size(0), matrix.size(1)),
            padding_value,
            dtype=matrix.dtype,
            device=matrix.device,
        )
        matrix = torch.cat([matrix, pad_row], dim=0)

    # Pad columns if needed
    if matrix.size(1) < pad_size_y:
        pad_col = torch.full(
            (matrix.size(0), pad_size_y - matrix.size(1)),
            padding_value,
            dtype=matrix.dtype,
            device=matrix.device,
        )
        matrix = torch.cat([matrix, pad_col], dim=1)

    return matrix


def compute_distance_matrix(positions):
    """Compute pairwise distance matrix from atomic positions."""
    # Simple Euclidean distance calculation
    n_atoms = positions.size(0)
    dist_mat = torch.zeros(
        (n_atoms, n_atoms), dtype=positions.dtype, device=positions.device
    )

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = torch.norm(positions[i] - positions[j])
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    return dist_mat


def split_dataset(dataset, train_data_fraction, val_data_fraction, seed):
    """Splits a dataset into training and validation subsets."""
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_data_fraction)
    remaining_size = dataset_size - val_size
    train_size = max(1, int(remaining_size * train_data_fraction))

    random.seed(seed)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size : val_size + train_size]

    train_subset = Subset(dataset, indices=train_indices)
    val_subset = Subset(dataset, indices=val_indices)

    return train_subset, val_subset, train_indices, val_indices


class SimpleDatasetPaddedCollate:
    """Simplified collate function for padded tensors."""

    def __init__(self, max_n_atoms):
        self.max_n_atoms = max_n_atoms

    def __call__(self, batch):
        atomic_numbers = [sample["atomic_numbers"] for sample in batch]
        positions = [sample["positions"] for sample in batch]
        distance_matrices = [sample["distance_matrix"] for sample in batch]
        energies = torch.stack([sample["energy"] for sample in batch], dim=0)
        forces = [sample["forces"] for sample in batch]
        stresses = torch.stack([sample["stress"] for sample in batch], dim=0)

        # Pad to max_n_atoms
        padded_atomic_numbers = torch.stack(
            [
                pad_tensor(tensor, self.max_n_atoms, dim=0, padding_value=0)
                for tensor in atomic_numbers
            ],
            dim=0,
        )
        padded_positions = torch.stack(
            [
                pad_tensor(tensor, self.max_n_atoms, dim=0, padding_value=0.0)
                for tensor in positions
            ],
            dim=0,
        )
        padded_forces = torch.stack(
            [
                pad_tensor(tensor, self.max_n_atoms, dim=0, padding_value=0.0)
                for tensor in forces
            ],
            dim=0,
        )
        padded_distance_matrices = torch.stack(
            [
                pad_matrix(
                    tensor, self.max_n_atoms, self.max_n_atoms, padding_value=0.0
                )
                for tensor in distance_matrices
            ],
            dim=0,
        )

        return {
            "atomic_numbers": padded_atomic_numbers,
            "positions": padded_positions,
            "distance_matrix": padded_distance_matrices,
            "energy": energies,
            "forces": padded_forces,
            "stress": stresses,
        }


class OMat24Dataset(Dataset):
    def __init__(
        self,
        dataset_paths: List[Path],
        debug: bool = False,
    ):
        print(f"[INIT] Initializing dataset in process {os.getpid()}")
        self.debug = debug
        self.dataset_paths = dataset_paths
        self.has_init = False

    def _init_dataset(self):
        print(f"[INIT AseDBDataset] Initializing AseDBDataset in process {os.getpid()}")
        self.dataset = AseDBDataset(config=dict(src=self.dataset_paths))
        self.has_init = True

    def __len__(self):
        print(f"[LEN] Getting length in process {os.getpid()}")
        if not self.has_init:
            print(f"[LEN] Initializing dataset in process {os.getpid()}")
            self._init_dataset()
        return len(self.dataset)

    def __getitem__(self, idx):
        print(f"[GETITEM] Getting item in process {os.getpid()}")
        if not self.has_init:
            print(f"[GETITEM] Initializing dataset in process {os.getpid()}")
            self._init_dataset()

        print(f"[GETITEM] Returning item in process {os.getpid()}") 
        return {
            "idx": idx,
        }


class ConcatMinimalDataset(ConcatDataset):
    """A thin wrapper around ConcatDataset."""

    pass


def worker_init_fn(worker_id):
    """Initialize worker-specific resources"""
    print(f"[WORKER_INIT] Initializing worker {worker_id} in process {os.getpid()}")
    # Each worker needs its own random seed
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)

    # Workers will initialize their database connections when needed
    # through the lazy loading mechanism in get_atoms


def get_dataloaders(
    dataset_paths,
    train_data_fraction,
    batch_size,
    seed,
    val_data_fraction=0.1,
    train_workers=0,
    val_workers=0,
    distributed=False,
    debug=False,
):
    """Creates training and validation DataLoaders from dataset paths."""
    if debug:
        print(
            f"[DATALOADER] Loading datasets from: {dataset_paths} in process {os.getpid()}"
        )

    # train_subsets = []
    # val_subsets = []

    max_n_atoms = 236

    # for path in dataset_paths:
    #     if debug:
    #         print(f"[DATALOADER] Loading dataset from {path} in process {os.getpid()}")

    #     dataset = MinimalOMat24Dataset(dataset_paths=[path], debug=debug)

    #     if debug:
    #         print(f"[DATALOADER] Dataset size: {len(dataset)} in process {os.getpid()}")

    #     train_subset, val_subset, _, _ = split_dataset(
    #         dataset, train_data_fraction, val_data_fraction, seed
    #     )

    #     if debug:
    #         print(f"[DATALOADER] Split sizes - Train: {len(train_subset)}, Val: {len(val_subset)} in process {os.getpid()}")

    #     train_subsets.append(train_subset)
    #     val_subsets.append(val_subset)

    train_dataset = OMat24Dataset(dataset_paths=dataset_paths, debug=debug)
    val_dataset = OMat24Dataset(dataset_paths=dataset_paths, debug=debug)

    if debug:
        print(
            f"[DATALOADER] Combined dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)} in process {os.getpid()}"
        )

    # Configure samplers for DDP
    if distributed:
        if debug:
            print(f"[DATALOADER] Using DistributedSampler in process {os.getpid()}")
        train_sampler = DistributedSampler(train_dataset, seed=seed)
        val_sampler = DistributedSampler(val_dataset, seed=seed, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    # Create collate function
    collate_fn = SimpleDatasetPaddedCollate(max_n_atoms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle and not distributed,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=train_workers,
        persistent_workers=train_workers > 0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,  # Add worker initialization
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=val_workers,
        persistent_workers=val_workers > 0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,  # Add worker initialization
    )

    if debug:
        print(
            f"[DATALOADER] Created DataLoaders - Train batches: {len(train_loader)}, Val batches: {len(val_loader)} in process {os.getpid()}"
        )

    return train_loader, val_loader


def test_dataloader(rank, world_size, args):
    """Test dataloader functionality."""
    # Set up DDP
    setup_ddp(rank, world_size)
    print(f"[RANK {rank}] DDP setup complete in process {os.getpid()}")

    # Create dataset path
    dataset_names = args.datasets
    dataset_paths = [
        Path(f"{args.datasets_base_path}/{args.split}/{dataset_name}")
        for dataset_name in dataset_names
    ]

    if rank == 0:
        print(f"[RANK {rank}] Testing data loading for datasets: {dataset_names}")
        print(f"[RANK {rank}] Using {world_size} GPUs")

    try:
        # Get dataloaders with debug flag enabled
        train_loader, val_loader = get_dataloaders(
            dataset_paths=dataset_paths,
            train_data_fraction=args.data_fraction,
            batch_size=args.batch_size,
            seed=SEED,
            val_data_fraction=0.1,
            train_workers=args.workers,
            val_workers=args.workers,
            distributed=True,
            debug=True,  # Only show debug info on rank 0
        )

        # Print info about dataloaders
        if rank == 0:
            print(f"[RANK {rank}] Successfully created dataloaders!")
            print(f"[RANK {rank}] Training samples: {len(train_loader.dataset)}")
            print(f"[RANK {rank}] Validation samples: {len(val_loader.dataset)}")
            print(f"[RANK {rank}] Training batches: {len(train_loader)}")
            print(f"[RANK {rank}] Validation batches: {len(val_loader)}")

        # Test fetching a batch
        if rank == 0:
            print(f"[RANK {rank}] Fetching first batch from train loader...")

        # Synchronize before fetching batch to ensure all processes are ready
        torch.cuda.synchronize()
        dist.barrier()
        print(f"[RANK {rank}] Barrier passed in process {os.getpid()}")

        for batch_idx, batch in enumerate(train_loader):
            if rank == 0:
                print(
                    f"[RANK {rank}] Successfully loaded batch {batch_idx+1}/{len(train_loader)}"
                )
                print(f"[RANK {rank}] Batch keys: {batch.keys()}")
                # print(
                #     f"[RANK {rank}] Atomic numbers shape: {batch['atomic_numbers'].shape}"
                # )
            else:
                print(f"[RANK {rank}] Loaded batch {batch_idx+1}")

            # Only test the first batch
            if batch_idx == 0:
                break

        # Synchronize after fetching to ensure all processes complete
        torch.cuda.synchronize()
        dist.barrier()
        print(f"[RANK {rank}] Final barrier passed in process {os.getpid()}")

        if rank == 0:
            print("[RANK 0] Dataloader test completed successfully!")

    except Exception as e:
        print(f"[RANK {rank}] Encountered error in process {os.getpid()}: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up
        cleanup_ddp()
        print(f"[RANK {rank}] DDP cleanup complete in process {os.getpid()}")


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
    print(f"Main process ID: {os.getpid()}")

    # Launch processes
    mp.spawn(test_dataloader, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    # Need spawn method for CUDA initialization
    mp.set_start_method("spawn", force=True)
    main()
