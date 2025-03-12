import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from pathlib import Path

# Import existing data functions
from data import get_dataloaders
from data_utils import download_dataset, VALID_DATASETS

# Set seed & device
SEED = 1024
torch.manual_seed(SEED)


class ToyModel(nn.Module):
    """A simple toy model for testing DDP."""

    def __init__(self, n_elements=100):
        super().__init__()
        self.name = "ToyModel"
        self.num_params = 14610  # Just for logging

        # Simple embedding layer
        self.embedding = nn.Embedding(n_elements, 16)

        # Energy prediction branch
        self.energy_net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))

        # Forces prediction branch
        self.forces_net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))

        # Stress prediction branch
        self.stress_net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 6))

    def forward(self, atomic_numbers, positions, distance_matrix, mask=None):
        # Get embeddings and apply mask
        x = self.embedding(atomic_numbers)  # [batch_size, max_atoms, 16]

        if mask is not None:
            # Apply mask to zero out padding
            mask = mask.unsqueeze(-1)  # [batch_size, max_atoms, 1]
            x = x * mask

        # Pooling over atoms dimension for energy and stress
        x_mean = x.mean(dim=1)  # [batch_size, 16]

        # Predict energy (scalar per structure)
        energy = self.energy_net(x_mean)  # [batch_size, 1]

        # Predict forces (3D vector per atom)
        # Simple implementation - in reality forces would be derivatives of energy
        forces = self.forces_net(x)  # [batch_size, max_atoms, 3]

        # Predict stress (6D vector per structure: xx, yy, zz, xy, xz, yz)
        stress = self.stress_net(x_mean)  # [batch_size, 6]

        return forces, energy, stress


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


def train_epoch(model, train_loader, optimizer, device, rank):
    """Run one epoch of training."""
    model.train()
    total_loss = 0.0

    print(f"Training on {len(train_loader.dataset)} samples")
    for batch_idx, batch in enumerate(train_loader):
        print(f"Training batch {batch_idx} of {len(train_loader)}")
        optimizer.zero_grad()

        # Extract data from batch
        atomic_numbers = batch["atomic_numbers"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        distance_matrix = batch["distance_matrix"].to(device, non_blocking=True)
        true_forces = batch["forces"].to(device, non_blocking=True)
        true_energy = batch["energy"].to(device, non_blocking=True)
        true_stress = batch["stress"].to(device, non_blocking=True)
        mask = atomic_numbers != 0

        # Forward pass
        pred_forces, pred_energy, pred_stress = model(
            atomic_numbers, positions, distance_matrix, mask
        )

        # Simple loss calculation
        energy_loss = torch.mean((pred_energy - true_energy.unsqueeze(1)) ** 2)
        forces_loss = torch.mean((pred_forces - true_forces) ** 2)
        stress_loss = torch.mean((pred_stress - true_stress) ** 2)

        # Total loss
        loss = energy_loss + forces_loss + stress_loss

        # Backward pass
        loss.backward()

        # Synchronize gradients across processes
        if dist.is_initialized():
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= dist.get_world_size()

        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0 and rank == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    """Run validation."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            # Extract data from batch
            atomic_numbers = batch["atomic_numbers"].to(device, non_blocking=True)
            positions = batch["positions"].to(device, non_blocking=True)
            distance_matrix = batch["distance_matrix"].to(device, non_blocking=True)
            true_forces = batch["forces"].to(device, non_blocking=True)
            true_energy = batch["energy"].to(device, non_blocking=True)
            true_stress = batch["stress"].to(device, non_blocking=True)
            mask = atomic_numbers != 0

            # Forward pass
            pred_forces, pred_energy, pred_stress = model(
                atomic_numbers, positions, distance_matrix, mask
            )

            # Simple loss calculation
            energy_loss = torch.mean((pred_energy - true_energy.unsqueeze(1)) ** 2)
            forces_loss = torch.mean((pred_forces - true_forces) ** 2)
            stress_loss = torch.mean((pred_stress - true_stress) ** 2)

            # Total loss
            loss = energy_loss + forces_loss + stress_loss
            total_loss += loss.item()

    return total_loss / len(val_loader)


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def run_training(rank, world_size, args):
    """Main training function."""
    # Set up DDP
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Create dataset path
    dataset_names = args.datasets
    dataset_paths = [
        Path(f"{args.datasets_base_path}/{args.split}/{dataset_name}")
        for dataset_name in dataset_names
    ]

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        dataset_paths=dataset_paths,
        train_data_fraction=args.data_fraction,
        batch_size=args.batch_size,
        seed=SEED,
        architecture="FCN",  # Use FCN format for toy model
        batch_padded=False,
        val_data_fraction=0.1,
        train_workers=args.workers,
        val_workers=args.workers,
        graph=False,
        factorize=False,
        distributed=True,  # Enable distributed sampler
    )

    # Create model
    model = ToyModel(n_elements=args.n_elements)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Print training info on rank 0
    if rank == 0:
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        print(f"Using {world_size} GPUs")

    # Training loop
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, rank)

        # Reduce train loss across processes
        train_loss_tensor = torch.tensor(train_loss, device=device)
        train_loss = reduce_tensor(train_loss_tensor, world_size).item()

        # Validate
        val_loss = validate(model, val_loader, device)

        # Reduce val loss across processes
        val_loss_tensor = torch.tensor(val_loss, device=device)
        val_loss = reduce_tensor(val_loss_tensor, world_size).item()

        # Print results on rank 0
        if rank == 0:
            print(
                f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        # Synchronize before next epoch
        dist.barrier()

    # Save model on rank 0
    if rank == 0:
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"toy_model_ddp_{world_size}gpus.pt",
        )
        print(f"Model saved to toy_model_ddp_{world_size}gpus.pt")

    # Clean up
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Simple DDP training example")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["rattled-300-subsampled"],
        choices=VALID_DATASETS,
        help="Dataset name",
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
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--n-elements", type=int, default=100, help="Number of possible elements"
    )
    parser.add_argument(
        "--data-fraction", type=float, default=0.1, help="Fraction of dataset to use"
    )

    args = parser.parse_args()
    if args.datasets == "all":
        args.datasets = VALID_DATASETS
    print("Datasets:")
    print(args.datasets)

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
    mp.spawn(run_training, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    # Need spawn method for CUDA initialization
    mp.set_start_method("spawn", force=True)
    main()
