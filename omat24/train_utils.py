# External
import copy
import torch
import torch.nn as nn
from typing import Union
from torch.utils.flop_counter import FlopCounterMode
from contextlib import nullcontext
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Internal
from loss import compute_loss
from log_utils import partial_json_log, log_tb_metrics
from torch_geometric.data import Batch


def reduce_losses(tensor, average=True):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= dist.get_world_size()
    return rt


def forward_pass(
    model: nn.Module,
    batch: Union[dict, Batch],
    graph: bool,
    training: bool,
    device: torch.device,
    factorize: bool,
):
    """A common forward pass function for inference across different architectures & dataloaders.

    Args:
        model (nn.Module): PyTorch model to use.
        batch (Union[Dict, Batch]): The batch of data to forward pass.
        graph (bool): Whether the model is a graph-based model.
        training (bool): Whether the model is in training mode.
        device (torch.device): The device to run the model on.
        factorize (bool): Whether to factorize the distance matrix.
    """
    if training or graph:
        context_manager = torch.enable_grad()
    else:
        context_manager = torch.no_grad()

    # Get the model name, handling both regular models and DDP-wrapped models
    model_name = (
        model.name
        if hasattr(model, "name")
        else model.module.name if hasattr(model, "module") else None
    )

    with context_manager:
        if type(batch) == dict:

            atomic_numbers = batch["atomic_numbers"].to(device, non_blocking=True)
            positions = batch["positions"].to(device, non_blocking=True)
            true_forces = batch["forces"].to(device, non_blocking=True)
            true_energy = batch["energy"].to(device, non_blocking=True)
            true_stress = batch["stress"].to(device, non_blocking=True)
            mask = atomic_numbers != 0
            natoms = mask.sum(dim=1).to(device)

            if factorize:
                factorized_distances = batch["factorized_matrix"].to(
                    device, non_blocking=True
                )
                pred_forces, pred_energy, pred_stress = model(
                    atomic_numbers, positions, factorized_distances, mask
                )
            else:
                distance_matrix = batch["distance_matrix"].to(device, non_blocking=True)
                pred_forces, pred_energy, pred_stress = model(
                    atomic_numbers, positions, distance_matrix, mask
                )

        elif isinstance(batch, Batch):
            # PyG Batch
            atomic_numbers = batch.atomic_numbers.to(device, non_blocking=True)
            positions = batch.pos.to(device, non_blocking=True)
            true_forces = batch.forces.to(device, non_blocking=True)
            true_energy = batch.energy.to(device, non_blocking=True)
            true_stress = batch.stress.to(device, non_blocking=True)
            mask = None
            if hasattr(batch, "natoms"):
                natoms = (
                    batch.natoms.to(device)
                    if hasattr(batch.natoms, "to")
                    else torch.tensor(batch.natoms, device=device)
                )
            else:
                natoms = None

            if model_name == "SchNet":
                edge_index = batch.edge_index.to(device, non_blocking=True)
                structure_index = batch.batch.to(device, non_blocking=True)

                pred_forces, pred_energy, pred_stress = model(
                    atomic_numbers,
                    positions,
                    edge_index,
                    structure_index,
                )
            elif model_name == "EquiformerV2":
                # equiformer constructs graphs internally
                batch = batch.to(device)
                pred_forces, pred_energy, pred_stress = model(batch)

    return (
        pred_forces,
        pred_energy,
        pred_stress,
        true_forces,
        true_energy,
        true_stress,
        mask,
        natoms,
    )


def collect_samples_helper(num_visualization_samples, dataset, model, graph, device):
    samples = []
    for i in range(min(num_visualization_samples, len(dataset))):
        if graph:
            batch = Batch.from_data_list([dataset[i]])
            positions = batch["pos"]
            atomic_numbers = batch["atomic_numbers"]
            sample_length = len(atomic_numbers)
            idx = batch["idx"].cpu().tolist()[0]
        else:
            batch = {
                k: torch.unsqueeze(v, 0) if isinstance(v, torch.Tensor) else v
                for k, v in dataset[i].items()
            }
            # Extract data from batch and sqeeuze batch dimension
            positions = batch["positions"].squeeze(0)
            atomic_numbers = batch["atomic_numbers"].squeeze(0)
            sample_length = (batch["atomic_numbers"] != 0).sum(dim=1)[0].item()
            idx = batch["idx"]

        symbols = batch["symbols"]

        (
            pred_forces,
            pred_energy,
            pred_stress,
            true_forces,
            true_energy,
            true_stress,
            _,
            _,
        ) = forward_pass(model, batch, graph, False, device, False)

        if not graph:
            pred_forces = pred_forces.squeeze(0)
            true_forces = true_forces.squeeze(0)
            pred_stress = pred_stress.squeeze(0)
            true_stress = true_stress.squeeze(0)

        samples.append(
            {
                "idx": idx,
                "symbols": symbols,
                "atomic_numbers": atomic_numbers[:sample_length].cpu().tolist(),
                "positions": positions[:sample_length].cpu().tolist(),
                "true": {
                    "forces": true_forces[:, :sample_length].cpu().tolist(),
                    "energy": true_energy.cpu().tolist()[0],
                    "stress": true_stress.cpu().tolist(),
                },
                "pred": {
                    "forces": pred_forces[:, :sample_length].cpu().tolist(),
                    "energy": pred_energy.cpu().tolist()[0],
                    "stress": pred_stress.cpu().tolist(),
                },
            }
        )
    return samples


def collect_samples_for_visualizing(
    model, graph, train_loader, val_loader, device, num_visualization_samples
):
    """Collect samples and predictions from both training and validation sets.
    Uses fixed indices for consistent visualization.

    Args:
        model (torch.nn.Module): Trained model.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        device (torch.device): Device to run model on.
        num_visualization_samples (int): Number of samples to visualize.

    Returns:
        dict: Dictionary containing samples and predictions for training and validation sets.
    """
    # Get the underlying dataset from the DataLoader
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    return {
        "train": collect_samples_helper(
            num_visualization_samples, train_dataset, model, graph, device
        ),
        "val": collect_samples_helper(
            num_visualization_samples, val_dataset, model, graph, device
        ),
    }


def run_validation(model, val_loader, graph, device, factorize):
    """
    Compute and return the average validation loss.

    Args:
        model (nn.Module): The PyTorch model to validate.
        val_loader (DataLoader): The validation data loader.
        device (torch.device): The device to run validation on.
        graph (bool): Whether the model is graph-based.
        factorize (bool): Whether to use factorized distance matrices.

    Returns:
        tuple: The average validation losses (total, energy, force, stress_iso, stress_aniso).
    """
    model.to(device)
    model.eval()
    val_loss_sum = 0.0
    energy_loss_sum = 0.0
    force_loss_sum = 0.0
    stress_iso_loss_sum = 0.0
    stress_aniso_loss_sum = 0.0
    n = len(val_loader)

    for batch in val_loader:
        (
            pred_forces,
            pred_energy,
            pred_stress,
            true_forces,
            true_energy,
            true_stress,
            mask,
            natoms,
        ) = forward_pass(
            model=model,
            batch=batch,
            graph=graph,
            training=False,
            device=device,
            factorize=factorize,
        )

        # Ensure natoms is on the correct device
        if natoms is not None and natoms.device != device:
            natoms = natoms.to(device)

        # Mapping atoms to their respective structures (for graphs)
        structure_index = batch.batch if graph and hasattr(batch, "batch") else []

        # Fix: Check if structure_index is a non-empty tensor before checking device
        is_tensor = isinstance(structure_index, torch.Tensor)
        has_elements = is_tensor and structure_index.numel() > 0
        wrong_device = (
            has_elements
            and hasattr(structure_index, "device")
            and structure_index.device != device
        )

        if wrong_device:
            structure_index = structure_index.to(device)

        val_loss_dict = compute_loss(
            pred_forces,
            pred_energy,
            pred_stress,
            true_forces,
            true_energy,
            true_stress,
            mask,
            device,
            natoms,
            graph,
            structure_index,
        )
        val_loss_sum += val_loss_dict["total_loss"].item()
        energy_loss_sum += val_loss_dict["energy_loss"].item()
        force_loss_sum += val_loss_dict["force_loss"].item()
        stress_iso_loss_sum += val_loss_dict["stress_iso_loss"].item()
        stress_aniso_loss_sum += val_loss_dict["stress_aniso_loss"].item()

    if n == 0:
        return float("inf")
    return (
        val_loss_sum / n,
        energy_loss_sum / n,
        force_loss_sum / n,
        stress_iso_loss_sum / n,
        stress_aniso_loss_sum / n,
    )


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    pbar,
    graph,
    device,
    distributed=False,
    rank=0,
    patience=5,
    factorize=False,
    results_path=None,
    experiment_results=None,
    data_size_key=None,
    run_entry=None,
    writer=None,
    tensorboard_prefix="model",
    num_visualization_samples=3,
    gradient_clip=1,
    validate_every=500,
    visualize_every=500,
    use_fsdp=False,
):
    """
    Train model with validation at epoch 0 and every 'validate_every' epochs.
    Includes early stopping and optional JSON + TensorBoard logging.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): PyTorch optimizer.
        scheduler (LRScheduler, optional): Learning rate scheduler.
        pbar (tqdm.tqdm): Progress bar for tracking epochs.
        graph (bool): Whether the model architecture is graph-based.
        device (torch.device): Device to run training on.
        distributed (bool, optional): Whether to use distributed training.
        rank (int, optional): Process rank for distributed training.
        patience (int, optional): Number of epochs to wait for improvement before early stopping.
        factorize (bool, optional): Whether to use factorized distance matrices.
        results_path (Path, optional): Path to save training results.
        experiment_results (dict, optional): Experiment results dictionary to update.
        data_size_key (str, optional): Key for dataset size in experiment_results.
        run_entry (dict, optional): Run entry to update in experiment_results.
        writer (SummaryWriter, optional): TensorBoard writer.
        tensorboard_prefix (str, optional): Prefix for TensorBoard metrics.
        num_visualization_samples (int, optional): Number of samples to visualize.
        gradient_clip (float, optional): Maximum gradient norm for clipping.
        validate_every (int, optional): Validate every this many epochs.
        visualize_every (int, optional): Visualize every this many epochs.
        use_fsdp (bool, optional): Whether to use FSDP mode.

    Returns:
        tuple: (trained_model, losses dictionary)
    """
    is_main_process = (not distributed) or (rank == 0)
    can_write_partial = results_path is not None and is_main_process

    # Initialize losses dictionary
    losses = {}
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    best_val_model = None
    best_val_loss_dict = None
    val_loss = float("inf")

    # Check if we need to validate at epoch 0
    if validate_every > 0:
        (
            val_loss,
            val_energy_loss,
            val_force_loss,
            val_stress_iso_loss,
            val_stress_aniso_loss,
        ) = run_validation(model, val_loader, graph, device, factorize)

        if distributed:
            val_loss_tensor = torch.tensor(val_loss, device=device)
            val_energy_loss_tensor = torch.tensor(val_energy_loss, device=device)
            val_force_loss_tensor = torch.tensor(val_force_loss, device=device)
            val_stress_iso_loss_tensor = torch.tensor(
                val_stress_iso_loss, device=device
            )
            val_stress_aniso_loss_tensor = torch.tensor(
                val_stress_aniso_loss, device=device
            )

            if use_fsdp:
                # For FSDP we use fsdp.all_reduce
                val_loss = fsdp.all_reduce(val_loss_tensor).item()
                val_energy_loss = fsdp.all_reduce(val_energy_loss_tensor).item()
                val_force_loss = fsdp.all_reduce(val_force_loss_tensor).item()
                val_stress_iso_loss = fsdp.all_reduce(val_stress_iso_loss_tensor).item()
                val_stress_aniso_loss = fsdp.all_reduce(
                    val_stress_aniso_loss_tensor
                ).item()
            else:
                # For DDP we use our reduce_losses function
                val_loss = reduce_losses(val_loss_tensor, average=True).item()
                val_energy_loss = reduce_losses(
                    val_energy_loss_tensor, average=True
                ).item()
                val_force_loss = reduce_losses(
                    val_force_loss_tensor, average=True
                ).item()
                val_stress_iso_loss = reduce_losses(
                    val_stress_iso_loss_tensor, average=True
                ).item()
                val_stress_aniso_loss = reduce_losses(
                    val_stress_aniso_loss_tensor, average=True
                ).item()

        losses[0] = {"val_loss": val_loss}
        best_val_loss = val_loss
        best_val_model = copy.deepcopy(model)
        best_val_loss_dict = copy.deepcopy(losses)

        if writer is not None:
            log_tb_metrics(
                {
                    "": val_loss,
                    "energy": val_energy_loss,
                    "force": val_force_loss,
                    "stress_iso": val_stress_iso_loss,
                    "stress_aniso": val_stress_aniso_loss,
                },
                writer,
                0,
                tensorboard_prefix,
                train=False,
            )

    # Calculate the number of batches
    n_train_batches = len(train_loader)

    # Use FlopCounterMode to count FLOPs, but only once and only for the first epoch
    flops_per_epoch = 0
    flop_counter = FlopCounterMode()

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        energy_loss_sum = 0.0
        force_loss_sum = 0.0
        stress_iso_loss_sum = 0.0
        stress_aniso_loss_sum = 0.0

        context = flop_counter if epoch == 1 else nullcontext()
        with context:
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                (
                    pred_forces,
                    pred_energy,
                    pred_stress,
                    true_forces,
                    true_energy,
                    true_stress,
                    mask,
                    natoms,
                ) = forward_pass(
                    model=model,
                    batch=batch,
                    graph=graph,
                    training=True,
                    device=device,
                    factorize=factorize,
                )
                # Mapping atoms to their respective structures (for graphs)
                model_name = (
                    model.name
                    if hasattr(model, "name")
                    else model.module.name if hasattr(model, "module") else None
                )
                structure_index = (
                    batch.batch if graph and hasattr(batch, "batch") else []
                )
                train_loss_dict = compute_loss(
                    pred_forces,
                    pred_energy,
                    pred_stress,
                    true_forces,
                    true_energy,
                    true_stress,
                    mask,
                    device,
                    natoms,
                    graph,
                    structure_index,
                )
                total_train_loss = train_loss_dict["total_loss"]
                total_train_loss.backward()

                if distributed:
                    if use_fsdp:
                        # FSDP handles gradient synchronization internally
                        pass
                    else:
                        # DDP needs manual gradient synchronization
                        for param in model.parameters():
                            if param.grad is not None:
                                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                                param.grad.data /= dist.get_world_size()

                # Handle gradient clipping differently for FSDP
                if use_fsdp:
                    # FSDP requires a different clipping approach
                    FSDP.clip_grad_norm_(model, gradient_clip)
                else:
                    # Standard gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                optimizer.step()

                train_loss_sum += total_train_loss.item()
                energy_loss_sum += train_loss_dict["energy_loss"].item()
                force_loss_sum += train_loss_dict["force_loss"].item()
                stress_iso_loss_sum += train_loss_dict["stress_iso_loss"].item()
                stress_aniso_loss_sum += train_loss_dict["stress_aniso_loss"].item()
                current_avg_loss = train_loss_sum / (batch_idx + 1)

                if is_main_process:
                    pbar.set_description(
                        f"train_loss={current_avg_loss:.2f} val_loss={val_loss:.2f}"
                    )

        if epoch == 1:
            flops_per_epoch = flop_counter.get_total_flops()

        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()

        if distributed:
            train_loss_tensor = torch.tensor(train_loss_sum, device=device)
            energy_loss_tensor = torch.tensor(energy_loss_sum, device=device)
            force_loss_tensor = torch.tensor(force_loss_sum, device=device)
            stress_iso_loss_tensor = torch.tensor(stress_iso_loss_sum, device=device)
            stress_aniso_loss_tensor = torch.tensor(
                stress_aniso_loss_sum, device=device
            )

            if use_fsdp:
                # For FSDP we use fsdp.all_reduce
                train_loss_sum = fsdp.all_reduce(train_loss_tensor).item()
                energy_loss_sum = fsdp.all_reduce(energy_loss_tensor).item()
                force_loss_sum = fsdp.all_reduce(force_loss_tensor).item()
                stress_iso_loss_sum = fsdp.all_reduce(stress_iso_loss_tensor).item()
                stress_aniso_loss_sum = fsdp.all_reduce(stress_aniso_loss_tensor).item()
            else:
                # For DDP we use our reduce_losses function
                train_loss_sum = reduce_losses(train_loss_tensor, average=True).item()
                energy_loss_sum = reduce_losses(energy_loss_tensor, average=True).item()
                force_loss_sum = reduce_losses(force_loss_tensor, average=True).item()
                stress_iso_loss_sum = reduce_losses(
                    stress_iso_loss_tensor, average=True
                ).item()
                stress_aniso_loss_sum = reduce_losses(
                    stress_aniso_loss_tensor, average=True
                ).item()

        avg_epoch_train_loss = train_loss_sum / n_train_batches
        avg_epoch_energy_loss = energy_loss_sum / n_train_batches
        avg_epoch_force_loss = force_loss_sum / n_train_batches
        avg_epoch_stress_iso_loss = stress_iso_loss_sum / n_train_batches
        avg_epoch_stress_aniso_loss = stress_aniso_loss_sum / n_train_batches

        losses[epoch] = {"train_loss": float(avg_epoch_train_loss)}

        # When collecting validation loss:
        if epoch % validate_every == 0:
            # Inside the validate_every block, use FSDP-specific code for validation loss reduction:
            if distributed:
                if use_fsdp:
                    val_loss = fsdp.all_reduce(val_loss_tensor).item()
                    val_energy_loss = fsdp.all_reduce(val_energy_loss_tensor).item()
                    val_force_loss = fsdp.all_reduce(val_force_loss_tensor).item()
                    val_stress_iso_loss = fsdp.all_reduce(
                        val_stress_iso_loss_tensor
                    ).item()
                    val_stress_aniso_loss = fsdp.all_reduce(
                        val_stress_aniso_loss_tensor
                    ).item()
                else:
                    # Your existing DDP code...
                    val_loss = reduce_losses(val_loss_tensor, average=True).item()
                    val_energy_loss = reduce_losses(
                        val_energy_loss_tensor, average=True
                    ).item()
                    val_force_loss = reduce_losses(
                        val_force_loss_tensor, average=True
                    ).item()
                    val_stress_iso_loss = reduce_losses(
                        val_stress_iso_loss_tensor, average=True
                    ).item()
                    val_stress_aniso_loss = reduce_losses(
                        val_stress_aniso_loss_tensor, average=True
                    ).item()

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                best_val_model = copy.deepcopy(model)
                best_val_loss_dict = copy.deepcopy(losses)
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    return best_val_model, best_val_loss_dict

        # Visualization samples every 'visualize_every' epochs
        if epoch % visualize_every == 0:
            samples = collect_samples_for_visualizing(
                model,
                graph,
                train_loader,
                val_loader,
                device,
                num_visualization_samples,
            )

        # Write partial JSON if requested
        if can_write_partial:
            partial_json_log(
                experiment_results,
                data_size_key,
                run_entry,
                epoch,
                avg_epoch_train_loss,
                val_loss if epoch % validate_every == 0 else float("nan"),
                results_path,
                samples if epoch % visualize_every == 0 else None,
                flops_per_epoch * epoch,
            )

        if is_main_process:
            pbar.update(1)

    return model, losses
