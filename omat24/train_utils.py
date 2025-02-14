# External
import torch
import torch.nn as nn
from typing import Union, Dict

# Internal
from loss import compute_loss
from log_utils import partial_json_log, tensorboard_log
from torch_geometric.data import Batch


def forward_pass(
    model: nn.Module,
    batch: Union[Dict, Batch],
    graph: bool,
    training: bool,
    device: torch.device,
):
    """A common forward pass function for inference across different architectures & dataloaders.

    Args:
        model (nn.Module): PyTorch model to use.
        batch (Union[Dict, Batch]): The batch of data to forward pass.
        graph (bool): Whether the model is a graph-based model.
        training (bool): Whether the model is in training mode.
        device (torch.device): The device to run the model on.
    """
    if training or graph:
        context_manager = torch.enable_grad()
    else:
        context_manager = torch.no_grad()

    with context_manager:
        if type(batch) == dict:

            atomic_numbers = batch["atomic_numbers"].to(device, non_blocking=True)
            positions = batch["positions"].to(device, non_blocking=True)
            factorized_distances = batch["factorized_matrix"].to(
                device, non_blocking=True
            )
            true_forces = batch["forces"].to(device, non_blocking=True)
            true_energy = batch["energy"].to(device, non_blocking=True)
            true_stress = batch["stress"].to(device, non_blocking=True)
            mask = atomic_numbers != 0
            natoms = mask.sum(dim=1)

            pred_forces, pred_energy, pred_stress = model(
                atomic_numbers, positions, factorized_distances, mask
            )

        elif isinstance(batch, Batch):
            # PyG Batch
            atomic_numbers = batch.atomic_numbers.to(device, non_blocking=True)
            positions = batch.pos.to(device, non_blocking=True)
            edge_index = batch.edge_index.to(device, non_blocking=True)
            structure_index = batch.batch.to(device, non_blocking=True)
            true_forces = batch.forces.to(device, non_blocking=True)
            true_energy = batch.energy.to(device, non_blocking=True)
            true_stress = batch.stress.to(device, non_blocking=True)
            mask = None
            natoms = batch.natoms

            pred_forces, pred_energy, pred_stress = model(
                atomic_numbers, positions, edge_index, structure_index
            )

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
        ) = forward_pass(model, batch, graph, False, device)

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


def run_validation(model, val_loader, graph, device):
    """
    Compute and return the average validation loss.

    Args:
        model (nn.Module): The PyTorch model to validate.
        val_loader (DataLoader): The validation data loader.
        device (torch.device): The device to run validation on.

    Returns:
        float: The average validation loss across the validation set.
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
            model=model, batch=batch, graph=graph, training=False, device=device
        )

        # Mapping atoms to their respective structures (for graphs)
        structure_index = batch.batch if graph and hasattr(batch, "batch") else []
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
    patience=50,
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
):
    """
    Train model with validation at epoch 0 and every 'validate_every' epochs.
    Includes early stopping and optional JSON + TensorBoard logging.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler, or None.
        pbar (tqdm): A tqdm progress bar initialized with the total number of epochs.
        device (torch.device): The device to run training on.
        patience (int): Early stopping patience (number of checks with no improvement).
        results_path (str, optional): Path to JSON results file. If provided, partial logs are written.
        experiment_results (dict, optional): Dict for storing experiment results.
        data_size_key (str, optional): Key to label experiment_results by dataset size.
        run_entry (dict, optional): Dictionary describing the current run (e.g., model_name, config).
        writer (SummaryWriter, optional): TensorBoard writer for logging.
        tensorboard_prefix (str, optional): Prefix for naming logs in TensorBoard.
        num_visualization_samples (int, optional): Number of samples to visualize in logs.

    Returns:
        (nn.Module, dict): The trained model and a dictionary of recorded losses.
    """
    model.to(device)
    can_write_partial = all(
        [results_path, experiment_results, data_size_key, run_entry]
    )
    losses = {}

    # Initial validation at epoch 0
    (
        val_loss,
        val_energy_loss,
        val_force_loss,
        val_stress_iso_loss,
        val_stress_aniso_loss,
    ) = run_validation(model, val_loader, graph, device)
    losses[0] = {"val_loss": float(val_loss)}
    if writer is not None:
        tensorboard_log(
            val_loss,
            "",
            train=False,
            writer=writer,
            epoch=0,
            tensorboard_prefix=tensorboard_prefix,
        )
        tensorboard_log(
            val_energy_loss,
            "energy",
            train=False,
            writer=writer,
            epoch=0,
            tensorboard_prefix=tensorboard_prefix,
        )
        tensorboard_log(
            val_force_loss,
            "force",
            train=False,
            writer=writer,
            epoch=0,
            tensorboard_prefix=tensorboard_prefix,
        )
        tensorboard_log(
            val_stress_iso_loss,
            "stress_iso",
            train=False,
            writer=writer,
            epoch=0,
            tensorboard_prefix=tensorboard_prefix,
        )
        tensorboard_log(
            val_stress_aniso_loss,
            "stress_aniso",
            train=False,
            writer=writer,
            epoch=0,
            tensorboard_prefix=tensorboard_prefix,
        )

    # Write partial JSON if everything is provided
    if can_write_partial:
        partial_json_log(
            experiment_results,
            data_size_key,
            run_entry,
            0,
            float("nan"),
            val_loss,
            results_path,
        )

    # Early stopping setup
    best_val_loss = val_loss
    epochs_since_improvement = 0
    last_val_loss = val_loss
    samples = None

    # Training loop
    for epoch in range(1, len(pbar) + 1):
        model.train()
        train_loss_sum = 0.0
        energy_loss_sum = 0.0
        force_loss_sum = 0.0
        stress_iso_loss_sum = 0.0
        stress_aniso_loss_sum = 0.0

        n_train_batches = len(train_loader)

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
                model=model, batch=batch, graph=graph, training=True, device=device
            )

            # Mapping atoms to their respective structures (for graphs)
            structure_index = batch.batch if graph and hasattr(batch, "batch") else []
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            train_loss_sum += total_train_loss.item()
            energy_loss_sum += train_loss_dict["energy_loss"].item()
            force_loss_sum += train_loss_dict["force_loss"].item()
            stress_iso_loss_sum += train_loss_dict["stress_iso_loss"].item()
            stress_aniso_loss_sum += train_loss_dict["stress_aniso_loss"].item()
            current_avg_loss = train_loss_sum / (batch_idx + 1)

            pbar.set_description(
                f"train_loss={current_avg_loss:.2f} val_loss={last_val_loss:.2f}"
            )

        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()

        avg_epoch_train_loss = train_loss_sum / n_train_batches
        avg_epoch_energy_loss = energy_loss_sum / n_train_batches
        avg_epoch_force_loss = force_loss_sum / n_train_batches
        avg_epoch_stress_iso_loss = stress_iso_loss_sum / n_train_batches
        avg_epoch_stress_aniso_loss = stress_aniso_loss_sum / n_train_batches

        losses[epoch] = {"train_loss": float(avg_epoch_train_loss)}

        # TensorBoard logging for training loss
        if writer is not None:
            # Log parameter norms (example usage)
            tensorboard_log(
                avg_epoch_train_loss,
                "",
                train=True,
                writer=writer,
                epoch=epoch,
                tensorboard_prefix=tensorboard_prefix,
            )
            tensorboard_log(
                avg_epoch_energy_loss,
                "energy",
                train=True,
                writer=writer,
                epoch=epoch,
                tensorboard_prefix=tensorboard_prefix,
            )
            tensorboard_log(
                avg_epoch_force_loss,
                "force",
                train=True,
                writer=writer,
                epoch=epoch,
                tensorboard_prefix=tensorboard_prefix,
            )
            tensorboard_log(
                avg_epoch_stress_iso_loss,
                "stress_iso",
                train=True,
                writer=writer,
                epoch=epoch,
                tensorboard_prefix=tensorboard_prefix,
            )
            tensorboard_log(
                avg_epoch_stress_aniso_loss,
                "stress_aniso",
                train=True,
                writer=writer,
                epoch=epoch,
                tensorboard_prefix=tensorboard_prefix,
            )
            # Simple gradient logging for debugging (skip bias layers)
            for name, param in model.named_parameters():
                if (
                    param is not None
                    and param.requires_grad
                    and param.grad is not None
                    and not name.endswith("bias")
                ):  # Skip bias layers
                    # Log mean gradient - key indicator for vanishing/exploding gradients
                    grad_mean = param.grad.abs().mean().item()
                    writer.add_scalar(
                        f"{tensorboard_prefix}/Grads/{name}",
                        grad_mean,
                        global_step=epoch,
                    )

                    # Log gradient-to-weight ratio - indicates if updates are well-scaled
                    grad_to_weight = (
                        param.grad.abs().mean() / (param.data.abs().mean() + 1e-8)
                    ).item()
                    writer.add_scalar(
                        f"{tensorboard_prefix}/G2W/{name}",
                        grad_to_weight,
                        global_step=epoch,
                    )

        # Validate every 'validate_every' epochs
        if epoch % validate_every == 0:
            (
                val_loss,
                val_energy_loss,
                val_force_loss,
                val_stress_iso_loss,
                val_stress_aniso_loss,
            ) = run_validation(model, val_loader, graph, device)
            if writer is not None:
                tensorboard_log(
                    val_loss,
                    "",
                    train=False,
                    writer=writer,
                    epoch=epoch,
                    tensorboard_prefix=tensorboard_prefix,
                )
                tensorboard_log(
                    val_energy_loss,
                    "energy",
                    train=False,
                    writer=writer,
                    epoch=epoch,
                    tensorboard_prefix=tensorboard_prefix,
                )
                tensorboard_log(
                    val_force_loss,
                    "force",
                    train=False,
                    writer=writer,
                    epoch=epoch,
                    tensorboard_prefix=tensorboard_prefix,
                )
                tensorboard_log(
                    val_stress_iso_loss,
                    "stress_iso",
                    train=False,
                    writer=writer,
                    epoch=epoch,
                    tensorboard_prefix=tensorboard_prefix,
                )
                tensorboard_log(
                    val_stress_aniso_loss,
                    "stress_aniso",
                    train=False,
                    writer=writer,
                    epoch=epoch,
                    tensorboard_prefix=tensorboard_prefix,
                )
            last_val_loss = val_loss
            losses[epoch]["val_loss"] = float(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    return model, losses

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
            )

        pbar.update(1)

    return model, losses
