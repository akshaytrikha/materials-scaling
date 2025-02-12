# External
import torch

# Internal
from loss import compute_loss
from log_utils import partial_json_log, collect_train_val_samples, tensorboard_log
from torch_geometric.data import Batch


def forward_pass(model, batch, graph: bool, training: bool, device):
    """"""
    if training or graph:
        context_manager = torch.enable_grad()
    else:
        context_manager = torch.no_grad()

    with context_manager:
        if type(batch) == dict:
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            factorized_distances = batch["factorized_matrix"].to(device)
            true_forces = batch["forces"].to(device)
            true_energy = batch["energy"].to(device)
            true_stress = batch["stress"].to(device)
            mask = atomic_numbers != 0
            natoms = mask.sum(dim=1)

            pred_forces, pred_energy, pred_stress = model(
                atomic_numbers, positions, factorized_distances, mask
            )

        elif isinstance(batch, Batch):
            # PyG Batch
            true_forces = batch.forces.to(device)
            true_energy = batch.energy.to(device)
            true_stress = batch.stress.to(device)
            mask = None
            natoms = batch.natoms

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
    total_val_loss = 0.0
    num_val_batches = len(val_loader)

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
        ) = forward_pass(model, batch, graph, False, device)

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
        )
        total_val_loss += val_loss_dict["total_loss"].item()

    if num_val_batches == 0:
        return float("inf")
    return total_val_loss / num_val_batches


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
    val_loss = run_validation(model, val_loader, graph, device)
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
            ) = forward_pass(model, batch, graph, True, device)

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
            )
            total_train_loss = train_loss_dict["total_loss"]
            total_train_loss.backward()
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
            val_loss = run_validation(model, val_loader, graph, device)
            last_val_loss = val_loss
            losses[epoch]["val_loss"] = float(val_loss)

            # Also log validation loss to TensorBoard
            if writer is not None:
                tensorboard_log(
                    val_loss,
                    "",
                    train=False,
                    writer=writer,
                    epoch=epoch,
                    tensorboard_prefix=tensorboard_prefix,
                )

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
            samples = collect_train_val_samples(
                model,
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
