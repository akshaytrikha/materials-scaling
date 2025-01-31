# External
import torch
import json
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Internal
from loss import compute_loss


def partial_json_log(
    experiment_results,
    data_size_key,
    run_entry,
    step,
    avg_train_loss,
    val_loss,
    results_path,
    val_samples=[],
):
    """
    Append train_loss and val_loss for the given step to the specified run_entry in experiment_results,
    then write the updated experiment_results dictionary to disk.

    Args:
        experiment_results (dict): Existing JSON data for all experiments.
        data_size_key (str): Dataset size as a string key (e.g., "256").
        run_entry (dict): Dict with "model_name", "config", "losses", etc.
        step (int): Current training step (for indexing in the 'losses' dict).
        avg_train_loss (float): Current averaged training loss.
        val_loss (float): Current validation loss.
        results_path (str): Path to the JSON file where logs are written.
    """
    if data_size_key not in experiment_results:
        experiment_results[data_size_key] = []

    # Check if this run_entry is already in the list for data_size_key
    found_existing = False
    for existing_run in experiment_results[data_size_key]:
        if existing_run.get("model_name", "") == run_entry["model_name"]:
            found_existing = True
            if "losses" not in existing_run:
                existing_run["losses"] = {}
            existing_run["losses"][str(step)] = {
                "train_loss": float(avg_train_loss),
                "val_loss": float(val_loss),
                "val_samples": val_samples,
            }
            break

    if not found_existing:
        run_entry["losses"] = {
            str(step): {
                "train_loss": float(avg_train_loss),
                "val_loss": float(val_loss),
                "val_samples": val_samples,
            }
        }
        experiment_results[data_size_key].append(run_entry)

    with open(results_path, "w") as f:
        json.dump(experiment_results, f, indent=2)


def tensorboard_log(
    loss_dict: dict, train: bool, writer: SummaryWriter, epoch, tensorboard_prefix: str
):
    """
    Log losses to TensorBoard.

    Args:
        train_loss_dict (dict): Dictionary of training losses.
        val_loss_dict (dict): Dictionary of validation losses.
        writer (SummaryWriter): TensorBoard SummaryWriter object.
        step (int): Current training step.
        tensorboard_prefix (str): Prefix for naming the logs in TensorBoard.
    """
    if train:
        log_var_name_prefix = "train"
    else:
        log_var_name_prefix = "val"

    for loss_name, loss_value in loss_dict.items():
        writer.add_scalar(
            f"{tensorboard_prefix}/{log_var_name_prefix}_{loss_name}",
            loss_value,
            global_step=epoch,
        )


def run_validation(
    model,
    val_loader,
    device,
    writer,
    epoch: int,
    tensorboard_prefix: str,
    num_samples=3,
):
    """Compute and return the average validation loss."""
    model.to(device)
    model.eval()
    total_val_loss = 0.0
    num_val_batches = len(val_loader)

    val_samples = []
    with torch.no_grad():
        for batch in val_loader:
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            true_forces = batch["forces"].to(device)
            true_energy = batch["energy"].to(device)
            true_stress = batch["stress"].to(device)

            pred_forces, pred_energy, pred_stress = model(atomic_numbers, positions)

            # Get samples from this batch if we still need more
            if len(val_samples) < num_samples:
                batch_samples_needed = num_samples - len(val_samples)
                for i in range(min(batch_samples_needed, atomic_numbers.shape[0])):
                    # Get the length of non-zero atomic numbers
                    sample_length = (
                        (atomic_numbers[i : i + 1] != 0).sum(dim=1)[0].item()
                    )
                    val_samples.append(
                        {
                            "sample": {
                                "atomic_numbers": atomic_numbers[
                                    i : i + 1, :sample_length
                                ]
                                .cpu()
                                .tolist(),
                                "positions": positions[i : i + 1, :sample_length]
                                .cpu()
                                .tolist(),
                                "forces": true_forces[i : i + 1, :sample_length]
                                .cpu()
                                .tolist(),
                                "energy": true_energy[i : i + 1].cpu().tolist(),
                                "stress": true_stress[i : i + 1].cpu().tolist(),
                            },
                            "pred": {
                                "forces": pred_forces[i : i + 1, :sample_length]
                                .cpu()
                                .tolist(),
                                "energy": pred_energy[i : i + 1].cpu().tolist(),
                                "stress": pred_stress[i : i + 1].cpu().tolist(),
                            },
                        }
                    )

            mask = atomic_numbers != 0
            natoms = mask.sum(dim=1)

            # Modified compute_loss to return total_loss and a dict of sub-losses
            val_loss_dict = compute_loss(
                pred_forces,
                pred_energy,
                pred_stress,
                true_forces,
                true_energy,
                true_stress,
                mask,
                device,
                natoms=natoms,
                use_mask=True,
                force_magnitude=False,
            )
            total_val_loss += val_loss_dict["total_loss"].item()

            # Log validation loss to TensorBoard
            tensorboard_log(
                loss_dict=val_loss_dict,
                train=False,
                writer=writer,
                epoch=epoch,
                tensorboard_prefix=tensorboard_prefix,
            )

    if num_val_batches == 0:
        return float("inf")
    return total_val_loss / num_val_batches, val_samples


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    pbar,
    device,
    patience=6,
    results_path=None,
    experiment_results=None,
    data_size_key=None,
    run_entry=None,
    writer=None,
    tensorboard_prefix="model",
):
    """
    Train model with validation at epoch 0 and every 10 epochs.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        optimizer: Optimizer
        scheduler: Learning rate scheduler (or None)
        pbar: A tqdm progress bar
        device: Torch device (cuda, cpu, etc.)
        patience: Early stopping patience
        results_path, experiment_results, data_size_key, run_entry: For JSON logging
        writer: TensorBoard SummaryWriter for logging
        tensorboard_prefix: Prefix for naming the logs in TensorBoard
    """
    model.to(device)
    can_write_partial = all(
        [results_path, experiment_results, data_size_key, run_entry]
    )
    losses = {}

    # Initial validation at epoch 0
    val_loss, val_samples = run_validation(
        model,
        val_loader,
        device,
        writer,
        epoch,
        tensorboard_prefix,
    )
    losses[0] = {"val_loss": float(val_loss)}
    if can_write_partial:
        partial_json_log(
            experiment_results,
            data_size_key,
            run_entry,
            0,
            float("nan"),
            val_loss,
            results_path,
            val_samples,
        )

    # Early stopping setup
    best_val_loss = val_loss
    epochs_since_improvement = 0
    last_val_loss = val_loss

    # Training loop starting from epoch 1
    for epoch in range(1, len(pbar) + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            true_forces = batch["forces"].to(device)
            true_energy = batch["energy"].to(device)
            true_stress = batch["stress"].to(device)

            optimizer.zero_grad()
            pred_forces, pred_energy, pred_stress = model(atomic_numbers, positions)

            mask = atomic_numbers != 0
            natoms = mask.sum(dim=1)

            train_loss_dict = compute_loss(
                pred_forces,
                pred_energy,
                pred_stress,
                true_forces,
                true_energy,
                true_stress,
                mask,
                device,
                natoms=natoms,
                use_mask=True,
                force_magnitude=False,
            )
            total_train_loss = train_loss_dict["total_loss"]
            total_train_loss.backward()
            optimizer.step()
            train_loss_sum += total_train_loss.item()

            # Update running average
            current_avg_loss = train_loss_sum / (batch_idx + 1)
            pbar.set_description(
                f"train_loss={current_avg_loss:.2f} val_loss={last_val_loss:.2f}"
            )

        if scheduler is not None:
            scheduler.step()

        # Average training loss for the epoch
        avg_epoch_train_loss = train_loss_sum / n_train_batches
        losses[epoch] = {"train_loss": float(avg_epoch_train_loss)}

        # Validation every N epochs
        val_loss_to_log = float("nan")
        if epoch % 1000 == 0:
            val_loss, val_samples = run_validation(
                model,
                val_loader,
                device,
                writer,
                epoch,
                tensorboard_prefix,
            )
            last_val_loss = val_loss
            losses[epoch]["val_loss"] = float(val_loss)
            val_loss_to_log = val_loss

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    return model, losses

        # JSON partial logging
        if can_write_partial:
            partial_json_log(
                experiment_results,
                data_size_key,
                run_entry,
                epoch,
                avg_epoch_train_loss,
                val_loss_to_log,
                results_path,
                val_samples if epoch % 10 == 0 else [],
            )

        # === TensorBoard logging (once per epoch) ===
        # Log train loss to TensorBoard
        tensorboard_log(
            loss_dict=train_loss_dict,
            train=True,
            writer=writer,
            epoch=epoch,
            tensorboard_prefix=tensorboard_prefix,
        )

        # Log layer norms for *all* parameters
        for name, param in model.named_parameters():
            writer.add_scalar(
                f"{tensorboard_prefix}/LayerNorm/{name}",
                param.data.norm().item(),
                global_step=epoch,
            )

        pbar.update(1)

    return model, losses
