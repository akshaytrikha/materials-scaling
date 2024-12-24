# External
import torch
import json

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
            }
            break

    if not found_existing:
        run_entry["losses"] = {
            str(step): {
                "train_loss": float(avg_train_loss),
                "val_loss": float(val_loss),
            }
        }
        experiment_results[data_size_key].append(run_entry)

    with open(results_path, "w") as f:
        json.dump(experiment_results, f, indent=4)


def run_validation(model, val_loader, device):
    """Compute and return the average validation loss."""
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            true_forces = batch["forces"].to(device)
            true_energy = batch["energy"].to(device)
            true_stress = batch["stress"].to(device)

            pred_forces, pred_energy, pred_stress = model(atomic_numbers, positions)

            mask = atomic_numbers != 0
            natoms = mask.sum(dim=1)
            val_loss = compute_loss(
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
            total_val_loss += val_loss.item()
            num_val_batches += 1

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
    device,
    val_interval,
    results_path,
    experiment_results,
    data_size_key,
    run_entry,
    total_val_steps=40,
    patience=6,
):
    """
    Train the model with incremental JSON logging after each validation step.

    Args:
        model (nn.Module): PyTorch model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Model optimizer.
        scheduler (_LRScheduler or None): Learning rate scheduler.
        pbar (tqdm): Progress bar for epochs.
        device (torch.device): CPU/CUDA device.
        val_interval (int): Validate every `val_interval` steps.
        total_val_steps (int): Max number of validations before stopping.
        patience (int): Validation patience for early stopping.
        results_path (str): File path to log JSON data incrementally.
        experiment_results (dict): Loaded from a JSON file or empty.
        data_size_key (str): Key for the current dataset size in `experiment_results`.
        run_entry (dict): Run metadata dict, typically includes "model_name", "config", "losses", etc.

    Returns:
        (model, losses_dict): Updated model and dictionary of losses indexed by step.
    """
    model.to(device)
    losses = {}
    step = 0
    val_steps_done = 0

    best_val_loss = float("inf")
    epochs_since_improvement = 0

    # We only log partial steps if we have everything we need
    can_write_partial = (
        results_path is not None
        and experiment_results is not None
        and data_size_key is not None
        and run_entry is not None
    )

    for epoch in pbar:
        model.train()
        train_loss_sum = 0.0
        num_train_batches = 0

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
            train_loss = compute_loss(
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
            train_loss.backward()
            optimizer.step()

            train_loss_sum += train_loss.item()
            num_train_batches += 1
            step += 1

            # --- Validation Step ---
            if step % val_interval == 0 and val_steps_done < total_val_steps:
                val_loss = run_validation(model, val_loader, device)
                avg_train_loss = train_loss_sum / num_train_batches

                # Store in the local 'losses' dictionary
                losses[step] = {
                    "train_loss": float(avg_train_loss),
                    "val_loss": float(val_loss),
                }

                # Write partial logs
                if can_write_partial:
                    partial_json_log(
                        experiment_results=experiment_results,
                        data_size_key=data_size_key,
                        run_entry=run_entry,
                        step=step,
                        avg_train_loss=avg_train_loss,
                        val_loss=val_loss,
                        results_path=results_path,
                    )

                # Early stopping logic
                if val_loss < best_val_loss - 0.1:
                    best_val_loss = val_loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= patience:
                        print("Early stopping triggered")
                        return model, losses

                val_steps_done += 1

            # If we've done enough validation steps, stop early
            if val_steps_done >= total_val_steps:
                print("Reached maximum number of validations.")
                return model, losses

        if scheduler is not None:
            scheduler.step()

    return model, losses
