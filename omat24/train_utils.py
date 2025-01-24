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
    model.to(device)
    model.eval()
    total_val_loss = 0.0
    num_val_batches = len(val_loader)

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
    patience=6,
    results_path=None,
    experiment_results=None,
    data_size_key=None,
    run_entry=None,
):
    """
    Train for a fixed number of epochs (default=50), with exactly 2 validations per epoch.
    We incorporate early stopping with a given `patience` (default=6).
    We also log training/validation losses:
      - mid-epoch (2 times per epoch),
      - end-of-epoch,
    and store them in a local 'losses' dict keyed by 'step'.
    """

    model.to(device)
    step = 0  # Tracks total training steps across epochs

    # References for partial JSON logging
    can_write_partial = (
        results_path is not None
        and experiment_results is not None
        and data_size_key is not None
        and run_entry is not None
    )

    # Collect local logs (like before)
    # losses[step] = {"train_loss": float, "val_loss": float}
    losses = {}

    # Early stopping bookkeeping
    best_val_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in pbar:
        model.train()
        train_loss_sum = 0.0
        n_train_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # ---------------------------
            # Forward + Backprop
            # ---------------------------
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
            step += 1

        # Optionally step the scheduler each epoch
        if scheduler is not None:
            scheduler.step()
    
        avg_epoch_train_loss = train_loss_sum / n_train_batches

        # ---------------------------
        # End-of-epoch Validation - every 5 epochs
        # ---------------------------

        if epoch % 5 == 0:
            avg_epoch_val_loss = run_validation(model, val_loader, device)
            losses[step] = {
                "train_loss": float(avg_epoch_train_loss),
                "val_loss": float(avg_epoch_val_loss),
            }
            if can_write_partial:
                partial_json_log(
                    experiment_results,
                    data_size_key,
                    run_entry,
                    step,
                    avg_epoch_train_loss,
                    avg_epoch_val_loss,
                    results_path,
                )

            # Early stopping check at epoch’s end
            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print("Early stopping triggered (end-of-epoch).")
                    return model, losses

    return model, losses
