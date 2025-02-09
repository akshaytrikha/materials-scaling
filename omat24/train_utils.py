# External
import torch
import json
import math
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
    samples=None,
):
    """
    Append train_loss and val_loss for the given step to the specified run_entry in experiment_results,
    then write the updated experiment_results dictionary to disk.

    Structure:
    {
      "dataset_size": [{
        "model_name": str,
        "config": dict,
        "samples": {
          "train": [{idx, symbols, atomic_numbers, positions}],
          "val": [{idx, symbols, atomic_numbers, positions}]
        },
        "losses": {
          "step": {
            train_loss: float,
            val_loss: float,
            pred: {
              train: [{forces, energy, stress}],
              val: [{forces, energy, stress}]
            }
          }
        }
      }]
    }

    Args:
        experiment_results (dict): The dictionary holding all results for this experiment.
        data_size_key (str): A string key referring to the dataset size or category.
        run_entry (dict): Dictionary containing model_name, config, etc. identifying this run.
        step (int or str): The training step or epoch number being logged.
        avg_train_loss (float): The average training loss at this step.
        val_loss (float): The validation loss at this step.
        results_path (str): Path to the JSON file where results are saved.
        samples (dict, optional): Dictionary of training and validation samples/predictions.

    Returns:
        None
    """
    if data_size_key not in experiment_results:
        experiment_results[data_size_key] = []

    # Check if this run_entry is already in the list for data_size_key
    found_existing = False
    for existing_run in experiment_results[data_size_key]:
        if existing_run.get("model_name", "") == run_entry["model_name"]:
            found_existing = True

            # Initialize samples dict if first time seeing samples
            if samples and "samples" not in existing_run:
                existing_run["samples"] = {"train": [], "val": []}
                for split in ["train", "val"]:
                    for sample in samples[split]:
                        existing_run["samples"][split].append(
                            {
                                "idx": sample["idx"],
                                "symbols": sample["symbols"],
                                "atomic_numbers": sample["atomic_numbers"],
                                "positions": sample["positions"],
                                "forces": sample["true"]["forces"],
                                "energy": sample["true"]["energy"],
                                "stress": sample["true"]["stress"],
                            }
                        )

            # Add (or extend) loss entry
            if "losses" not in existing_run:
                existing_run["losses"] = {}

            loss_entry = {}
            if not math.isnan(avg_train_loss):
                loss_entry["train_loss"] = float(avg_train_loss)
            if not math.isnan(val_loss):
                loss_entry["val_loss"] = float(val_loss)

            # Add predictions if samples exist
            if samples:
                loss_entry["pred"] = {
                    "train": [
                        {
                            "forces": s["pred"]["forces"],
                            "energy": s["pred"]["energy"],
                            "stress": s["pred"]["stress"],
                        }
                        for s in samples["train"]
                    ],
                    "val": [
                        {
                            "forces": s["pred"]["forces"],
                            "energy": s["pred"]["energy"],
                            "stress": s["pred"]["stress"],
                        }
                        for s in samples["val"]
                    ],
                }

            if loss_entry:  # Only add if there's at least one valid value
                existing_run["losses"][str(step)] = loss_entry
            break

    if not found_existing:
        # Initialize new run entry
        new_entry = {
            "model_name": run_entry["model_name"],
            "config": run_entry["config"],
            "samples": {"train": [], "val": []},
            "losses": {},
        }

        # Add initial samples if they exist
        if samples:
            for split in ["train", "val"]:
                for sample in samples[split]:
                    new_entry["samples"][split].append(
                        {
                            "idx": sample["idx"],
                            "symbols": sample["symbols"],
                            "atomic_numbers": sample["atomic_numbers"],
                            "positions": sample["positions"],
                            "forces": sample["true"]["forces"],
                            "energy": sample["true"]["energy"],
                            "stress": sample["true"]["stress"],
                        }
                    )

        # Add initial loss entry
        loss_entry = {}
        if not math.isnan(avg_train_loss):
            loss_entry["train_loss"] = float(avg_train_loss)
        if not math.isnan(val_loss):
            loss_entry["val_loss"] = float(val_loss)

        # Add predictions if samples exist
        if samples:
            loss_entry["pred"] = {
                "train": [
                    {
                        "forces": s["pred"]["forces"],
                        "energy": s["pred"]["energy"],
                        "stress": s["pred"]["stress"],
                    }
                    for s in samples["train"]
                ],
                "val": [
                    {
                        "forces": s["pred"]["forces"],
                        "energy": s["pred"]["energy"],
                        "stress": s["pred"]["stress"],
                    }
                    for s in samples["val"]
                ],
            }

        if loss_entry:
            new_entry["losses"][str(step)] = loss_entry

        experiment_results[data_size_key].append(new_entry)

    with open(results_path, "w") as f:
        json.dump(experiment_results, f)


def tensorboard_log(loss_value, train, writer, epoch, tensorboard_prefix):
    """
    Log a loss value to TensorBoard.

    Args:
        loss_value (float): The loss value to log.
        train (bool): Whether this is training (True) or validation (False) loss.
        writer (SummaryWriter): TensorBoard writer object.
        epoch (int): Current training epoch.
        tensorboard_prefix (str): Prefix for naming the logs.

    Returns:
        None
    """
    if writer is None:
        return
    tag = f"{tensorboard_prefix}/{'train' if train else 'val'}_loss"
    writer.add_scalar(tag, loss_value, global_step=epoch)


def run_validation(model, val_loader, device):
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

    with torch.no_grad():
        for batch in val_loader:
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            factorized_distances = batch["factorized_matrix"].to(device)
            true_forces = batch["forces"].to(device)
            true_energy = batch["energy"].to(device)
            true_stress = batch["stress"].to(device)

            mask = atomic_numbers != 0

            pred_forces, pred_energy, pred_stress = model(
                atomic_numbers, positions, factorized_distances, mask
            )

            natoms = mask.sum(dim=1)
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
            )
            total_val_loss += val_loss_dict["total_loss"].item()

    if num_val_batches == 0:
        return float("inf")
    return total_val_loss / num_val_batches


def collect_train_val_samples(
    model, train_loader, val_loader, device, num_visualization_samples
):
    """
    Collect samples and predictions from both training and validation sets.
    Uses fixed indices for consistent visualization.

    Args:
        model (nn.Module): The PyTorch model to use for inference.
        train_loader (DataLoader): DataLoader for training samples.
        val_loader (DataLoader): DataLoader for validation samples.
        device (torch.device): The device to run inference on.
        num_visualization_samples (int): Number of samples to collect for visualization.

    Returns:
        dict: A dictionary containing 'train' and 'val' keys, each mapping to a list of sample dicts.
    """
    model.eval()
    samples = {"train": [], "val": []}

    def process_batch(batch, device):
        idx = batch["idx"]
        symbols = batch["symbols"]
        atomic_numbers = batch["atomic_numbers"].to(device)
        positions = batch["positions"].to(device)
        factorized_distances = batch["factorized_matrix"].to(device)
        true_forces = batch["forces"].to(device)
        true_energy = batch["energy"].to(device)
        true_stress = batch["stress"].to(device)

        mask = atomic_numbers != 0
        pred_forces, pred_energy, pred_stress = model(
            atomic_numbers, positions, factorized_distances, mask
        )
        return (
            idx,
            symbols,
            atomic_numbers,
            positions,
            true_forces,
            true_energy,
            true_stress,
            pred_forces,
            pred_energy,
            pred_stress,
        )

    # Get the underlying dataset from the DataLoader
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    with torch.no_grad():
        # Collect training samples
        for i in range(min(num_visualization_samples, len(train_dataset))):
            batch = {
                k: torch.unsqueeze(v, 0) if isinstance(v, torch.Tensor) else v
                for k, v in train_dataset[i].items()
            }
            (
                idx,
                symbols,
                atomic_numbers,
                positions,
                true_forces,
                true_energy,
                true_stress,
                pred_forces,
                pred_energy,
                pred_stress,
            ) = process_batch(batch, device)

            sample_length = (atomic_numbers != 0).sum(dim=1)[0].item()
            samples["train"].append(
                {
                    "idx": idx,
                    "symbols": symbols,
                    "atomic_numbers": atomic_numbers[:, :sample_length]
                    .cpu()
                    .tolist()[0],
                    "positions": positions[:, :sample_length].cpu().tolist()[0],
                    "true": {
                        "forces": true_forces[:, :sample_length].cpu().tolist()[0],
                        "energy": true_energy.cpu().tolist()[0],
                        "stress": true_stress.cpu().tolist()[0],
                    },
                    "pred": {
                        "forces": pred_forces[:, :sample_length].cpu().tolist()[0],
                        "energy": pred_energy.cpu().tolist()[0],
                        "stress": pred_stress.cpu().tolist()[0],
                    },
                }
            )

        # Collect validation samples
        for i in range(min(num_visualization_samples, len(val_dataset))):
            batch = {
                k: torch.unsqueeze(v, 0) if isinstance(v, torch.Tensor) else v
                for k, v in val_dataset[i].items()
            }
            (
                idx,
                symbols,
                atomic_numbers,
                positions,
                true_forces,
                true_energy,
                true_stress,
                pred_forces,
                pred_energy,
                pred_stress,
            ) = process_batch(batch, device)

            sample_length = (atomic_numbers != 0).sum(dim=1)[0].item()
            samples["val"].append(
                {
                    "idx": idx,
                    "symbols": symbols,
                    "atomic_numbers": atomic_numbers[:, :sample_length]
                    .cpu()
                    .tolist()[0],
                    "positions": positions[:, :sample_length].cpu().tolist()[0],
                    "true": {
                        "forces": true_forces[:, :sample_length].cpu().tolist()[0],
                        "energy": true_energy.cpu().tolist()[0],
                        "stress": true_stress.cpu().tolist()[0],
                    },
                    "pred": {
                        "forces": pred_forces[:, :sample_length].cpu().tolist()[0],
                        "energy": pred_energy.cpu().tolist()[0],
                        "stress": pred_stress.cpu().tolist()[0],
                    },
                }
            )

    return samples


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    pbar,
    device,
    patience=50,
    results_path=None,
    experiment_results=None,
    data_size_key=None,
    run_entry=None,
    writer=None,
    tensorboard_prefix="model",
    num_visualization_samples=3,
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
    can_write_partial = all([results_path, experiment_results, data_size_key, run_entry])
    losses = {}

    # Initial validation at epoch 0
    val_loss = run_validation(model, val_loader, device)
    losses[0] = {"val_loss": float(val_loss)}
    if writer is not None:
        tensorboard_log(val_loss, train=False, writer=writer, epoch=0, tensorboard_prefix=tensorboard_prefix)

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
    validate_every = 1000
    visualize_every = 500

    for epoch in range(1, len(pbar) + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            factorized_distances = batch["factorized_matrix"].to(device)
            true_forces = batch["forces"].to(device)
            true_energy = batch["energy"].to(device)
            true_stress = batch["stress"].to(device)

            mask = atomic_numbers != 0

            optimizer.zero_grad()
            pred_forces, pred_energy, pred_stress = model(
                atomic_numbers, positions, factorized_distances, mask
            )

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
            )
            total_train_loss = train_loss_dict["total_loss"]
            total_train_loss.backward()
            optimizer.step()

            train_loss_sum += total_train_loss.item()
            current_avg_loss = train_loss_sum / (batch_idx + 1)

            pbar.set_description(
                f"train_loss={current_avg_loss:.2f} val_loss={last_val_loss:.2f}"
            )

        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()

        avg_epoch_train_loss = train_loss_sum / n_train_batches
        losses[epoch] = {"train_loss": float(avg_epoch_train_loss)}

        # TensorBoard logging for training loss
        if writer is not None:
            tensorboard_log(avg_epoch_train_loss, train=True, writer=writer, epoch=epoch, tensorboard_prefix=tensorboard_prefix)
            # Log parameter norms (example usage)
            for name, param in model.named_parameters():
                if param is not None and param.requires_grad:
                    writer.add_scalar(
                        f"{tensorboard_prefix}/LayerNorm/{name}",
                        param.data.norm().item(),
                        global_step=epoch,
                    )

        # Validate every 'validate_every' epochs
        if epoch % validate_every == 0:
            val_loss = run_validation(model, val_loader, device)
            last_val_loss = val_loss
            losses[epoch]["val_loss"] = float(val_loss)

            # Also log validation loss to TensorBoard
            if writer is not None:
                tensorboard_log(val_loss, train=False, writer=writer, epoch=epoch, tensorboard_prefix=tensorboard_prefix)

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
