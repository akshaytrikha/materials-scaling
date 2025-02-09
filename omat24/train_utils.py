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
    """
    if data_size_key not in experiment_results:
        experiment_results[data_size_key] = []

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

            # Add loss entry
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

            if loss_entry:  # Only add if there's something valid
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

        loss_entry = {}
        if not math.isnan(avg_train_loss):
            loss_entry["train_loss"] = float(avg_train_loss)
        if not math.isnan(val_loss):
            loss_entry["val_loss"] = float(val_loss)

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


def tensorboard_log(
    loss_value, train: bool, writer: SummaryWriter, epoch: int, tensorboard_prefix: str
):
    """
    Log a single loss value to TensorBoard.
    """
    if writer is None:
        return

    if train:
        tag = f"{tensorboard_prefix}/train_loss"
    else:
        tag = f"{tensorboard_prefix}/val_loss"

    writer.add_scalar(tag, loss_value, global_step=epoch)


def run_validation(
    model,
    val_loader,
    device,
    writer=None,
    epoch=0,
    tensorboard_prefix="model",
):
    """
    Compute and return the average validation loss, optionally logging to TensorBoard.
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
            )
            total_val_loss += val_loss.item()

    if num_val_batches == 0:
        average_val_loss = float("inf")
    else:
        average_val_loss = total_val_loss / num_val_batches

    # Log to TensorBoard
    tensorboard_log(
        loss_value=average_val_loss,
        train=False,
        writer=writer,
        epoch=epoch,
        tensorboard_prefix=tensorboard_prefix,
    )

    return average_val_loss


def collect_train_val_samples(
    model, train_loader, val_loader, device, num_visualization_samples
):
    """
    Collect samples and predictions from both training and validation sets.
    Uses fixed indices for consistent visualization.
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

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    with torch.no_grad():
        # Training samples
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

        # Validation samples
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
    Train the model, log progress to JSON (optional) and TensorBoard (optional),
    with validation at epoch 0 and then periodically.
    """
    model.to(device)
    can_write_partial = all([results_path, experiment_results, data_size_key, run_entry])
    losses = {}

    # Initial validation at epoch 0
    val_loss = run_validation(
        model,
        val_loader,
        device,
        writer=writer,
        epoch=0,
        tensorboard_prefix=tensorboard_prefix,
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
        )

    best_val_loss = val_loss
    epochs_since_improvement = 0
    last_val_loss = val_loss
    samples = None

    # Training loop
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
            )
            train_loss.backward()
            optimizer.step()

            train_loss_sum += train_loss.item()
            current_avg_loss = train_loss_sum / (batch_idx + 1)
            pbar.set_description(
                f"train_loss={current_avg_loss:.2f} val_loss={last_val_loss:.2f}"
            )

        if scheduler is not None:
            scheduler.step()

        avg_epoch_train_loss = train_loss_sum / n_train_batches
        losses[epoch] = {"train_loss": float(avg_epoch_train_loss)}

        # === TensorBoard logging for training loss (epoch-level) ===
        tensorboard_log(
            loss_value=avg_epoch_train_loss,
            train=True,
            writer=writer,
            epoch=epoch,
            tensorboard_prefix=tensorboard_prefix,
        )

        # Log parameter norms if requested
        if writer is not None:
            for name, param in model.named_parameters():
                if param is not None and param.requires_grad:
                    writer.add_scalar(
                        f"{tensorboard_prefix}/LayerNorm/{name}",
                        param.data.norm().item(),
                        global_step=epoch,
                    )

        validate_every = 1000
        visualize_every = 500

        # Validation check
        if epoch % validate_every == 0:
            val_loss = run_validation(
                model,
                val_loader,
                device,
                writer=writer,
                epoch=epoch,
                tensorboard_prefix=tensorboard_prefix,
            )
            last_val_loss = val_loss
            losses[epoch]["val_loss"] = float(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
     
