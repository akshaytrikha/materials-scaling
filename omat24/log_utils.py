# External
import torch
import json
import math


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

            if loss_entry:  # Only add if there's at least one non-NaN value
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

        if loss_entry:  # Only add if there's at least one non-NaN value
            new_entry["losses"][str(step)] = loss_entry

        experiment_results[data_size_key].append(new_entry)

    with open(results_path, "w") as f:
        json.dump(experiment_results, f)


def collect_train_val_samples(
    model, train_loader, val_loader, device, num_visualization_samples
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
    model.eval()  # Set model to evaluation mode
    samples = {"train": [], "val": []}

    # Helper function to process a single batch
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

    # Process fixed samples
    with torch.no_grad():
        # Process training samples
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

        # Process validation samples
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


def tensorboard_log(loss_value, loss_type, train, writer, epoch, tensorboard_prefix):
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
    tag = f"{tensorboard_prefix}/{'train' if train else 'val'}_{loss_type}_loss"
    writer.add_scalar(tag, loss_value, global_step=epoch)
