# External
import torch
import json
import math

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


def run_validation(model, val_loader, device):
    """Compute and return the average validation loss."""
    model.to(device)
    model.eval()
    total_val_loss = 0.0
    num_val_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            # print(batch)
            atomic_numbers = batch["atomic_numbers"].to(device)
            # positions = batch["positions"].to(device)
            # factorized_distances = batch["factorized_matrix"].to(device)
            true_forces = batch["forces"].to(device)
            true_energy = batch["energy"].to(device)
            true_stress = batch["stress"].to(device)

            mask = atomic_numbers != 0

            pred_forces, pred_energy = model(
                
            )

            natoms = mask.sum(dim=1)
            val_loss = compute_loss(
                pred_forces=pred_forces,
                pred_energy=pred_energy,
                true_forces=true_forces,
                true_energy=true_energy,
                device=device,
                natoms=natoms,
            )
            total_val_loss += val_loss.item()

    if num_val_batches == 0:
        return float("inf")
    return total_val_loss / num_val_batches


def collect_train_val_samples(
    model, train_loader, val_loader, device, num_visualization_samples
):
    """
    Collect samples and predictions from both training and validation sets.
    Uses fixed indices for consistent visualization.
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
    num_visualization_samples=3,
):
    """Train model with validation at epoch 0 and every 10 epochs."""
    model.to(device)
    can_write_partial = all(
        [results_path, experiment_results, data_size_key, run_entry]
    )
    losses = {}

    # Initial validation at epoch 0
    # val_loss = run_validation(model, val_loader, device)
    val_loss = 100000000000000
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

    # Early stopping setup
    best_val_loss = val_loss
    epochs_since_improvement = 0
    last_val_loss = val_loss
    samples = None  # For visualization

    # Training loop starting from epoch 1
    for epoch in range(1, len(pbar) + 1):
        model.train()
        train_loss_sum = 0.0
        n_train_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            atomic_numbers = batch["atomic_numbers"].to(device)
            # positions = batch["positions"].to(device)
            # factorized_distances = batch["factorized_matrix"].to(device)
            true_forces = batch["forces"].to(device)
            true_energy = batch["energy"].to(device)
            true_stress = batch["stress"].to(device)

            mask = atomic_numbers != 0

            optimizer.zero_grad()
            # pred_forces, pred_energy, pred_stress = model(
            #     atomic_numbers, positions, factorized_distances, mask
            # )
            pred_forces, pred_energy = model(
                batch
            )

            # natoms = mask.sum(dim=1)
            natoms = len(atomic_numbers)
            train_loss = compute_loss(
                pred_forces=pred_forces,
                pred_energy=pred_energy,
                true_forces=true_forces,
                true_energy=true_energy,
                device=device,
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

        validate_every = 200
        visualize_every = 200

        # Run validation every 10 epochs
        if epoch % validate_every == 0:
            val_loss = run_validation(model, val_loader, device)
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

        if epoch % visualize_every == 0:
            samples = collect_train_val_samples(
                model,
                train_loader,
                val_loader,
                device,
                num_visualization_samples,
            )

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
