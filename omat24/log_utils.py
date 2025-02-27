# External
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
    flops=0
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
            flops: float
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
            loss_entry["flops"] = flops

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
        loss_entry["flops"] = flops

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


def log_tb_metrics(
    metrics: dict, writer, epoch: int, tensorboard_prefix: str, train: bool
):
    """
    Log multiple metrics to TensorBoard.

    Args:
        metrics (dict): Dictionary where keys are loss types and values are the loss values.
        writer (SummaryWriter): TensorBoard writer object.
        epoch (int): Current training epoch.
        tensorboard_prefix (str): Prefix for naming the logs.
        train (bool): True for training metrics, False for validation metrics.
    """
    if writer is None:
        return
    for loss_type, loss_value in metrics.items():
        tag = f"{tensorboard_prefix}/{'train' if train else 'val'}_{loss_type}_loss"
        writer.add_scalar(tag, loss_value, global_step=epoch)
