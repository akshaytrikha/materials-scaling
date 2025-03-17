import os
import sys
import wandb
import subprocess
from sweep_config import sweep_config


def sweep_agent():
    """Function called by wandb agent for each sweep run"""
    # Initialize a wandb run
    run = wandb.init()

    # Extract the parameters for this run
    lr = wandb.config.learning_rate
    bs = wandb.config.batch_size
    arch = wandb.config.architecture
    epochs = wandb.config.epochs
    datasets = wandb.config.datasets
    split_name = wandb.config.split_name
    data_fractions = wandb.config.data_fractions
    val_data_fraction = wandb.config.val_data_fraction
    vis_every = wandb.config.vis_every
    val_every = wandb.config.val_every
    train_workers = wandb.config.train_workers
    val_workers = wandb.config.val_workers
    mixed_precision = wandb.config.mixed_precision
    no_log = wandb.config.no_log

    # Build the command to run your training script with these parameters
    cmd = [
        "python",
        "train.py",
        "--architecture",
        arch,
        "--batch_size",
        str(bs),
        "--lr",
        str(lr),
        "--epochs",
        str(epochs),
        "--datasets",
        datasets,
        "--split_name",
        split_name,
        "--data_fractions",
        str(data_fractions),
        "--val_data_fraction",
        str(val_data_fraction),
        "--vis_every",
        str(vis_every),
        "--val_every",
        str(val_every),
        "--train_workers",
        str(train_workers),
        "--val_workers",
        str(val_workers),
        "--wandb",  # Flag to enable wandb logging
    ]

    if mixed_precision:
        cmd.append("--mixed_precision")

    if no_log:
        cmd.append("--no_log")

    # Set the WANDB_RUN_ID environment variable to ensure the subprocess
    # connects to the same run
    env = os.environ.copy()
    env["WANDB_RUN_ID"] = wandb.run.id

    # Run the training script as a subprocess
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )

    # Stream the output to the console
    for line in process.stdout:
        print(line, end="")

    # Wait for the process to complete
    process.wait()

    # End the wandb run
    wandb.finish()


if __name__ == "__main__":
    # Login to wandb (first time only)
    wandb.login()

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="omat-training")

    # Start the sweep agent
    wandb.agent(sweep_id, function=sweep_agent, count=None)  # Run all combinations
