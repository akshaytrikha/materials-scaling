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
    dataset = wandb.config.dataset

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
        dataset,
        "--data_fractions",
        "1.0",  # Use full dataset
        "--wandb",  # Flag to enable wandb logging
    ]

    # Run the training script as a subprocess
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
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
