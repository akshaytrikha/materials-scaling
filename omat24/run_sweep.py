import wandb
from train import main, get_args  # Import main and get_args directly from your train.py


def sweep_agent():
    """Function called by wandb agent for each sweep run"""
    # Initialize a wandb run - this creates the wandb.run object
    run = wandb.init()

    # Get default args structure first
    args = get_args()

    # Override args with values from wandb.config
    args.architecture = wandb.config.architecture
    args.batch_size = [
        wandb.config.batch_size
    ]  # Note: batch_size is a list in your code
    args.lr = [wandb.config.learning_rate]  # Note: lr is a list in your code
    args.epochs = wandb.config.epochs
    args.datasets = [wandb.config.datasets]  # Assuming datasets is a list
    args.split_name = wandb.config.split_name
    args.data_fractions = [
        wandb.config.data_fractions
    ]  # Assuming data_fractions is a list
    args.val_data_fraction = wandb.config.val_data_fraction
    args.vis_every = wandb.config.vis_every
    args.val_every = wandb.config.val_every
    args.train_workers = wandb.config.train_workers
    args.val_workers = wandb.config.val_workers
    args.mixed_precision = wandb.config.mixed_precision
    args.no_log = wandb.config.no_log
    args.wandb = True  # Explicitly enable wandb logging

    # Set a custom name for this run
    run.name = f"{args.architecture}_bs={args.batch_size[0]}_lr={args.lr[0]}"

    # Print confirmation message
    print(f"Starting direct integration run: {run.name} (ID: {run.id})")
    print(
        f"Arguments: architecture={args.architecture}, batch_size={args.batch_size}, lr={args.lr}"
    )

    # Call main function directly with configured args
    main(args=args)

    # Print confirmation when run completes
    print(f"Completed run: {run.name} (ID: {run.id})")


if __name__ == "__main__":
    # Login to wandb (first time only)
    wandb.login()

    # Define sweep configuration
    sweep_config = {
        "method": "grid",
        "name": "ADiT-df=0.001",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"values": [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]},
            "batch_size": {"values": [8, 16, 32, 64, 128]},
            "architecture": {"value": "ADiT"},
            "epochs": {"value": 30},
            "dataset": {"value": "all"},  # wandb needs a "dataset" key
            "datasets": {"value": "all"},
            "split_name": {"value": "val"},
            "data_fractions": {"value": 0.001},
            "val_data_fraction": {"value": 0.01},
            "vis_every": {"value": 1000},
            "val_every": {"value": 5},
            "train_workers": {"value": 0},
            "val_workers": {"value": 0},
            "mixed_precision": {"value": True},
            "no_log": {"value": True},
        },
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="omat24")

    # Start the sweep agent
    wandb.agent(sweep_id, function=sweep_agent, count=None)  # Run all combinations
