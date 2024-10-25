import argparse


def float_or_int(value):
    """Helper function to convert a string to a float or int.

    Useful for parsing --data_fractions as 0.01 0.1 1 or 0.01, 0.1, 1.0"""
    try:
        if "." in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}")


def get_args():
    parser = argparse.ArgumentParser(description="Training script for the model.")
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["FCN", "VanillaTransformer"],
        default="FCN",
        help='Model architecture to use: "FCN" or "VanillaTransformer"',
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of  epochs for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--dataset_version",
        type=str,
        choices=["small", "large"],
        default="small",
        help='Dataset size to use: "small" or "big"',
    )
    parser.add_argument(
        "--seq_max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--data_fractions",
        type=float_or_int,
        nargs="+",
        default=[0.01, 0.1],
        help="List of data fractions to use for training",
    )
    parser.add_argument(
        "--wandb_log", action="store_true", help="Enable Weights and Biases logging"
    )
    parser.add_argument(
        "--checkpoint_group_name",
        type=str,
        default=None,
        help="Name of the checkpoint group",
    )
    parser.add_argument("--ddp", action="store_true", help="Enable DDP training")
    return parser.parse_args()
