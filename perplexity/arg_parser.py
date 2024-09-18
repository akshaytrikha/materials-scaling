import argparse


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
        "--num_epochs", type=int, default=5, help="Number of epochs for training"
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
        "--data_fraction",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
        help="List of data fractions to use for training",
    )
    parser.add_argument(
        "--wandb_log", action="store_true", help="Enable Weights and Biases logging"
    )
    return parser.parse_args()
