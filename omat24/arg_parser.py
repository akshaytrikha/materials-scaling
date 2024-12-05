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
        choices=["FCN", "Transformer"],
        default="FCN",
        help='Model architecture to use: "FCN", "Transformer", or "TransformerConcatenated"',
    )
    parser.add_argument(
        "--batch_sizes", type=int, nargs="+", default=[16, 32, 64], help="Batch sizes for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs for training"
    )
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.1, 0.01, 0.001, 0.0001], help="Learning rates")
    # parser.add_argument(
    #     "--dataset_version",
    #     type=str,
    #     choices=["small", "large"],
    #     default="small",
    #     help='Dataset size to use: "small" or "big"',
    # )
    parser.add_argument(
        "--data_fraction", type=float, default=1.0, help="Fraction of data to use"
    )
    # parser.add_argument(
    #     "--data_fractions",
    #     type=float_or_int,
    #     nargs="+",
    #     default=[0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
    #     help="List of data fractions to use for training",
    # )
    parser.add_argument(
        "--wandb_log", action="store_true", help="Enable Weights and Biases logging"
    )
    parser.add_argument(
        "--concatenated", action="store_true", help="Enable concatenation"
    )
    parser.add_argument(
        "--checkpoint_group_name",
        type=str,
        default=None,
        help="Name of the checkpoint group",
    )
    parser.add_argument(
        "--max_n_atoms",
        type=int,
        default=300,
        help="Maximum number of atoms in a sample",
    )
    parser.add_argument(
        "--n_elements",
        type=int,
        default=119,
        help="Maximum number of unique elements in a sample",
    )
    parser.add_argument("--augment", action="store_true", help="Rotation augmentation")

    return parser.parse_args()
