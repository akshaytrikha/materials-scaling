import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Training script for the model.")
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["FCN", "Transformer"],
        default="FCN",
        help="Model architecture to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=[0.0001],
        help="Learning rate",
    )
    parser.add_argument(
        "--data_fractions",
        type=float,
        nargs="+",
        default=[1.0],
        help="Fractions of data",
    )
    parser.add_argument(
        "--no_log", action="store_true", default=False, help="Enable logging"
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
        "--n_elements",
        type=int,
        default=119,
        help="Maximum number of unique elements in a sample",
    )
    parser.add_argument("--augment", action="store_true", help="Rotation augmentation")

    parser.add_argument("--factorize", action="store_true", help="Factorize and use inverse distance matrix")

    return parser.parse_args()
