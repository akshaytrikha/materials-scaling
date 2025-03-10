import argparse


def check_val_data_fraction(value):
    fvalue = float(value)
    if fvalue == 1.0:
        raise argparse.ArgumentTypeError("val_data_fraction cannot be 1")
    return fvalue


def get_args():
    parser = argparse.ArgumentParser(description="Training script for the model.")
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["FCN", "Transformer", "SchNet", "EquiformerV2"],
        default="FCN",
        help="Model architecture to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[64],
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=[0.001],
        help="Learning rate",
    )
    parser.add_argument(
        "--data_fractions",
        type=float,
        nargs="+",
        default=[1.0],
        help="Fractions of data for training (applied to the remainder after validation split)",
    )
    parser.add_argument(
        "--val_data_fraction",
        type=check_val_data_fraction,
        default=0.1,
        help="Fraction of the dataset to use for validation",
    )
    parser.add_argument(
        "--no_log", action="store_true", default=False, help="Enable logging"
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
    parser.add_argument(
        "--factorize",
        action="store_true",
        help="Factorize and use inverse distance matrix",
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1,
        help="Maximum norm for gradient clipping",
    )
    parser.add_argument(
        "--num_visualization_samples",
        type=int,
        default=3,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--train_workers",
        type=int,
        default=0,
        help="Number of workers for training",
    )
    parser.add_argument(
        "--val_workers",
        type=int,
        default=0,
        help="Number of workers for validation",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=500,
        help="Number of epochs to run in between validations",
    )
    parser.add_argument(
        "--vis_every",
        type=int,
        default=500,
        help="Number of epochs to run in between visualizations",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="val",
        help="OMat24 split to use",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["rattled-300-subsampled"],
        help="Dataset(s) to use",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training" "--datasets_base_path",
    )
    parser.add_argument(
        "--datasets_base_path",
        type=str,
        default="./datasets",
        help="Base path for dataset storage",
    )
    parser.add_argument(
        "--mixed_precision", 
        action="store_true",
        help="Use mixed precision training with FP16"
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="None"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
