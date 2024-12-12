# External
import torch
import torch.optim as optim
from pathlib import Path
import pprint
import json
from datetime import datetime
from tqdm import tqdm

# Internal
from data import OMat24Dataset, get_dataloaders
from data_utils import download_dataset
from arg_parser import get_args
from models.fcn import MetaFCNModels
from models.transformer_models import MetaTransformerModels
from train_utils import train, EPOCHS_SCHEDULE

# Set seed & device
seed = 1024
torch.manual_seed(seed)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def save_results_to_file(results_path, experiment_results):
    """Save experiment results to the JSON file."""
    with open(results_path, "w") as f:
        json.dump(experiment_results, f, indent=4)


if __name__ == "__main__":
    args = get_args()

    # Load dataset
    split_name = "val"
    dataset_name = "rattled-300-subsampled"

    dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
    if not dataset_path.exists():
        download_dataset(dataset_name, split_name)
    dataset = OMat24Dataset(dataset_path=dataset_path, augment=args.augment)

    # User Hyperparam Feedback
    params = vars(args) | {
        "dataset_name": f"{split_name}/{dataset_name}",
        "max_n_atoms": dataset.max_n_atoms,
    }
    pprint.pprint(params)
    print()

    # Initialize meta model class
    if args.architecture == "FCN":
        meta_models = MetaFCNModels(vocab_size=args.n_elements)
    elif args.architecture == "Transformer":
        meta_models = MetaTransformerModels(
            vocab_size=args.n_elements,
            max_seq_len=dataset.max_n_atoms,
            concatenated=True,
        )

    experiment_results = {}

    # Create results path and initialize file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path("results") / f"experiments_{timestamp}.json"
    Path("results").mkdir(exist_ok=True)
    if not results_path.exists():
        with open(results_path, "w") as f:
            json.dump({}, f)  # Initialize as empty JSON

    print(f"\nTraining starting. Results continuously saved to {results_path}")

    for data_fraction in args.data_fractions:
        print(f"\nData fraction: {data_fraction}")
        for model_idx, model in enumerate(meta_models):
            print(
                f"\nModel {model_idx + 1}/{len(meta_models)} is on device {DEVICE} and has {model.num_params} parameters"
            )
            for batch_size in args.batch_sizes:
                train_loader, val_loader = get_dataloaders(
                    dataset,
                    data_fraction=data_fraction,
                    batch_size=batch_size,
                    batch_padded=False,
                )
                dataset_size = len(train_loader.dataset)

                for lr in args.lrs:
                    # Initialize optimizer and scheduler
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    num_epochs = int(args.epochs * EPOCHS_SCHEDULE[data_fraction])
                    total_steps = num_epochs * len(train_loader)
                    val_interval = max(
                        1, total_steps // 40
                    )  # Ensure at least every step

                    pbar = tqdm(range(num_epochs), desc="Training")
                    trained_model, losses = train(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        scheduler=None,
                        pbar=pbar,
                        device=DEVICE,
                        val_interval=val_interval,
                        total_val_steps=40,
                    )

                    # Generate a unique model name and checkpoint path
                    model_name = f"model_ds{dataset_size}_p{int(model.num_params)}"
                    checkpoint_path = f"checkpoints/{args.architecture}_ds{dataset_size}_p{int(model.num_params)}_{timestamp}.pth"
                    Path("checkpoints").mkdir(exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": trained_model.state_dict(),
                            "losses": losses,
                            "batch_size": batch_size,
                            "lr": lr,
                        },
                        checkpoint_path,
                    )

                    # Prepare run entry
                    run_entry = {
                        "model_name": model_name,
                        "config": {
                            "architecture": args.architecture,
                            "embedding_dim": getattr(model, "embedding_dim", None),
                            "depth": getattr(model, "depth", None),
                            "num_params": model.num_params,
                            "dataset_size": float(dataset_size),
                        },
                        "losses": {str(k): v for k, v in losses.items()},
                        "checkpoint_path": checkpoint_path,
                    }

                    ds_key = str(dataset_size)
                    if ds_key not in experiment_results:
                        experiment_results[ds_key] = []
                    experiment_results[ds_key].append(run_entry)

                    # Write results to JSON after each run entry
                    save_results_to_file(results_path, experiment_results)

    print(f"\nTraining completed. Results continuously saved to {results_path}")
