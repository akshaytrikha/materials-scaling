# train.py
# External
import torch
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
import train_utils as train_utils

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
            concatenated=False,
        )

    experiment_results = {}

    for data_fraction in args.data_fractions:
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
                    optimizer = train_utils.get_optimizer(model, learning_rate=lr)
                    scheduler = train_utils.get_scheduler(optimizer)

                    pbar = tqdm(range(args.epochs), desc="Training")
                    trained_model, losses = train_utils.train(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        pbar=pbar,
                        device=DEVICE,
                        val_interval=max(1, len(train_loader) // args.val_steps_target),
                    )

                    # Generate a unique model name and checkpoint path
                    model_name = f"model_ds{dataset_size}_p{int(model.num_params)}"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    # Save all results to JSON
    results_path = Path("results") / f"scaling_experiments_{timestamp}.json"
    Path("results").mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(experiment_results, f, indent=4)

    print(f"\nTraining completed. Results saved to {results_path}")
