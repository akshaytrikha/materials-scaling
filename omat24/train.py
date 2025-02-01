# External
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import pprint
import json
import math
from datetime import datetime
from tqdm import tqdm
import subprocess

# Internal
from data import OMat24Dataset, get_dataloaders
from data_utils import download_dataset
from arg_parser import get_args
from models.fcn import MetaFCNModels
from models.transformer_models import MetaTransformerModels
from train_utils import train, partial_json_log, run_validation

# Set seed & device
SEED = 1024
torch.manual_seed(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

if __name__ == "__main__":
    args = get_args()
    log = not args.no_log

    # Load dataset
    # split_name = "val"
    # dataset_name = "rattled-300-subsampled"
    split_name = "vc_dim"
    dataset_name = "2_atoms_1_samples"

    dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
    # if not dataset_path.exists():
    #     download_dataset(dataset_name, split_name)
    dataset = OMat24Dataset(dataset_path=dataset_path, augment=args.augment)

    # User Hyperparam Feedback
    params = vars(args) | {
        "dataset_name": f"{split_name}/{dataset_name}",
        "max_n_atoms": dataset.max_n_atoms,
    }
    pprint.pprint(params)
    print()

    batch_size = args.batch_size[0]
    lr = args.lr[0]
    use_factorize = args.factorize
    num_epochs = args.epochs

    # Initialize meta model class based on architecture choice
    if args.architecture == "FCN":
        meta_models = MetaFCNModels(
            vocab_size=args.n_elements, use_factorized=use_factorize
        )
    elif args.architecture == "Transformer":
        meta_models = MetaTransformerModels(
            vocab_size=args.n_elements,
            max_seq_len=dataset.max_n_atoms,
            concatenated=True,
            use_factorized=use_factorize,
        )

    experiment_results = {}

    # Create results path and initialize file if logging is enabled
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path("results") / f"experiments_{timestamp}.json"
    if log:
        Path("results").mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({}, f)  # Initialize as empty JSON
        print(f"\nLogging enabled. Results will be saved to {results_path}")
    else:
        print("\nLogging disabled. No experiment log will be saved.")

    for data_fraction in args.data_fractions:
        print(f"\nData fraction: {data_fraction}")
        for model_idx, model in enumerate(meta_models):
            print(
                f"\nModel {model_idx + 1}/{len(meta_models)} is on device {DEVICE} and has {model.num_params} parameters"
            )

            train_loader, val_loader = get_dataloaders(
                dataset,
                train_data_fraction=data_fraction,
                batch_size=batch_size,
                seed=SEED,
                batch_padded=False,
            )
            dataset_size = len(train_loader.dataset)
            optimizer = optim.AdamW(model.parameters(), lr=lr)

            lambda_schedule = lambda epoch: 0.5 * (
                1 + math.cos(math.pi * epoch / num_epochs)
            )
            scheduler = LambdaLR(optimizer, lr_lambda=lambda_schedule)

            # Prepare run entry etc.
            model_name = f"model_ds{dataset_size}_p{int(model.num_params)}"
            checkpoint_path = f"checkpoints/{args.architecture}_ds{dataset_size}_p{int(model.num_params)}_{timestamp}.pth"
            run_entry = {
                "model_name": model_name,
                "config": {
                    "architecture": args.architecture,
                    "embedding_dim": getattr(model, "embedding_dim", None),
                    "depth": getattr(model, "depth", None),
                    "num_params": model.num_params,
                    "dataset_size": dataset_size,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "seed": SEED,
                },
                "losses": {},
                "checkpoint_path": checkpoint_path,
            }
            ds_key = str(dataset_size)
            if ds_key not in experiment_results:
                experiment_results[ds_key] = []
            experiment_results[ds_key].append(run_entry)
            if log:
                with open(results_path, "w") as f:
                    json.dump(experiment_results, f, indent=4)

            pbar = tqdm(range(num_epochs + 1))
            trained_model, losses = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                pbar=pbar,
                device=DEVICE,
                patience=6,
                results_path=results_path if log else None,
                experiment_results=experiment_results if log else None,
                data_size_key=ds_key if log else None,
                run_entry=run_entry if log else None,
                num_visualization_samples=args.num_visualization_samples,
            )

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
            pbar.close()

    print(
        f"\nTraining completed. {'Results continuously saved to ' + str(results_path) if log else 'No experiment log was written.'}"
    )

    subprocess.run(
        [
            "python3",
            "model_prediction_evolution.py",
            str(results_path),
            "--split",
            "train",
        ]
    )
