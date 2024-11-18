# External
import torch
from pathlib import Path
import pprint
import json
from datetime import datetime
from tqdm import tqdm

# Internal
from data import OMat24Dataset, get_dataloaders
from arg_parser import get_args
from models.fcn import MetaFCNModels
import train_utils.fcn_train_utils as train_utils
from models.transformer_models import XTransformerModel

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


def train_model(model, train_loader, val_loader, args):
    # Initialize optimizer and scheduler
    optimizer = train_utils.get_optimizer(model, learning_rate=args.lr)
    scheduler = train_utils.get_scheduler(optimizer)

    # Create progress bar for epochs
    pbar = tqdm(range(args.num_epochs), desc="Training")

    # Train model
    model, losses = train_utils.train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        pbar,
        device=DEVICE,
    )

    return model, losses


if __name__ == "__main__":
    args = get_args()

    # Load dataset
    dataset_name = "rattled-300-subsampled"
    dataset_path = Path(f"datasets/{dataset_name}")
    dataset = OMat24Dataset(dataset_path=dataset_path, augment=args.augment)
    train_loader, val_loader = get_dataloaders(
        dataset, data_fraction=0.1, batch_size=args.batch_size, batch_padded=False
    )

    # # # Initialize model

    # model = XTransformerModel(
    #     vocab_size=args.max_n_elements,
    #     max_seq_len=args.max_n_atoms,
    #     d_model=64,
    #     n_layers=6,
    #     n_heads=8,
    #     d_ff=64,
    # )

    # # Initialize optimizer and scheduler
    # optimizer = train_utils.get_optimizer(model, learning_rate=args.lr)
    # scheduler = train_utils.get_scheduler(optimizer)

    # # User Hyperparam Feedback
    # pprint.pprint(vars(args))
    # print()

    # # Train model
    # model, losses = train_utils.train(
    #     model,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     scheduler,
    #     num_epochs=args.num_epochs,
    #     device=DEVICE,
    # )

    # print(losses)

    # # Save model
    # torch.save(model.state_dict(), f"{args.architecture}_model.pth")

    # Initialize meta model class
    meta_models = MetaFCNModels(vocab_size=args.max_n_elements)

    # Dictionary to store results for all models
    all_results = {}

    # Train each model configuration
    print("\nStarting training for multiple architectures...")
    for model_idx, model in enumerate(meta_models):
        print(f"\nModel {model_idx + 1}/{len(meta_models)}")
        print(
            f"Config: e{model.embedding_dim}_h{model.hidden_dim}_d{model.depth} ({model.num_params:,} params)"
        )

        # Train the model
        trained_model, losses = train_model(model, train_loader, val_loader, args)

        # Save model checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"checkpoints/fcn_model_{model_idx}_{timestamp}.pth"
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(
            {
                "model_state_dict": trained_model.state_dict(),
                "config": {
                    "embedding_dim": model.embedding_dim,
                    "hidden_dim": model.hidden_dim,
                    "depth": model.depth,
                    "num_params": model.num_params,
                },
                "losses": losses,
            },
            checkpoint_path,
        )

        # Store results
        all_results[f"model_{model_idx}"] = {
            "config": {
                "embedding_dim": model.embedding_dim,
                "hidden_dim": model.hidden_dim,
                "depth": model.depth,
                "num_params": model.num_params,
            },
            "losses": losses,
            "checkpoint_path": checkpoint_path,
        }

    # Save all results to JSON
    results_path = Path("results") / f"results_{timestamp}.json"
    Path("results").mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nTraining completed. Results saved to {results_path}")
