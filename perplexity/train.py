# External
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pprint
from tqdm.auto import tqdm
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)

# Internal
from data import setup_dataset, get_dataloaders
from model import *
from train_utils import train_epoch
from arg_parser import get_args


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


if __name__ == "__main__":
    # Parse Arguments
    args = get_args()

    # Setup Dataset
    if args.dataset_version == "small":
        dataset_name = "wikitext-2-v1"
    elif args.dataset_version == "large":
        dataset_name = "wikitext-103-v1"
    dataset, tokenizer = setup_dataset(dataset_name)

    # Models, Loss
    if args.architecture == "FCN":
        models = MetaFullyConnectedModels(vocab_size=len(tokenizer))
    elif args.architecture == "VanillaTransformer":
        models = MetaVanillaTransformers(vocab_size=len(tokenizer))
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # User Hyperparam Feedback
    pprint.pprint(vars(args))
    print()

    # Scaling Experiments
    timestamp = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    group_name = f"{dataset_name}_{args.architecture}_ts={timestamp}"  # for wandb

    for data_fraction in tqdm(args.data_fractions, desc="Data Iteration"):
        # Create a subset of the dataset
        train_loader, val_loader = get_dataloaders(
            dataset, data_fraction, args.batch_size
        )

        for model in models:
            model.to(DEVICE)
            print(
                f"\nModel is on device {DEVICE} and has {model.num_params} parameters"
            )
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            model_name = f"{args.architecture}_dv={args.dataset_version}_df={data_fraction}_p={model.num_params}"

            if args.wandb_log:
                run = wandb.init(
                    project="wikitext-scaling",
                    name=model_name,
                    group=group_name,
                    config={
                        "learning_rate": args.lr,
                        "num_epochs": args.num_epochs,
                        "batch_size": args.batch_size,
                        "data_fraction": f"{int(data_fraction*100)}%",
                    },
                )

            # Train the model
            best_val_loss = float("inf")
            for epoch in range(args.num_epochs):
                train_loss, val_loss = train_epoch(
                    model, train_loader, val_loader, optimizer, loss_fn, DEVICE
                )
                print(
                    f"Dataset Size: {int(data_fraction*100)}%, Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(f"saved_models/{group_name}", exist_ok=True)
                    model_save_path = f"saved_models/{group_name}/{model_name}.pt"
                    torch.save(model, model_save_path)
                    print(f"Model saved to {model_save_path}")

            # Evaluate Perplexity
            train_perplexity = torch.exp(torch.tensor(train_loss)).item()
            val_perplexity = torch.exp(torch.tensor(val_loss)).item()
            print(
                f"Dataset Size: {int(data_fraction*100)}%, Train Perplexity: {train_perplexity}, Val Perplexity: {val_perplexity}\n"
            )
            if args.wandb_log:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_perplexity": train_perplexity,
                        "val_loss": val_loss,
                        "val_perplexity": val_perplexity,
                        "num_params": model.num_params,
                    }
                )
            wandb.finish()
