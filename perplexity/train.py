import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import wandb
import pprint
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Internal
from data import setup_dataset, get_dataloaders
from model import *
from train_utils import train_epoch, evaluate_perplexity
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
    dataset, tokenizer = setup_dataset(dataset_name, seq_max_length=args.seq_max_length)

    # Models, Loss
    if args.architecture == "FCN":
        models = MetaFullyConnectedModels(vocab_size=len(tokenizer))
    elif args.architecture == "VanillaTransformer":
        models = MetaVanillaTransformers(vocab_size=len(tokenizer))
    loss_fn = nn.CrossEntropyLoss()

    # User Feedback
    pprint.pprint(vars(args))
    print()

    # Scaling Experiments
    for fraction in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]:
        train_loader, val_loader = get_dataloaders(dataset, fraction, args.batch_size)

        # Create a subset of the dataset
        size = int(len(dataset["train"]) * fraction)
        subset = Subset(dataset["train"], indices=range(size))
        train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

        for model in models:
            model.to(DEVICE)
            print(
                f"\nModel is on device: {DEVICE} and has {model.num_params} parameters"
            )
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            # Model Name Schema
            model_name = f"{args.architecture}_dv={args.dataset_version}_df={fraction}_p={model.num_params}"

            # Initialize Logging
            if args.wandb_log:
                run = wandb.init(
                    project="wikitext-scaling",
                    name=model_name,
                    group=f"{dataset_name}_fcn",
                    config={
                        "learning_rate": args.lr,
                        "num_epochs": args.num_epochs,
                        "batch_size": args.batch_size,
                        "fraction": f"{int(fraction*100)}%",
                    },
                )

            # Train the model
            for epoch in range(args.num_epochs):
                train_loss = train_epoch(
                    model, train_loader, optimizer, loss_fn, DEVICE
                )

                print(
                    f"Dataset Size: {int(fraction*100)}%, Epoch: {epoch+1}, Loss: {train_loss}"
                )
                if args.wandb_log:
                    wandb.log({"loss": train_loss})

            # Evaluate Perplexity
            perplexity = evaluate_perplexity(model, train_loader, loss_fn, DEVICE)
            print(f"Dataset Size: {int(fraction*100)}%, Perplexity: {perplexity}\n")
            if args.wandb_log:
                wandb.log({"loss": train_loss, "perplexity": perplexity})
                wandb.finish()
