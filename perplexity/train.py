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
from data import setup_dataset
from model import *
from train_utils import train_epoch, evaluate_perplexity


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


if __name__ == "__main__":
    # Parse Arguments
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
        "--wandb_log", action="store_true", help="Enable Weights and Biases logging"
    )
    args = parser.parse_args()

    # Setup Dataset
    if args.dataset_version == "small":
        dataset = "wikitext-2-v1"
    elif args.dataset_version == "large":
        dataset = "wikitext-103-v1"
    dataset, tokenizer = setup_dataset(dataset, seq_max_length=args.seq_max_length)

    # Models, Loss, Optimizer
    if args.architecture == "FCN":
        models = MetaFullyConnectedModel(vocab_size=len(tokenizer))
    elif args.architecture == "VanillaTransformer":
        models = MetaVanillaTransformer(vocab_size=len(tokenizer))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # User Feedback
    pprint.pprint(vars(args))
    print()

    # Scaling Experiments
    for fraction in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]:
        for model in models:
            model.to(DEVICE)
            print(
                f"\nModel is on device: {DEVICE} and has {model.num_params} parameters"
            )

            # Create a subset of the dataset
            size = int(len(dataset["train"]) * fraction)
            subset = Subset(dataset["train"], indices=range(size))
            train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

            # model name schema
            model_name = f"{args.architecture}_dv={args.dataset_version}_df={fraction}_p={model.num_params}"

            if args.wandb_log:
                run = wandb.init(
                    project="wikitext-scaling",
                    name=f"{dataset}_{int(fraction*100)}%",
                    group=f"{dataset}_transformer",
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
