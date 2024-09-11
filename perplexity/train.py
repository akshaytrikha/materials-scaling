import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import wandb

# Internal
from data import setup_dataset
from model import FullyConnectedModel
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
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dataset_size', type=str, choices=['small', 'large'], default='small', help='Dataset size to use: "small" or "big"')
    parser.add_argument('--wandb_log', action='store_true', help='Enable Weights and Biases logging')
    args = parser.parse_args()

    # Setup Dataset
    if args.dataset_size == "small":
        dataset = "wikitext-2-v1"
    elif args.dataset_size == "large":
        dataset = "wikitext-103-v1"
    dataset, tokenizer = setup_dataset(dataset)

    # Init Model, Loss, Optimizer
    model = FullyConnectedModel(vocab_size=len(tokenizer))
    model.to(DEVICE)
    print(f"Model is on device: {DEVICE}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scaling Experiments
    for fraction in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]:
        # Create a subset of the dataset
        size = int(len(dataset["train"]) * fraction)
        subset = Subset(dataset["train"], indices=range(size))
        train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

        if args.wandb_log:
            run = wandb.init(
                project="wikitext-scaling",
                # name=f"{full_dataset}_{int(fraction*100)}%",
                name=f"test_{fraction}",
                config={
                    "learning_rate": args.lr,
                    "num_epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "fraction": f"{int(fraction*100)}%",
                },
            )

        # Train the model
        for epoch in range(args.num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)

            print(f"Dataset Size: {int(fraction*100)}%, Epoch: {epoch+1}, Loss: {train_loss}")
            if args.wandb_log:
                wandb.log({"loss": train_loss})

        # Evaluate Perplexity
        perplexity = evaluate_perplexity(model, train_loader, loss_fn, DEVICE)
        print(f"Dataset Size: {int(fraction*100)}%, Perplexity: {perplexity}\n")
        if args.wandb_log:
            wandb.log({"loss": train_loss, "perplexity": perplexity})
            wandb.finish()
