import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import wandb
import matplotlib.pyplot as plt
import pprint
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Internal
from data import setup_dataset
from model import FullyConnectedModel, VanillaTransformer
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
        dataset_name = "wikitext-2-v1"
    elif args.dataset_version == "large":
        dataset_name = "wikitext-103-v1"
    dataset, tokenizer = setup_dataset(dataset_name)

    # Init Model, Loss, Optimizer
    if args.architecture == "FCN":
        model = FullyConnectedModel(vocab_size=len(tokenizer))
    elif args.architecture == "VanillaTransformer":
        model = VanillaTransformer(vocab_size=len(tokenizer))
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # User Feedback
    print(f"\nModel is on device: {DEVICE} and has {model.num_params} parameters")
    pprint.pprint(vars(args))
    print()

    # Scaling Experiments
    data_and_perplexities = []
    for fraction in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]:
        # Create a subset of the dataset
        train_size = int(len(dataset["train"]) * fraction)
        validation_size = int(len(dataset["validation"]) * fraction)
        train_subset = Subset(dataset["train"], indices=range(train_size))
        validation_subset = Subset(dataset["validation"], indices=range(validation_size))
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_subset, batch_size=args.batch_size, shuffle=True)

        # model name schema
        model_name = f"{args.architecture}_dv={args.dataset_version}_df={fraction}_p={model.num_params}"

        if args.wandb_log:
            run = wandb.init(
                project="wikitext-scaling",
                name=f"{dataset_name}_c{args.num_epochs}_d{int(fraction*100)}%",
                group=f"{dataset_name}_transformer",
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

            print(
                f"Dataset Size: {int(fraction*100)}%, Epoch: {epoch+1}, Loss: {train_loss}"
            )
            if args.wandb_log:
                wandb.log({"loss": train_loss})

        # Evaluate Perplexity
        train_perplexity = evaluate_perplexity(model, train_loader, loss_fn, DEVICE)
        validation_perplexity = evaluate_perplexity(model, validation_loader, loss_fn, DEVICE)
        data_and_perplexities.append((args.batch_size * len(train_loader) * args.seq_max_length, train_perplexity, validation_perplexity))
        print(f"Dataset Size: {int(fraction*100)}%, Train Loss: {train_loss}, Train Perplexity: {train_perplexity}, Validation Perplexity: {validation_perplexity}\n")
        if args.wandb_log:
            wandb.log({"train_loss": train_loss, "train_perplexity": train_perplexity, "validation_perplexity": validation_perplexity})
        wandb.finish()
    data_sizes = [entry[0] for entry in data_and_perplexities]
    train_perplexities = [entry[1] for entry in data_and_perplexities]
    validation_perplexities = [entry[2] for entry in data_and_perplexities]
    plt.figure(figsize=(8, 6))
    plt.loglog(data_sizes, train_perplexities, marker="o", linestyle="-", color="blue")
    plt.loglog(data_sizes, validation_perplexities, marker="o", linestyle="-", color="green")
    plt.legend()
    plt.xlabel("Data Set Size")
    plt.ylabel("Validation Loss")
    plt.title(model_name)
    plt.grid(True, which="both", ls="--")
    plt.show()