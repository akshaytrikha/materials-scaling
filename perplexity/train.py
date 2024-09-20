import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import matplotlib.pyplot as plt
import pprint
from tqdm.auto import tqdm
import pickle as pkl
import warnings

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
data_and_perplexities = []
for data_fraction in tqdm(args.data_fractions, desc="Data Iteration"):
    train_loader, val_loader = get_dataloaders(dataset, data_fraction, args.batch_size)

    for model in models:
        model.to(DEVICE)
        print(f"\nModel is on device {DEVICE} and has {model.num_params} parameters")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # name schemas
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{args.architecture}_dv={args.dataset_version}_df={data_fraction}_p={model.num_params}"
        group_name = f"{dataset_name}_{args.architecture}_ts={timestamp}"

        if args.wandb_log:
            run = wandb.init(
                project="wikitext-scaling",
                name=model_name,
                group=group_name,
                config={
                    "learning_rate": args.lr,
                    "num_epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "fraction": f"{int(data_fraction*100)}%",
                },
            )

        # Train the model
        for epoch in range(args.num_epochs):
            train_loss, val_loss = train_epoch(
                model, train_loader, val_loader, optimizer, loss_fn, DEVICE
            )

            print(
                f"Dataset Size: {int(data_fraction*100)}%, Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}"
            )
            if args.wandb_log:
                wandb.log({"loss": train_loss})

        # Evaluate Perplexity
        train_perplexity = torch.exp(torch.tensor(train_loss)).item()
        validation_perplexity = torch.exp(torch.tensor(val_loss)).item()
        # validation_perplexity = evaluate_perplexity(model, val_loader, loss_fn, DEVICE)
        data_and_perplexities.append(
            {
                "dataset_size": args.batch_size
                * len(train_loader)
                * args.seq_max_length,
                "train_perplexity": train_perplexity,
                "validation_perplexity": validation_perplexity,
                "num_params": model.num_params,
            }
        )
        print(
            f"Dataset Size: {int(data_fraction*100)}%, Train Perplexity: {train_perplexity}, Val Perplexity: {validation_perplexity}\n"
        )
        if args.wandb_log:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_perplexity": train_perplexity,
                    "val_loss": val_loss,
                    "validation_perplexity": validation_perplexity,
                    "num_params": model.num_params,
                }
            )
        wandb.finish()

# Save data_and_perplexities to a pickle file
with open("data_and_perplexities.pkl", "wb") as f:
    pkl.dump(data_and_perplexities, f)

plt.figure(figsize=(8, 6))

# Iterate over models and plot perplexities
for model in models:
    # Filter data for the current model's number of parameters
    model_data = [
        entry
        for entry in data_and_perplexities
        if entry["num_params"] == model.num_params
    ]

    # Extract dataset sizes, train perplexities, and validation perplexities
    data_sizes = [entry["dataset_size"] for entry in model_data]
    train_perplexities = [entry["train_perplexity"] for entry in model_data]
    validation_perplexities = [entry["validation_perplexity"] for entry in model_data]

    # Plot train perplexity for this model
    plt.loglog(
        data_sizes,
        train_perplexities,
        marker="o",
        linestyle="-",
        label=f"Train Perplexity - {model.num_params} params",
    )

    # Plot validation perplexity for this model
    plt.loglog(
        data_sizes,
        validation_perplexities,
        marker="o",
        linestyle="--",
        label=f"Validation Perplexity - {model.num_params} params",
    )

plt.legend()
plt.xlabel("Dataset Size")
plt.ylabel("Perplexity")
plt.title("Perplexity Scaling with Dataset Size and Model Parameters")
plt.grid(True, which="both", ls="--")
plt.savefig("plot.png")
