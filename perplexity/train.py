# train.py

# External
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pprint
from tqdm.auto import tqdm
import warnings
import math
import os
from tempfile import TemporaryDirectory

warnings.filterwarnings("ignore", category=FutureWarning)

# Internal
from data import setup_dataset, get_data_tensors
from arg_parser import get_args
from model import *
from train_utils import train_epoch

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
    else:
        raise ValueError(f"Unsupported dataset version: {args.dataset_version}")

    dataset, tokenizer = setup_dataset(dataset_name, seq_max_length=args.seq_max_length)

    # Models, Loss
    if args.architecture == "FCN":
        models = MetaFullyConnectedModels(vocab_size=len(tokenizer))
    elif args.architecture == "VanillaTransformer":
        models = MetaVanillaTransformers(vocab_size=len(tokenizer))
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    loss_fn = nn.CrossEntropyLoss()

    # User Hyperparam Feedback
    pprint.pprint(vars(args))
    print()

    # Scaling Experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_name = f"{dataset_name}_{args.architecture}_ts={timestamp}"  # for wandb

    for data_fraction in tqdm(args.data_fractions, desc="Data Iteration"):
        # Create batchified data tensors
        train_data, val_data = get_data_tensors(
            dataset, data_fraction, args.batch_size
        )

        for model in models:
            model.to(DEVICE)
            print(
                f"\nModel is on device {DEVICE} and has {model.num_params} parameters"
            )
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            model_name = f"{args.architecture}_dv={args.dataset_version}_df={int(data_fraction*100)}_p={model.num_params}"

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
                        "architecture": args.architecture,
                        "vocab_size": len(tokenizer),
                        # Add other config parameters if needed
                    },
                )

            # Define bptt (sequence length)
            bptt = 35

            # Train the model
            for epoch in range(1, args.num_epochs + 1):
                train_loss, val_loss = train_epoch(
                    model, train_data, val_data, optimizer, loss_fn, DEVICE, bptt
                )
                print(
                    f"Dataset Size: {int(data_fraction*100)}%, Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                if args.wandb_log:
                    # Calculate Perplexity
                    train_ppl = math.exp(train_loss)
                    val_ppl = math.exp(val_loss)

                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "train_perplexity": train_ppl,
                            "val_loss": val_loss,
                            "val_perplexity": val_ppl,
                            "num_params": model.num_params,
                        }
                    )

            # Optionally, save the model or perform additional logging here
            if args.wandb_log:
                wandb.finish()
