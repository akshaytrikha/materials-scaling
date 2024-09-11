# External
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset

# Internal
from data import setup_dataset
from constants import *
from model import FullyConnectedModel
from train_utils import train_epoch, evaluate_perplexity

WANDB_LOG = True


if __name__ == "__main__":
    # Setup Dataset
    subset_dataset = "wikitext-2-v1"
    full_dataset = "wikitext-103-v1"
    dataset, tokenizer = setup_dataset(full_dataset)

    # Init Model
    model = FullyConnectedModel(vocab_size=len(tokenizer))
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Scaling Experiments
    for fraction in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]:
        # Create a subset of the dataset
        size = int(len(dataset["train"]) * fraction)
        subset = Subset(dataset["train"], indices=range(size))
        train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

        if WANDB_LOG:
            run = wandb.init(
                project="wikitext-scaling",
                name=f"{subset_dataset}_{int(fraction*100)}%",
                config={
                    "learning_rate": LR,
                    "num_epochs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "fraction": f"{int(fraction*100)}%",
                },
            )

        # Train the model
        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)

            print(f"Dataset Size: {int(fraction*100)}%, Epoch: {epoch+1}, Loss: {train_loss}")
            if WANDB_LOG:
                wandb.log({"loss": train_loss})

        # Evaluate Perplexity
        perplexity = evaluate_perplexity(model, train_loader, loss_fn, DEVICE)
        print(f"Dataset Size: {int(fraction*100)}%, Perplexity: {perplexity}\n")
        if WANDB_LOG:
            wandb.log({"loss": train_loss, "perplexity": perplexity})
            wandb.finish()
