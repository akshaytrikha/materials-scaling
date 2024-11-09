# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# Import data handling
from data import OMat24Dataset, get_dataloaders
from arg_parser import get_args
from models.transformer_models import XTransformerModel
from models.fcn import FCNModel

# Import optimizer and training utilities
from train_utils.fcn_train_utils import get_optimizer, compute_loss


def train(model, train_loader, val_loader, optimizer, num_epochs, device):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            # Move data to device
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            forces_true = batch["forces"].to(device)
            energy_true = batch["energy"].to(device)
            stress_true = batch["stress"].to(device)

            # Create mask for valid atoms
            mask = atomic_numbers != 0  # Shape: [batch_size, max_atoms]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            forces_pred, energy_pred, stress_pred = model(atomic_numbers, positions)

            # Compute loss
            loss = compute_loss(
                forces_pred,
                energy_pred,
                stress_pred,
                forces_true,
                energy_true,
                stress_true,
                mask,
            )

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        # # Validation step
        # validate(model, val_loader, device)

    return model


# def validate(model, val_loader, device):
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for batch in val_loader:
#             # Move data to device
#             atomic_numbers = batch["atomic_numbers"].to(device)
#             positions = batch["positions"].to(device)
#             forces_true = batch["forces"].to(device)
#             energy_true = batch["energy"].to(device)
#             stress_true = batch["stress"].to(device)

#             # Create mask for valid atoms
#             mask = atomic_numbers != 0

#             # Forward pass
#             forces_pred, energy_pred, stress_pred = model(atomic_numbers, positions)

#             # Compute loss
#             loss = compute_loss(
#                 forces_pred, energy_pred, stress_pred,
#                 forces_true, energy_true, stress_true, mask
#             )

#             total_loss += loss.item()

#     avg_loss = total_loss / len(val_loader)
#     print(f"Validation Loss: {avg_loss:.4f}")


def main():
    args = get_args()

    # Load dataset
    dataset_path = Path("datasets/rattled-300-subsampled")
    dataset = OMat24Dataset(dataset_path=dataset_path)
    train_loader, val_loader = get_dataloaders(
        dataset, data_fraction=0.1, batch_size=args.batch_size, batch_padded=False
    )

    # # Initialize model
    # model = FCNModel()

    model = XTransformerModel(
        vocab_size=args.max_n_elements,
        max_seq_len=args.max_n_atoms,
        d_model=64,
        n_layers=6,
        n_heads=8,
        d_ff=64,
    )

    # Initialize optimizer
    optimizer = get_optimizer(model, learning_rate=args.lr)

    # Train model
    model = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=args.num_epochs,
        device="mps",
    )

    # # Save model
    # torch.save(model.state_dict(), "fcn_model.pth")
    # print("Model saved.")


if __name__ == "__main__":
    main()
