# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# Import data handling
from data import OMat24Dataset, get_dataloaders

# Import the FCN model
from fcn import FCNModel

# Import optimizer and training utilities
from fcn_train_utils import get_optimizer, compute_loss

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
                forces_pred, energy_pred, stress_pred,
                forces_true, energy_true, stress_true, mask
            )

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        # Validation step
        validate(model, val_loader, device)

    return model

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            forces_true = batch["forces"].to(device)
            energy_true = batch["energy"].to(device)
            stress_true = batch["stress"].to(device)

            # Create mask for valid atoms
            mask = atomic_numbers != 0

            # Forward pass
            forces_pred, energy_pred, stress_pred = model(atomic_numbers, positions)

            # Compute loss
            loss = compute_loss(
                forces_pred, energy_pred, stress_pred,
                forces_true, energy_true, stress_true, mask
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train FCN Model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of data to use")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset_path = Path(args.dataset_path)
    dataset = OMat24Dataset(dataset_path=dataset_path)
    train_loader, val_loader = get_dataloaders(
        dataset, data_fraction=args.data_fraction, batch_size=args.batch_size
    )

    # Initialize model
    model = FCNModel()

    # Initialize optimizer
    optimizer = get_optimizer(model, learning_rate=args.learning_rate)

    # Train model
    model = train(model, train_loader, val_loader, optimizer, num_epochs=args.num_epochs, device=device)

    # Save model
    torch.save(model.state_dict(), "fcn_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    main()
