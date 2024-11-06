# fcn_train_utils.py

import torch
import torch.optim as optim
from loss import compute_mse_loss

def get_optimizer(model, learning_rate=1e-3):
    return optim.Adam(model.parameters(), lr=learning_rate)

def get_scheduler(optimizer):
    # If you have a scheduler, define it here
    return None  # No scheduler for now

def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
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
            mask = (atomic_numbers != 0)  # Shape: [batch_size, max_atoms]

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

        # Update scheduler if it's defined
        if scheduler is not None:
            scheduler.step()

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
            mask = (atomic_numbers != 0)

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

def compute_loss(pred_forces, pred_energy, pred_stress, true_forces, true_energy, true_stress, mask):
    return compute_mse_loss(
        pred_forces, pred_energy, pred_stress,
        true_forces, true_energy, true_stress, mask
    )
