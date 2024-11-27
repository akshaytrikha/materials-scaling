# External
import torch
import torch.optim as optim

# Internal
from loss import compute_mae_loss


def get_optimizer(model, learning_rate=1e-3):
    return optim.Adam(model.parameters(), lr=learning_rate)


def get_scheduler(optimizer):
    return None  # No scheduler for now


def train(model, train_loader, val_loader, optimizer, scheduler, pbar, device):
    model = model.to(device)
    losses = {}

    for epoch in pbar:
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            # Move data to device
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            forces_true = batch["forces"].to(device)
            energy_true = batch["energy"].to(device)
            stress_true = batch["stress"].to(device)

            # Create mask for valid atoms
            mask = atomic_numbers != 0

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            forces_pred, energy_pred, stress_pred = model(atomic_numbers, positions)

            # Compute loss
            loss = compute_mae_loss(
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

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step
        avg_val_loss = validate(model, val_loader, device)

        # Store the losses
        losses[epoch] = {"train_loss": avg_train_loss, "val_loss": avg_val_loss}

        # Update progress bar
        pbar.set_description(f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # Update scheduler if it's defined
        if scheduler is not None:
            scheduler.step()

    return model, losses


def validate(model, val_loader, device):
    model.eval()
    total_val_loss = 0.0
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
            loss = compute_mae_loss(
                forces_pred,
                energy_pred,
                stress_pred,
                forces_true,
                energy_true,
                stress_true,
                mask,
            )

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss
