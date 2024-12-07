# External
import torch
import torch.optim as optim

# Internal
from loss import compute_mae_loss


def get_optimizer(model, learning_rate=1e-3):
    return optim.Adam(model.parameters(), lr=learning_rate)


def get_scheduler(optimizer):
    return None  # No scheduler for now


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    pbar,
    device,
    val_interval=50,
):
    model.to(device)
    losses = {}
    step = 0

    for epoch in pbar:
        model.train()
        train_loss_sum = 0.0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            forces_true = batch["forces"].to(device)
            energy_true = batch["energy"].to(device)
            stress_true = batch["stress"].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            forces_pred, energy_pred, stress_pred = model(atomic_numbers, positions)

            # Compute loss
            mask = atomic_numbers != 0
            train_loss = compute_mae_loss(
                forces_pred,
                energy_pred,
                stress_pred,
                forces_true,
                energy_true,
                stress_true,
                mask,
            )

            # Backward pass and optimization
            train_loss.backward()
            optimizer.step()

            # Accumulate training loss
            train_loss_sum += train_loss.item()
            num_train_batches += 1

            # Validate every `val_interval` batches
            if (batch_idx + 1) % val_interval == 0:
                val_loss = run_validation(model, val_loader, device)
                current_avg_train_loss = train_loss_sum / num_train_batches
                # Store the averaged training loss and current validation loss at this step
                losses[step] = {
                    "train_loss": float(current_avg_train_loss),
                    "val_loss": float(val_loss),
                }

            step += 1

        # At the end of the epoch, do a validation run
        val_loss = run_validation(model, val_loader, device)
        avg_train_loss = train_loss_sum / num_train_batches
        losses[step] = {
            "train_loss": float(avg_train_loss),
            "val_loss": float(val_loss),
        }

        if scheduler is not None:
            scheduler.step()

    return model, losses


def run_validation(model, val_loader, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            forces_true = batch["forces"].to(device)
            energy_true = batch["energy"].to(device)
            stress_true = batch["stress"].to(device)

            # Forward pass
            forces_pred, energy_pred, stress_pred = model(atomic_numbers, positions)

            # Compute loss
            mask = atomic_numbers != 0
            val_loss = compute_mae_loss(
                forces_pred,
                energy_pred,
                stress_pred,
                forces_true,
                energy_true,
                stress_true,
                mask,
            )

            total_val_loss += val_loss.item()

    return total_val_loss / len(val_loader)
