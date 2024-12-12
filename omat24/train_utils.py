# External
import torch

# Internal
from loss import compute_mae_loss

# maps data fraction to epochs multiplier
EPOCHS_SCHEDULE = {
    0.01: 5,
    0.02: 4.5,
    0.04: 3,
    0.08: 3,
    0.1: 2,
    0.2: 2,
    0.4: 1.5,
    0.8: 1,
    1.0: 1,
}


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    pbar,
    device,
    val_interval,
    total_val_steps=40,
    patience=5,
):
    model.to(device)
    losses = {}
    step = 0
    val_steps_done = 0  # Track the number of validations performed

    best_val_loss = float("inf")
    epochs_since_improvement = 0

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

            # Validate every val_interval batches
            if (step + 1) % val_interval == 0 and val_steps_done < total_val_steps:
                val_loss = run_validation(model, val_loader, device)
                current_avg_train_loss = train_loss_sum / num_train_batches
                # Store the averaged training loss and current validation loss at this step
                losses[step] = {
                    "train_loss": float(current_avg_train_loss),
                    "val_loss": float(val_loss),
                }
                val_steps_done += 1

                # Early stopping based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= patience:
                        print("Early stopping triggered")
                        return model, losses

            step += 1

            # Early stopping based on number of val steps
            if val_steps_done >= total_val_steps:
                print("Reached maximum number of validations.")
                return model, losses

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
