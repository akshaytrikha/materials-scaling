# External
import torch

# Internal
from loss import compute_mse_loss


def train_epoch(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
):
    """Train model for one epoch and compute the average train, validation loss.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (transformers.get_scheduler): Learning rate scheduler.

    Returns:
        avg_train_loss (float): Average training loss for the epoch.
        avg_val_loss (float): Average validation loss for the epoch.
    """
    # Training loop
    model.train()
    total_train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        # Forward pass
        pred_forces, pred_energy, pred_stress = model(
            batch["atomic_numbers"], batch["positions"]
        )

        # Compute loss
        loss = compute_mse_loss(pred_forces, pred_energy, pred_stress, batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # Step the scheduler after optimizer step
        scheduler.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Compute loss
            loss = compute_mse_loss(pred_forces, pred_energy, pred_stress, batch)
            if loss is not None:
                total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    # After both loops, print the final train and validation losses
    print(
        f"Epoch completed - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}"
    )

    return avg_train_loss, avg_val_loss
