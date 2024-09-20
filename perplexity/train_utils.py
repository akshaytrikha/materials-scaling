import torch


def train_epoch(model, train_loader, val_loader, optimizer, loss_fn, device):
    """Train model for one epoch and compute the average train * validation loss

    Args:
        model (torch.nn.Module): Model to train
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fn (torch.nn.Module): Loss function to compute loss
        device (torch.device): Device to run the training on
    Returns:
        avg_train_loss (float): Average training loss for the epoch
        avg_val_loss (float): Average validation loss for the epoch
    """
    # train loop
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        inputs = batch["input_ids"].to(device)  # Keep inputs as Long for embedding
        labels = torch.roll(
            inputs, -1, dims=1
        )  # Shift inputs for next-token prediction

        # Forward pass
        outputs = model(inputs)

        # Determine output shape and compute loss accordingly
        if outputs.dim() == 3:
            # Sequence-based model (e.g., VanillaTransformer)
            batch_size, seq_length, vocab_size = outputs.size()
            outputs = outputs.reshape(-1, vocab_size)  # Flatten for loss computation
            labels = labels.reshape(-1)  # Flatten labels
        elif outputs.dim() == 2:
            # Single token prediction model (e.g., FCN)
            labels = labels[:, -1]  # Use the last token as the target
        else:
            raise ValueError(f"Unsupported output dimension: {outputs.dim()}")

        # Compute train loss
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input_ids"].to(device)  # Keep inputs as Long for embedding
            labels = torch.roll(
                inputs, -1, dims=1
            )  # Shift inputs for next-token prediction

            # Forward pass
            outputs = model(inputs)

            # Determine output shape and compute loss accordingly
            if outputs.dim() == 3:
                # Sequence-based model (e.g., VanillaTransformer)
                batch_size, seq_length, vocab_size = outputs.size()
                outputs = outputs.reshape(
                    -1, vocab_size
                )  # Flatten for loss computation
                labels = labels.reshape(-1)  # Flatten labels
            elif outputs.dim() == 2:
                # Single token prediction model (e.g., FCN)
                labels = labels[:, -1]  # Use the last token as the target
            else:
                raise ValueError(f"Unsupported output dimension: {outputs.dim()}")

            # Compute val loss
            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    return avg_train_loss, avg_val_loss
