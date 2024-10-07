import torch
from transformers import GPT2Tokenizer

def generate_padding_mask(input_ids, pad_token_id):
    mask = input_ids == pad_token_id
    return mask

def compute_loss(batch, model, loss_fn, device):
    """Process a batch and compute the loss.

    Args:
        batch (dict): A batch of data containing 'input_ids', 'labels', 'label', 'src_key_padding_mask'.
        model (torch.nn.Module): The model to evaluate.
        loss_fn (torch.nn.Module): The loss function to compute loss.
        device (torch.device): The device to run computations on.

    Returns:
        loss (torch.Tensor): The computed loss for the batch.
    """
    inputs = batch["input_ids"].to(device)             # [batch_size, seq_length]
    labels = batch["labels"].to(device)               # [batch_size, seq_length]
    label = batch["label"].to(device)                 # [batch_size, 1]
    src_key_padding_mask = batch["src_key_padding_mask"].to(device)  # [batch_size, seq_length]

    # Forward pass
    outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)  # [batch_size, seq_length, vocab_size]

    # Determine output shape and compute loss accordingly
    if outputs.dim() == 3:
        # Sequence-based model (e.g., Transformer)
        batch_size, seq_length, vocab_size = outputs.size()
        outputs = outputs.reshape(-1, vocab_size)  # [batch_size * seq_length, vocab_size]
        labels = labels.reshape(-1)                # [batch_size * seq_length]
    elif outputs.dim() == 2:
        # Single token prediction model (e.g., FCN)
        labels = label.reshape(-1)
    else:
        raise ValueError(f"Unsupported output dimension: {outputs.dim()}")

    # Compute loss
    loss = loss_fn(outputs, labels)

    # Check for NaN loss and return None if NaN
    if torch.isnan(loss):
        print("Warning: NaN loss detected. Skipping this batch.")
        return None

    return loss

def train_epoch(model, train_loader, val_loader, optimizer, loss_fn, device):
    """
    Train model for one epoch and compute the average train * validation loss.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (torch.nn.Module): Loss function to compute loss.
        device (torch.device): Device to run the training on.

    Returns:
        avg_train_loss (float): Average training loss for the epoch.
        avg_val_loss (float): Average validation loss for the epoch.
    """
    # Training loop
    model.train()
    total_train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        # Compute loss
        loss = compute_loss(batch, model, loss_fn, device)

        # Skip this batch if loss is None (NaN)
        if loss is None:
            continue

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Compute loss
            loss = compute_loss(batch, model, loss_fn, device)
            if loss is not None:
                total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    # After both loops, print the final train and validation losses
    print(f"Epoch completed - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

    return avg_train_loss, avg_val_loss
