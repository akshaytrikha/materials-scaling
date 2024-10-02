import torch
from transformers import GPT2Tokenizer

def generate_padding_mask(input_ids, pad_token_id):
    mask = (input_ids == pad_token_id)
    return mask


def compute_loss(batch, model, loss_fn, device):
    """Process a batch and compute the loss.

    Args:
        batch (dict): A batch of data containing 'input_ids'.
        model (torch.nn.Module): The model to evaluate.
        loss_fn (torch.nn.Module): The loss function to compute loss.
        device (torch.device): The device to run computations on.

    Returns:
        loss (torch.Tensor): The computed loss for the batch.
    """
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    label = batch["label"].to(device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    padding_mask = generate_padding_mask(inputs, tokenizer.pad_token_id)
    src_mask = torch.triu(torch.ones(32, 32, device=device) * float('-inf'), diagonal=1)
    # print(padding_mask)
    # print(src_mask)
    # Decode and print the tokens
    # print("Input tokens:")
    # print(inputs[0])
    # print(tokenizer.decode(inputs[0]))
    # print("Label tokens:")
    # print(labels[0])
    # print(tokenizer.decode(labels[0]))
    # print("Last token:")
    # print(label[0])
    # print(tokenizer.decode(label[0]))
    # print("======")

    # Forward pass
    outputs = model(inputs, src_mask=src_mask, src_key_padding_mask=padding_mask)

    # Determine output shape and compute loss accordingly
    if outputs.dim() == 3:
        # Sequence-based model (e.g., VanillaTransformer)
        batch_size, seq_length, vocab_size = outputs.size()
        outputs = outputs.reshape(-1, vocab_size)  # Flatten for loss computation
        labels = labels.reshape(-1)  # Flatten labels
    elif outputs.dim() == 2:
        # Single token prediction model (e.g., FCN)
        labels = label.reshape(-1)
    else:
        raise ValueError(f"Unsupported output dimension: {outputs.dim()}")

    # Compute loss
    loss = loss_fn(outputs, labels)
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
    for batch in train_loader:
        # Compute loss
        loss = compute_loss(batch, model, loss_fn, device)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_train_loss += loss.item()

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            # Compute loss
            loss = compute_loss(batch, model, loss_fn, device)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    return avg_train_loss, avg_val_loss
