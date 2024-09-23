# train_utils.py

import torch
import math
from torch import nn, optim, Tensor
from typing import Tuple

def get_batch(source: Tensor, i: int, bptt: int = 35) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape `[seq_len, batch_size]`
        i: int
        bptt: int, sequence length

    Returns:
        tuple (data, target), where data has shape `[bptt, batch_size]` and
        target has shape `[bptt * batch_size]`
    """
    seq_len = min(bptt, source.size(0) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def compute_loss(data: Tensor, targets: Tensor, model: nn.Module, loss_fn: nn.Module, device: torch.device) -> Tensor:
    """Compute the loss for a batch of data.

    Args:
        data (Tensor): Input data tensor of shape `[bptt, batch_size]`.
        targets (Tensor): Target tensor of shape `[bptt * batch_size]`.
        model (nn.Module): The model to evaluate.
        loss_fn (nn.Module): The loss function to compute loss.
        device (torch.device): The device to run computations on.

    Returns:
        loss (Tensor): The computed loss for the batch.
    """
    data = data.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = model(data)  # For Transformer: [batch_size, seq_len, vocab_size]

    if outputs.dim() == 3:
        # Transformer output: [batch_size, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = outputs.size()
        outputs = outputs.reshape(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
    elif outputs.dim() == 2:
        # FCN output: [batch_size, vocab_size]
        pass  # Targets are already [batch_size]
    else:
        raise ValueError(f"Unsupported output dimension: {outputs.dim()}")

    # Compute loss
    loss = loss_fn(outputs, targets)
    return loss


def train_epoch(model: nn.Module, train_data: Tensor, val_data: Tensor, optimizer: optim.Optimizer, loss_fn: nn.Module, device: torch.device, bptt: int = 35) -> Tuple[float, float]:
    """
    Train the model for one epoch and evaluate on the validation set.

    Args:
        model (nn.Module): The model to train.
        train_data (Tensor): Batchified training data tensor.
        val_data (Tensor): Batchified validation data tensor.
        optimizer (optim.Optimizer): Optimizer for training.
        loss_fn (nn.Module): Loss function to compute loss.
        device (torch.device): Device to run the training on.
        bptt (int): Sequence length for training.

    Returns:
        Tuple containing average training loss and average validation loss.
    """
    # Training loop
    model.train()
    total_train_loss = 0.0
    train_steps = 0

    # Calculate number of batches based on bptt
    num_train_batches = train_data.size(0) // bptt

    for i in range(0, train_data.size(0) - 1, bptt):
        data, targets = get_batch(train_data, i, bptt)
        loss = compute_loss(data, targets, model, loss_fn, device)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_train_loss += loss.item()
        train_steps += 1

    avg_train_loss = total_train_loss / train_steps

    # Validation loop
    model.eval()
    total_val_loss = 0.0
    val_steps = 0

    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, bptt):
            data, targets = get_batch(val_data, i, bptt)
            loss = compute_loss(data, targets, model, loss_fn, device)
            total_val_loss += loss.item()
            val_steps += 1

    avg_val_loss = total_val_loss / val_steps

    return avg_train_loss, avg_val_loss
