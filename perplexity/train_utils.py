import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import math

def train_model(model, device, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 3, lr: float = 5e-5):
    """
    Trains a model and tracks perplexity.

    Args:
        model: model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)  # [batch_size, seq_len]
            labels = batch['labels'].to(device)        # [batch_size, seq_len]

            optimizer.zero_grad()
            trloss, logits = model(input_ids=input_ids, labels=labels)  # loss: scalar, logits: [batch_size, seq_len, ntoken]

            trloss.backward()
            optimizer.step()
            total_loss += trloss.item()

        avg_loss = total_loss / len(train_loader)
        train_perplexity = math.exp(avg_loss)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                vloss, logits = model(input_ids=input_ids, labels=labels)
                val_loss += vloss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_perplexity = math.exp(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
    return avg_loss, avg_val_loss, train_perplexity, val_perplexity
