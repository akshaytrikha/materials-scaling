# train_utils.py

import torch


def train_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        inputs = batch["input_ids"].to(device)  # Keep inputs as Long for embedding
        labels = torch.roll(inputs, -1, dims=1)  # Shift inputs for next-token prediction

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

        # Compute loss
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate_perplexity(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input_ids"].to(device)
            labels = inputs.clone()
            labels[:, :-1] = inputs[:, 1:]
            labels[:, -1] = -100  # Ignore the last token

            outputs = model(inputs)

            if outputs.dim() == 3:
                # Sequence-based model (e.g., VanillaTransformer)
                batch_size, seq_length, vocab_size = outputs.size()
                outputs = outputs[:, :-1, :].reshape(-1, vocab_size)
                labels = labels[:, :-1].reshape(-1)
            elif outputs.dim() == 2:
                # Single token prediction model (e.g., FCN)
                batch_size, vocab_size = outputs.size()
                labels = labels[:, -2]  # Adjust based on labeling strategy
            else:
                raise ValueError(f"Unsupported output dimension: {outputs.dim()}")

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity
