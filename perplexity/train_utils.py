# train_utils.py

import torch


def train_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        inputs = batch["input_ids"].to(device)  # [batch_size, seq_length]
        labels = inputs.clone()  # [batch_size, seq_length]
        labels[:, :-1] = inputs[:, 1:]
        labels[:, -1] = -100  # Ignore the last token

        # Forward pass
        outputs = model(inputs)  # Output shape depends on the model

        # Determine output shape and compute loss accordingly
        if outputs.dim() == 3:
            # Sequence-based model (e.g., VanillaTransformer)
            batch_size, seq_length, vocab_size = outputs.size()
            # Reshape outputs and labels to compute loss over all tokens except the last
            outputs = outputs[:, :-1, :].reshape(-1, vocab_size)  # [(batch_size * (seq_length-1)), vocab_size]
            labels = labels[:, :-1].reshape(-1)  # [batch_size * (seq_length-1)]
        elif outputs.dim() == 2:
            # Single token prediction model (e.g., FCN)
            batch_size, vocab_size = outputs.size()
            # Assuming FCN predicts the next token based on the entire sequence
            # Use the last token as the prediction target
            labels = labels[:, -2]  # Use the second last token as the target
            # Adjust based on your specific labeling strategy
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
