# train_utils.py

import torch


def train_epoch(model, train_loader, val_loader, optimizer, loss_fn, device):
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


# def evaluate_perplexity(model, data_loader, loss_fn, device):
#     """
#     Evaluate the perplexity of the model on the given data_loader.

#     Args:
#         model (torch.nn.Module): The trained model to evaluate.
#         data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
#         loss_fn (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
#         device (torch.device): Device to perform computations on.

#     Returns:
#         float: The perplexity score.
#     """
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(data_loader):
#             inputs = batch["input_ids"].to(device)  # Keep inputs as Long for embedding
#             labels = torch.roll(
#                 inputs, -1, dims=1
#             )  # Shift inputs for next-token prediction

#             # Forward pass
#             outputs = model(inputs)

#             # Determine output shape and compute loss accordingly
#             if outputs.dim() == 3:
#                 # Sequence-based model (e.g., VanillaTransformer)
#                 labels[:, -1] = -100  # Ignore the last token in loss computation

#                 batch_size, seq_length, vocab_size = outputs.size()
#                 outputs = outputs.reshape(
#                     -1, vocab_size
#                 )  # Flatten for loss computation
#                 labels = labels[:, :-1].reshape(
#                     -1
#                 )  # Flatten labels, exclude last token

#             elif outputs.dim() == 2:
#                 # Single token prediction model (e.g., FCN)
#                 labels = labels[:, -1].reshape(-1)  # Use the last token as the target

#             else:
#                 raise ValueError(f"Unsupported output dimension: {outputs.dim()}")

#             # Compute loss
#             loss = loss_fn(outputs, labels)
#             total_loss += loss.item()

#             # Optional: Debugging statements
#             # print(f"Batch {batch_idx + 1}: Loss = {loss.item()}")

#     avg_loss = total_loss / len(data_loader)
#     print(f"AVG LOSS ----------- {avg_loss}")
#     perplexity = torch.exp(torch.tensor(avg_loss)).item()
#     return perplexity
