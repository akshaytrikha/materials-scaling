import torch


def train_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        inputs = batch["input_ids"].to(device)  # Keep inputs as Long for embedding
        # Shift inputs to create labels for next-token prediction, but set the last token to a special padding token (e.g., -100)
        labels = inputs.clone()  # Clone inputs to create labels
        labels[:, :-1] = inputs[
            :, 1:
        ]  # Shift inputs left by one for next-token prediction
        labels[:, -1] = (
            -100
        )  # Use -100 to ignore the last token in the loss computation

        # Pass inputs to the model
        outputs = model(inputs)

        # Adjust labels for loss calculation (assuming single token prediction for simplification)
        loss = loss_fn(
            outputs, labels[:, -1]
        )  # Take the last token's output vs shifted label

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
            labels = torch.roll(inputs, -1, dims=1)[
                :, -1
            ]  # Last token prediction, labels are indices

            outputs = model(inputs)
            outputs = outputs  # Ensure this is [N, C]

            loss = loss_fn(
                outputs, labels
            )  # Check that outputs are [N, C] and labels are [N]
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity
