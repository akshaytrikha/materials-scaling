# train.py

import torch
import argparse
from pathlib import Path

# Import data handling
from data import OMat24Dataset, get_dataloaders

def main():
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--dataset_path", type=str, default="datasets/rattled-300-subsampled", help="Path to dataset")
    parser.add_argument("--model", type=str, required=True, help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of data to use")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset_path = Path(args.dataset_path)
    dataset = OMat24Dataset(dataset_path=dataset_path)
    train_loader, val_loader = get_dataloaders(
        dataset, data_fraction=args.data_fraction, batch_size=args.batch_size
    )

    print(args.model)

    # Dynamically import the model and training utilities based on the model name
    if args.model == "fcn":
        from models.fcn import FCNModel as Model
        import train_utils.fcn_train_utils as train_utils
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")

    # Initialize model
    model = Model()

    # Initialize optimizer and scheduler
    optimizer = train_utils.get_optimizer(model, learning_rate=args.learning_rate)
    scheduler = train_utils.get_scheduler(optimizer)

    # Train model
    model = train_utils.train(
        model, train_loader, val_loader, optimizer, scheduler,
        num_epochs=args.num_epochs, device=device
    )

    # Save model
    torch.save(model.state_dict(), f"{args.model}_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    main()
