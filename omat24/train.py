# External
import torch
from pathlib import Path
import pprint

# Internal
from data import OMat24Dataset, get_dataloaders
from arg_parser import get_args
from models.fcn import FCNModel
import train_utils.fcn_train_utils as train_utils
from models.transformer_models import XTransformerModel

# Set seed & device
seed = 1000
torch.manual_seed(seed)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    # Ensure deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


if __name__ == "__main__":
    args = get_args()

    # Load dataset
    dataset_name = "rattled-300-subsampled"
    dataset_path = Path(f"datasets/{dataset_name}")
    dataset = OMat24Dataset(dataset_path=dataset_path, augment=args.augment)
    train_loader, val_loader = get_dataloaders(
        dataset, data_fraction=0.1, batch_size=args.batch_size, batch_padded=False
    )

    # # Initialize model
    # model = FCNModel()

    model = XTransformerModel(
        vocab_size=args.max_n_elements,
        max_seq_len=args.max_n_atoms,
        d_model=64,
        n_layers=6,
        n_heads=8,
        d_ff=64,
    )

    # Initialize optimizer and scheduler
    optimizer = train_utils.get_optimizer(model, learning_rate=args.lr)
    scheduler = train_utils.get_scheduler(optimizer)

    # User Hyperparam Feedback
    pprint.pprint(vars(args))
    print()

    # Train model
    model, losses = train_utils.train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        num_epochs=args.num_epochs,
        device=DEVICE,
    )

    print(losses)

    # Save model
    torch.save(model.state_dict(), f"{args.architecture}_model.pth")
