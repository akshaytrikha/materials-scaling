# compare_scaled_unscaled_models.py

import torch
from pathlib import Path

# Import necessary classes from your existing codebase
from data import OMat24Dataset, get_dataloaders
from models.fcn import MetaFCNModels
from loss import compute_loss
from data_utils import download_dataset


# Set seed & device
SEED = 1024
torch.manual_seed(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def run_validation(model, loader, device, scale_coefficients=True):
    """
    Compute validation loss with an option to scale the loss coefficients.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to perform computations on.
        scale_coefficients (bool): Whether to apply scaling to loss coefficients.

    Returns:
        float: Average validation loss over the dataset.
    """
    model.to(device)
    model.eval()
    total_val_loss = 0.0
    num_val_batches = len(loader)

    with torch.no_grad():
        for batch in loader:
            # Move data to the appropriate device
            atomic_numbers = batch["atomic_numbers"].to(device)
            positions = batch["positions"].to(device)
            true_forces = batch["forces"].to(device)
            true_energy = batch["energy"].to(device)
            true_stress = batch["stress"].to(device)

            # Forward pass
            pred_forces, pred_energy, pred_stress = model(atomic_numbers, positions)

            # Create mask for valid atoms (non-padded)
            mask = atomic_numbers != 0
            natoms = mask.sum(dim=1)

            # Compute loss with or without scaling coefficients
            val_loss = compute_loss(
                pred_forces=pred_forces,
                pred_energy=pred_energy,
                pred_stress=pred_stress,
                true_forces=true_forces,
                true_energy=true_energy,
                true_stress=true_stress,
                mask=mask,
                device=device,
                natoms=natoms,
                use_mask=True,
                force_magnitude=False,
                scale_coefficients=scale_coefficients,  # Control scaling
            )
            total_val_loss += val_loss.item()

    if num_val_batches == 0:
        return float("inf")
    return total_val_loss / num_val_batches


def main():
    """Main function to compare scaled vs unscaled FCN models based on their validation loss."""
    # --- 1. Define Paths to the Two Checkpoint Files ---
    checkpoint_path_scaled = "checkpoints/FCN_ds33654_p9770_20250127_055858.pth"
    log_path_scaled = "checkpoints/experiments_20250127_055858.json"

    checkpoint_path_unscaled = "checkpoints/FCN_ds33654_p9770_20250127_061438.pth"
    log_path_unscaled = "checkpoints/experiments_20250127_061438.json"

    # Verify that checkpoint files exist
    if not Path(checkpoint_path_scaled).exists():
        raise FileNotFoundError(
            f"Scaled checkpoint file not found: {checkpoint_path_scaled}"
        )
    if not Path(checkpoint_path_unscaled).exists():
        raise FileNotFoundError(
            f"Unscaled checkpoint file not found: {checkpoint_path_unscaled}"
        )

    # --- 2. Prepare the Validation Dataset and DataLoader ---
    split_name = "val"
    dataset_name = "rattled-300-subsampled"
    dataset_path = Path(f"datasets/{split_name}/{dataset_name}")

    # Download dataset if not present
    if not dataset_path.exists():
        print(
            f"Dataset '{dataset_name}' not found in split '{split_name}'. Downloading..."
        )
        download_dataset(dataset_name, split_name)

    # Initialize the dataset without data augmentation for evaluation
    dataset = OMat24Dataset(dataset_path=dataset_path, augment=False)

    # Create DataLoaders with a consistent validation split
    train_loader, val_loader = get_dataloaders(
        dataset=dataset,
        train_data_fraction=0.9,  # 90% training, 10% validation
        batch_size=32,  # Adjust batch size as needed
        seed=42,  # Seed for reproducibility
        batch_padded=False,  # Use batch padding if required
    )

    # --- 3. Instantiate and Load Both FCN Models ---
    meta_models = MetaFCNModels(vocab_size=119)

    # Initialize Model Scaled
    model_scaled = meta_models[4]  # p9770 configuration
    checkpoint_scaled = torch.load(
        checkpoint_path_scaled, map_location="cpu"
    )  # Load on CPU first
    model_scaled.load_state_dict(checkpoint_scaled["model_state_dict"])
    model_scaled.to("cpu")  # Move to CPU or CUDA as needed
    model_scaled.eval()

    # Initialize Model Unscaled
    model_unscaled = meta_models[4]  # p9770 configuration
    checkpoint_unscaled = torch.load(
        checkpoint_path_unscaled, map_location="cpu"
    )  # Load on CPU first
    model_unscaled.load_state_dict(checkpoint_unscaled["model_state_dict"])
    model_unscaled.to("cpu")  # Move to CPU or CUDA as needed
    model_unscaled.eval()

    # Move models to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_scaled.to(device)
    model_unscaled.to(device)

    # --- 4. Evaluate Both Models on the Validation Set ---
    print("Evaluating Scaled Model with Scaled Loss...")
    val_loss_scaled_scaled, _ = run_validation(
        model_scaled, val_loader, device, scale_coefficients=True
    )

    print("Evaluating Scaled Model with Unscaled Loss...")
    val_loss_scaled_unscaled, _ = run_validation(
        model_scaled, val_loader, device, scale_coefficients=False
    )

    print("Evaluating Unscaled Model with Scaled Loss...")
    val_loss_unscaled_scaled, _ = run_validation(
        model_unscaled, val_loader, device, scale_coefficients=True
    )

    print("Evaluating Unscaled Model with Unscaled Loss...")
    val_loss_unscaled_unscaled, _ = run_validation(
        model_unscaled, val_loader, device, scale_coefficients=False
    )

    # --- 5. Compare and Report the Results ---
    print("\n--- Validation Loss Comparison ---")
    print(f"{'Model':<20} {'Loss_Scaled':<15} {'Loss_Unscaled':<15}")
    print("-" * 50)
    print(
        f"{'Scaled Model':<20} {val_loss_scaled_scaled:<15.5f} {val_loss_scaled_unscaled:<15.5f}"
    )
    print(
        f"{'Unscaled Model':<20} {val_loss_unscaled_scaled:<15.5f} {val_loss_unscaled_unscaled:<15.5f}"
    )

    # Optionally, you can save the results to a CSV file or handle them as needed
    # For simplicity, we're just printing them here

    # Determine which model performs better under each scaling condition
    print("\n--- Performance Analysis ---")
    if val_loss_scaled_scaled < val_loss_scaled_unscaled:
        print("Scaled Model performs better with Scaled Loss than with Unscaled Loss.")
    else:
        print("Scaled Model performs better with Unscaled Loss than with Scaled Loss.")

    if val_loss_unscaled_scaled < val_loss_unscaled_unscaled:
        print(
            "Unscaled Model performs better with Scaled Loss than with Unscaled Loss."
        )
    else:
        print(
            "Unscaled Model performs better with Unscaled Loss than with Scaled Loss."
        )

    # Overall comparison
    print("\n--- Overall Comparison ---")
    # Compare models under the same scaling condition
    if val_loss_scaled_scaled < val_loss_unscaled_scaled:
        print("Under Scaled Loss, Scaled Model performs better than Unscaled Model.")
    else:
        print("Under Scaled Loss, Unscaled Model performs better than Scaled Model.")

    if val_loss_scaled_unscaled < val_loss_unscaled_unscaled:
        print("Under Unscaled Loss, Scaled Model performs better than Unscaled Model.")
    else:
        print("Under Unscaled Loss, Unscaled Model performs better than Scaled Model.")


if __name__ == "__main__":
    main()
