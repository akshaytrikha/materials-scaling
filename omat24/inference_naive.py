# External
from pathlib import Path
import torch
from tqdm.auto import tqdm
import numpy as np
from fairchem.core.datasets import AseDBDataset
import math

# Internal
from models.naive import NaiveAtomModel
from data import download_dataset
from loss import compute_loss


def run_naive_k(model, ase_dataset, batch_size=256, device="cpu"):
    """Evaluates a naive k model on the ASE dataset in batches"""
    total_loss = 0
    dataset_size = len(ase_dataset)
    num_batches = math.ceil(dataset_size / batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
        # Determine the start and end indices of the current batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, dataset_size)

        # Extract the batch atoms
        batch_atoms = [ase_dataset.get_atoms(i) for i in range(start_idx, end_idx)]
        natoms = torch.tensor([len(atoms) for atoms in batch_atoms], device=device)

        # Extract true properties
        batch_true_properties = [
            (
                atoms.get_potential_energy(),
                np.linalg.norm(atoms.get_forces(), axis=-1),
                atoms.get_stress(),
            )
            for atoms in batch_atoms
        ]

        # Model predictions
        pred_energies, pred_forces, pred_stresses = model.predict_batch(batch_atoms)

        # Unpack true properties
        true_energies, true_forces, true_stresses = zip(*batch_true_properties)
        true_energies = torch.tensor(true_energies, device=device)
        true_stresses = torch.tensor(np.array(true_stresses), device=device)

        # Pad true forces to match the predicted forces shape
        max_atoms = pred_forces.shape[1]
        padded_true_forces = []
        for forces in true_forces:
            pad_width = (0, max_atoms - len(forces))
            padded_forces = np.pad(forces, pad_width, mode="constant")
            padded_true_forces.append(padded_forces)
        true_forces = torch.tensor(np.array(padded_true_forces), device=device)

        # Compute loss for the current batch
        batch_loss = compute_loss(
            pred_forces,
            pred_energies,
            pred_stresses,
            true_forces,
            true_energies,
            true_stresses,
            mask=torch.ones(len(true_forces), device=device),
            device=device,
            natoms=natoms,
            use_mask=False,
            convert_forces_to_magnitudes=False,
        )

        total_loss += batch_loss

    average_loss = total_loss / num_batches
    print(f"Average Loss Per Batch: {average_loss.item()}")


def run_naive_zero(ase_dataset, force_magnitude):
    """Calculates the loss if a model predicts zero for all properties."""
    total_loss = 0

    for i in range(len(ase_dataset)):
        atoms = ase_dataset.get_atoms(i)
        natoms = torch.tensor(len(atoms))
        true_energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
        forces = atoms.get_forces()
        if force_magnitude:
            true_forces = torch.tensor(
                np.linalg.norm(forces, axis=-1), dtype=torch.float32
            )
        else:
            true_forces = torch.tensor(forces, dtype=torch.float32)
        true_stress = torch.tensor(atoms.get_stress(), dtype=torch.float32).unsqueeze(0)

        # Predictions: zeros
        pred_energies = torch.zeros_like(true_energy)
        pred_forces = torch.zeros_like(true_forces)
        pred_stresses = torch.zeros_like(true_stress)

        # Compute loss
        loss = compute_loss(
            pred_forces,
            pred_energies,
            pred_stresses,
            true_forces,
            true_energy,
            true_stress,
            torch.ones(len(true_forces)),  # Weights (assuming equal weighting)
            device="cpu",
            natoms=natoms,
            use_mask=False,
            convert_forces_to_magnitudes=False,
        )

        total_loss += loss.item()

    average_loss = total_loss / len(ase_dataset)
    print(f"Average Loss Per Structure: {average_loss}")


if __name__ == "__main__":
    # Setup dataset
    split_name = "val"
    dataset_name = "rattled-300-subsampled"

    dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
    if not dataset_path.exists():
        download_dataset(dataset_name, split_name)

    ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))

    # Setup model
    k = 0
    model_path = Path(f"checkpoints/naive/{dataset_name}_naive_k={k}_model.pkl")
    model = NaiveAtomModel.load(model_path)

    run_naive_k(model, ase_dataset)
    # run_naive_zero(ase_dataset, force_magnitude=False)
