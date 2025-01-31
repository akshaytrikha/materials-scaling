# External
from pathlib import Path
import torch
from tqdm.auto import tqdm
import numpy as np
from fairchem.core.datasets import AseDBDataset
import math

# Internal
from models.naive import NaiveMagnitudeModel, NaiveDirectionModel, NaiveMeanModel
from data_utils import download_dataset
from loss import compute_loss


def run_naive_k(model, ase_dataset, force_magnitude, batch_size=256, device="cpu"):
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
        batch_true_properties = []
        for atoms in batch_atoms:
            e = atoms.get_potential_energy()
            if force_magnitude:
                f = np.linalg.norm(atoms.get_forces(), axis=-1)  # shape [n_atoms]
            else:
                f = atoms.get_forces()  # shape [n_atoms, 3]
            s = atoms.get_stress()
            batch_true_properties.append((e, f, s))

        # Unpack
        true_energies, true_forces, true_stresses = zip(*batch_true_properties)
        true_energies = torch.tensor(true_energies, device=device)
        true_stresses = torch.tensor(np.array(true_stresses), device=device)

        # Model predictions
        pred_energies, pred_forces, pred_stresses = model.predict_batch(batch_atoms)

        # Pad true_forces
        max_atoms = pred_forces.shape[1]  # number of atoms dimension
        padded_true_forces = []
        for f in true_forces:
            if force_magnitude:
                # shape [n_atoms]
                pad_width = (0, max_atoms - len(f))
            else:
                # shape [n_atoms, 3]
                pad_width = ((0, max_atoms - f.shape[0]), (0, 0))
            padded = np.pad(f, pad_width, mode="constant")
            padded_true_forces.append(padded)
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
            force_magnitude=force_magnitude,
        )

        total_loss += batch_loss

    average_loss = total_loss / num_batches

    if force_magnitude:
        print(f"Magnitude Average Loss Per Batch: {average_loss.item()}")
    else:
        print(f"Magnitude & Direction Average Loss Per Batch: {average_loss.item()}")


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
            force_magnitude=False,
        )

        total_loss += loss.item()

    average_loss = total_loss / len(ase_dataset)
    print(f"Average Loss Per Structure: {average_loss}")


def run_naive_mean(model, ase_dataset, batch_size=256, device="cpu"):
    """Evaluates the NaiveMeanModel on the ASE dataset in batches."""
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
        batch_true_properties = []
        for atoms in batch_atoms:
            e = atoms.get_potential_energy()
            f = atoms.get_forces()  # shape [n_atoms, 3]
            s = atoms.get_stress()
            batch_true_properties.append((e, f, s))

        # Unpack
        true_energies, true_forces, true_stresses = zip(*batch_true_properties)
        true_energies = torch.tensor(true_energies, device=device)
        true_stresses = torch.tensor(np.array(true_stresses), device=device)

        # Model predictions
        pred_energies, pred_forces, pred_stresses = model.predict_batch(batch_atoms)

        # Pad true_forces
        max_atoms = pred_forces.shape[1]  # number of atoms dimension
        padded_true_forces = []
        for f in true_forces:
            pad_width = ((0, max_atoms - f.shape[0]), (0, 0))
            padded = np.pad(f, pad_width, mode="constant")
            padded_true_forces.append(padded)
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
            force_magnitude=False,  # Since we're predicting (x, y, z) forces
        )

        total_loss += batch_loss

    average_loss = total_loss / num_batches
    print(f"NaiveMeanModel Average Loss Per Batch: {average_loss.item()}")


if __name__ == "__main__":
    # Setup val dataset (separate from training dataset)
    val_split_name = "val"
    val_dataset_name = "rattled-300-subsampled"

    val_dataset_path = Path(f"datasets/{val_split_name}/{val_dataset_name}")
    if not val_dataset_path.exists():
        download_dataset(val_dataset_name, val_split_name)

    ase_dataset = AseDBDataset(config=dict(src=str(val_dataset_path)))

    # Model was trained on the following dataset
    train_dataset_name = "rattled-1000"

    ## Setup k model
    # k = 1
    # force_magnitude = False

    # if force_magnitude:
    #     model_name = f"{dataset_name}_naive_magnitude_k={k}_model"
    #     k_model = NaiveMagnitudeModel.load(
    #         f"checkpoints/naive/{dataset_name}_naive_magnitude_k={k}_model.pkl"
    #     )
    # else:
    #     model_name = f"{dataset_name}_naive_direction_k={k}_model"
    #     k_model = NaiveDirectionModel.load(
    #         f"checkpoints/naive/{dataset_name}_naive_direction_k={k}_model.pkl"
    #     )

    # Setup mean model
    mean_model = NaiveMeanModel.load(
        f"checkpoints/naive/{train_dataset_name}_naive_mean_model.pkl"
    )

    # print(f"{k}: {force_magnitude}")
    # run_naive_k(k_model, ase_dataset, force_magnitude=False)
    # run_naive_zero(ase_dataset, force_magnitude=False)
    run_naive_mean(mean_model, ase_dataset)
