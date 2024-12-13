# External
from pathlib import Path
import torch
from tqdm.auto import tqdm
import numpy as np
from fairchem.core.datasets import AseDBDataset

# Internal
from models.naive import NaiveAtomModel
from data import download_dataset
from loss import compute_loss


def predict_with_model(batch_atoms, model_path):
    """
    Given a batch of Atoms objects and a path to a trained NaiveAtomModel,
    load the model and predict energies, forces, and stresses.

    Parameters
    ----------
    batch_atoms : list of ASE Atoms
        The structures for prediction.
    model_path : str or Path
        Filepath to the saved model.

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor)
        predicted energies, forces, and stresses.
    """
    model = NaiveAtomModel.load(Path(model_path))
    return model.predict_batch(batch_atoms)


def run_k_naive(ase_dataset):
    batch_size = len(ase_dataset)
    total_loss = 0
    num_batches = len(ase_dataset) // batch_size

    for batch_idx in tqdm(range(num_batches)):
        batch_atoms = [
            ase_dataset.get_atoms(i)
            for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        ]
        natoms = torch.tensor([len(atoms) for atoms in batch_atoms])

        # True properties
        batch_true_properties = [
            (
                atoms.get_potential_energy(),
                np.linalg.norm(atoms.get_forces(), axis=-1),
                atoms.get_stress(),
            )
            for atoms in batch_atoms
        ]

        # Predictions
        pred_energies, pred_forces, pred_stresses = predict_with_model(
            batch_atoms,
            model_path=Path(f"checkpoints/{dataset_name}_naive_atom_model.pkl"),
        )

        true_energies, true_forces, true_stresses = zip(*batch_true_properties)
        true_energies = torch.tensor(true_energies)
        true_stresses = torch.tensor(np.array(true_stresses))

        # Pad true forces to match predicted forces shape
        max_atoms = pred_forces.shape[1]
        padded_true_forces = []
        for forces in true_forces:
            pad_width = (0, max_atoms - len(forces))
            padded_true_forces.append(np.pad(forces, pad_width, mode="constant"))
        true_forces = torch.tensor(np.array(padded_true_forces))

        # Compute loss
        total_loss += compute_loss(
            pred_forces,
            pred_energies,
            pred_stresses,
            true_forces,
            true_energies,
            true_stresses,
            torch.ones(len(true_forces)),  # Weights
            device="cpu",
            natoms=natoms,
            use_mask=False,
            convert_forces_to_magnitudes=False,
        )

    print(f"Average Loss Per Batch: {total_loss / num_batches}")


def run_zero_naive(ase_dataset, force_magnitude):
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

    # run_k_naive(ase_dataset)
    run_zero_naive(ase_dataset, force_magnitude=False)
