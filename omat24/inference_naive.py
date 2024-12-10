# External Imports
from pathlib import Path
import torch
from tqdm.auto import tqdm

# Internal Imports
from models.naive import NaiveAtomModel
from data import download_dataset
from fairchem.core.datasets import AseDBDataset
from loss import compute_mae_loss
import numpy as np


def predict_with_model(batch_atoms, model_path):
    # Load the trained model
    model = NaiveAtomModel.load(Path(model_path))

    # Predict properties
    predicted_energy, predicted_forces, predicted_stress = model.predict_batch(batch_atoms)

    return predicted_energy, predicted_forces, predicted_stress


# Example usage
if __name__ == "__main__":
    # Setup dataset
    dataset_name = "rattled-300-subsampled"
    dataset_path = Path(f"datasets/{dataset_name}")

    if not dataset_path.exists():
        # Fetch and uncompress dataset
        download_dataset(dataset_name)

    ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))
    batch_size = len(ase_dataset)
    loss = 0
    num_batches = len(ase_dataset) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Batch Processing"):
        batch_atoms = [ase_dataset.get_atoms(i) for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)]
        natoms = torch.tensor([len(atoms) for atoms in batch_atoms])
        batch_true_properties = [
            (atoms.get_potential_energy(), np.linalg.norm(atoms.get_forces(), axis=-1), atoms.get_stress())
            for atoms in batch_atoms
        ]
        batch_predictions = predict_with_model(batch_atoms, model_path=Path(f"checkpoints/{dataset_name}_naive_atom_model.pkl"))
        pred_energies, pred_forces, pred_stresses = batch_predictions[:][0].clone().detach(), batch_predictions[:][1].clone().detach(), batch_predictions[:][2].clone().detach()
        true_energies, true_forces, true_stresses = zip(*batch_true_properties)
        true_energies, true_stresses = torch.tensor(np.array(true_energies)), torch.tensor(np.array(true_stresses))
        max_atoms = max(force.shape[0] for force in pred_forces)
        # Pad forces to have the same number of atoms
        padded_forces = []
        for force in true_forces:
            pad_width = (0, max_atoms - force.shape[0])  # Pad along atom dimension
            padded_forces.append(np.pad(force, pad_width, mode='constant'))
        # Convert to a PyTorch tensor
        true_forces = torch.tensor(np.array(padded_forces))
        loss += compute_mae_loss(
            pred_forces,
            pred_energies,
            pred_stresses,
            true_forces,
            true_energies,
            true_stresses,
            torch.ones(len(true_forces)),
            natoms=natoms,
            use_mask=False,
            convert_forces_to_magnitudes=False
        )
    print(f"Total Loss: {loss}")
