# External Imports
from pathlib import Path
import torch
from tqdm.auto import tqdm

# Internal Imports
from models.naive import NaiveAtomModel
from data import download_dataset
from fairchem.core.datasets import AseDBDataset
from loss import compute_mae_loss


def predict_with_model(atoms, model_path):
    # Load the trained model
    model = NaiveAtomModel.load(Path(model_path))

    # Predict properties
    predicted_energy, predicted_forces, predicted_stress = model.predict(atoms)

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

    loss = 0
    for i in tqdm(range(len(ase_dataset))):
        atoms = ase_dataset.get_atoms(i)
        true_energy = atoms.get_potential_energy()
        true_forces = atoms.get_forces()
        true_stress = atoms.get_stress()

        # Predict using the trained model
        pred_energy, pred_forces, pred_stress = predict_with_model(
            atoms, model_path=Path(f"checkpoints/{dataset_name}_naive_atom_model.pkl")
        )

        loss += compute_mae_loss(
            torch.tensor(pred_forces),
            torch.tensor(pred_energy),
            torch.tensor(pred_stress),
            torch.tensor(true_forces),
            torch.tensor(true_energy),
            torch.tensor(true_stress),
            torch.ones(len(atoms)),
        )

    print(f"Average Loss: {loss / len(ase_dataset)}")
