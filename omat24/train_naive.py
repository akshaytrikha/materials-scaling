# External
from pathlib import Path
from tqdm.auto import tqdm

# Internal
from models.naive import NaiveAtomModel
from data import download_dataset
from fairchem.core.datasets import AseDBDataset


if __name__ == "__main__":
    # Setup dataset
    split_name = "val"
    dataset_name = "rattled-300-subsampled"

    dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
    if not dataset_path.exists():
        download_dataset(dataset_name, split_name)

    ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))

    # Initialize the model
    k = 0
    model = NaiveAtomModel(k)

    # Train the model
    model.train(ase_dataset)

    # Finalize the model (compute means)
    model.finalize()

    # Save the model
    Path("checkpoints/naive").mkdir(exist_ok=True)
    model_filepath = Path(f"checkpoints/naive/{dataset_name}_naive_k={k}_model.pkl")
    model_filepath.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_filepath)
    print(f"Model saved to {model_filepath}")
