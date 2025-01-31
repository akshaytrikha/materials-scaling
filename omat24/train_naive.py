# External
from pathlib import Path
from tqdm.auto import tqdm

# Internal
from models.naive import NaiveMagnitudeModel, NaiveDirectionModel, NaiveMeanModel
from data_utils import download_dataset
from fairchem.core.datasets import AseDBDataset


if __name__ == "__main__":
    # Setup dataset
    split_name = "val"
    dataset_name = "rattled-1000"

    dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
    if not dataset_path.exists():
        download_dataset(dataset_name, split_name)

    ase_dataset = AseDBDataset(config=dict(src=str(dataset_path)))

    # Init k model
    # k = 1
    # force_magnitude = False
    # if force_magnitude:
    #     model = NaiveMagnitudeModel(k)
    #     model_name = f"{dataset_name}_naive_magnitude_k={k}_model"
    # else:
    #     model = NaiveDirectionModel(k)
    #     model_name = f"{dataset_name}_naive_direction_k={k}_model"

    # Init mean model
    model = NaiveMeanModel()
    model_name = f"{dataset_name}_naive_mean_model"

    # Train the model
    model.train(ase_dataset)

    # Finalize the model (compute means)
    model.finalize()

    # Save the model
    Path("checkpoints/naive").mkdir(exist_ok=True)
    model_filepath = Path(f"checkpoints/naive/{model_name}.pkl")
    model_filepath.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_filepath)
    print(f"Model saved to {model_filepath}")
