# External
from pathlib import Path

# Internal
from models.naive import NaiveMagnitudeModel, NaiveDirectionModel, NaiveMeanModel
from data_utils import download_dataset, VALID_DATASETS
from data import get_dataloaders
from fairchem.core.datasets import AseDBDataset


if __name__ == "__main__":
    # Setup dataset
    split_name = "val"

    # Download datasets if not present
    dataset_paths = []
    for dataset_name in VALID_DATASETS:
        dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
        if not dataset_path.exists():
            download_dataset(dataset_name, dataset_name)
        dataset_paths.append(dataset_path)

    # Load dataset
    train_loader, val_loader = get_dataloaders(
        dataset_paths,
        train_data_fraction=0.9,
        batch_size=64,
        seed=1024,
        batch_padded=False,
        val_data_fraction=0.1,
        train_workers=8,
        val_workers=8,
        graph=False,
    )

    train_dataset = train_loader.dataset

    # # Init k model
    # for k in range(1, 6):
    #     force_magnitude = False
    #     if force_magnitude:
    #         model = NaiveMagnitudeModel(k)
    #         model_name = f"{dataset_name}_naive_magnitude_k={k}_model"
    #     else:
    #         model = NaiveDirectionModel(k)
    #         model_name = f"{dataset_name}_naive_direction_k={k}_model"

    # Init mean model
    model = NaiveMeanModel()
    model_name = f"all_val_naive_mean_model"

    # Train the model
    model.train(train_dataset)

    # Finalize the model (compute means)
    model.finalize()

    # Save the model
    Path("checkpoints/naive").mkdir(exist_ok=True)
    model_filepath = Path(f"checkpoints/naive/{model_name}.pkl")
    model_filepath.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_filepath)
    print(f"Model saved to {model_filepath}")
