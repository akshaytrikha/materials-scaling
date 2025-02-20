# External
from pathlib import Path
import unittest

# Internal
from data import get_dataloaders, OMat24Dataset
from data_utils import download_dataset


# Load dataset
split_name = "val"
dataset_name = "rattled-300-subsampled"

dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
if not dataset_path.exists():
    download_dataset(dataset_name, split_name)
dataset = OMat24Dataset(dataset_paths=[dataset_path])


class TestGetDataloaders(unittest.TestCase):
    def test_dataset_split_sizes(self):
        """Test if the dataset is split into correct train and validation sizes."""
        train_data_fraction = 0.01  # Use 1% of the full dataset to train with
        val_data_fraction = 0.1
        # We first reserve 10% of the dataset for validation, and then use 1% of the remaining data for training
        val_size_expected = int(len(dataset) * val_data_fraction)
        train_size_expected = int(
            (len(dataset) - val_size_expected) * train_data_fraction
        )

        train_loader, val_loader = get_dataloaders(
            dataset=dataset,
            train_data_fraction=train_data_fraction,
            batch_size=10,
            batch_padded=True,
            seed=42,
        )

        # Method 1 to get total number of training samples
        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset)

        self.assertEqual(
            train_samples, train_size_expected, "Training set size mismatch."
        )
        self.assertEqual(
            val_samples, val_size_expected, "Validation set size mismatch."
        )

        # Method 2 to get total number of training samples
        count = 0
        for batch in train_loader:
            # Assuming 'energy' is always present and has shape [batch_size]
            batch_size = batch["energy"].size(0)
            count += batch_size

        self.assertEqual(count, len(train_loader.dataset), "Sample counting mismatch.")

    def test_reproducibility_same_seed(self):
        """Test that using the same seed results in identical splits."""
        train_data_fraction = 0.05
        seed = 123

        train_loader_1, val_loader_1, train_indices_1, val_indices_1 = get_dataloaders(
            dataset=dataset,
            train_data_fraction=train_data_fraction,
            batch_size=10,
            batch_padded=True,
            seed=seed,
            return_indices=True,  # Get indices for testing
        )

        train_loader_2, val_loader_2, train_indices_2, val_indices_2 = get_dataloaders(
            dataset=dataset,
            train_data_fraction=train_data_fraction,
            batch_size=10,
            batch_padded=True,
            seed=seed,
            return_indices=True,  # Get indices for testing
        )

        self.assertEqual(
            train_indices_1,
            train_indices_2,
            "Train splits do not match for the same seed.",
        )
        self.assertEqual(
            val_indices_1,
            val_indices_2,
            "Validation splits do not match for the same seed.",
        )

    def test_different_seeds_produce_different_splits(self):
        """Test that different seeds produce different splits."""
        train_data_fraction = 0.05
        seed1 = 123
        seed2 = 4560

        train_loader_1, val_loader_1, train_indices_1, val_indices_1 = get_dataloaders(
            dataset=dataset,
            train_data_fraction=train_data_fraction,
            batch_size=10,
            batch_padded=True,
            seed=seed1,
            return_indices=True,  # Get indices for testing
        )

        train_loader_2, val_loader_2, train_indices_2, val_indices_2 = get_dataloaders(
            dataset=dataset,
            train_data_fraction=train_data_fraction,
            batch_size=10,
            batch_padded=True,
            seed=seed2,
            return_indices=True,  # Get indices for testing
        )

        # Check that at least one of the splits is different
        train_diff = train_indices_1 != train_indices_2
        val_diff = val_indices_1 != val_indices_2

        self.assertTrue(
            train_diff or val_diff, "Different seeds produced identical splits."
        )

    def test_custom_collate_fn_dataset_padded_dimensions(self):
        """Test that custom_collate_fn_dataset_padded() returns the correct dimensions."""
        train_data_fraction = 0.01
        seed = 42

        train_loader, _ = get_dataloaders(
            dataset=dataset,
            train_data_fraction=train_data_fraction,
            batch_size=10,
            batch_padded=False,
            seed=seed,
        )

        # Check that the atomic numbers are padded to max_n_atoms
        batch = next(iter(train_loader))
        atoms_dim = batch["atomic_numbers"].size(1)
        max_n_atoms = train_loader.dataset.dataset.max_n_atoms
        self.assertEqual(
            atoms_dim,
            max_n_atoms,
            "Batch size is not padded to max_n_atoms for dataset.",
        )

    def test_batch_keys_graph_false(self):
        """Test that the batch dictionary contains all expected keys when graph=False.

        Though the dataset's __getitem__() returns a dict containing the keys "idx" and "symbols",
        the batch dictionary returned by the DataLoader does not contain these keys because of the
        custom collate functions.
        """
        train_loader, _ = get_dataloaders(
            dataset=dataset,
            train_data_fraction=0.01,
            batch_size=10,
            batch_padded=False,
            seed=42,
            graph=False,
        )
        batch = next(iter(train_loader))
        expected_keys = {
            "atomic_numbers",
            "positions",
            "distance_matrix",
            "factorized_matrix",
            "energy",
            "forces",
            "stress",
        }
        self.assertTrue(
            expected_keys.issubset(batch.keys()),
            "Batch dictionary is missing keys in non-graph mode.",
        )

        # Check that "idx" and "symbols" are present when you directly index the dataset
        expected_keys.update({"idx", "symbols"})
        x = dataset[0]
        self.assertTrue(
            expected_keys.issubset(x.keys()),
            "Dataset dictionary is missing keys in non-graph mode.",
        )

    def test_batch_keys_graph_true(self):
        """Test that the PyG Data object contains all expected attributes when graph=True."""
        # Instantiate a dataset that returns PyG Data objects.
        dataset_graph = OMat24Dataset(dataset_paths=[dataset_path], graph=True)
        train_loader, _ = get_dataloaders(
            dataset=dataset_graph,
            train_data_fraction=0.01,
            batch_size=10,
            seed=42,
            graph=True,
        )
        batch = next(iter(train_loader))
        # Expected attributes in the PyG Data object.
        expected_attrs = {
            "pos",
            "atomic_numbers",
            "edge_index",
            "edge_attr",
            "energy",
            "forces",
            "stress",
            "natoms",
        }
        # PyG Data objects support the keys() method.
        actual_keys = (
            set(batch.keys()) if hasattr(batch, "keys") else set(batch.__dict__.keys())
        )
        self.assertTrue(
            expected_attrs.issubset(actual_keys),
            "PyG Data object is missing attributes in graph mode.",
        )
