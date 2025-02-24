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
# Create dataset using a list with a single path (as a string)
dataset = OMat24Dataset(dataset_paths=[dataset_path])


class TestGetDataloaders(unittest.TestCase):
    def test_dataset_split_sizes(self):
        """Test if the dataset is split into correct train and validation sizes."""
        train_data_fraction = 0.01  # Use 1% of the remaining data for training
        val_data_fraction = 0.1
        # We first reserve 10% of the dataset for validation, and then use 1% of the remaining data for training
        val_size_expected = int(len(dataset) * val_data_fraction)
        train_size_expected = int(
            (len(dataset) - val_size_expected) * train_data_fraction
        )

        train_loader, val_loader = get_dataloaders(
            [dataset_path],
            train_data_fraction=train_data_fraction,
            batch_size=10,
            seed=42,
            batch_padded=True,
            val_data_fraction=val_data_fraction,
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

        # Since our get_dataloaders now returns only loaders, we check lengths as a proxy
        train_loader_1, val_loader_1 = get_dataloaders(
            [dataset_path],
            train_data_fraction=train_data_fraction,
            batch_size=10,
            batch_padded=True,
            seed=seed,
            val_data_fraction=0.1,
        )

        train_loader_2, val_loader_2 = get_dataloaders(
            [dataset_path],
            train_data_fraction=train_data_fraction,
            batch_size=10,
            batch_padded=True,
            seed=seed,
            val_data_fraction=0.1,
        )

        # Check that the number of samples is the same
        self.assertEqual(
            len(train_loader_1.dataset),
            len(train_loader_2.dataset),
            "Train splits do not match for the same seed (size check).",
        )
        self.assertEqual(
            len(val_loader_1.dataset),
            len(val_loader_2.dataset),
            "Validation splits do not match for the same seed (size check).",
        )

        # Additionally, compare actual indices (the "idx" field) to ensure they're identical
        train_idx_1 = []
        for i in range(len(train_loader_1.dataset)):
            sample = train_loader_1.dataset[i]
            idx_value = (
                int(sample["idx"]) if hasattr(sample["idx"], "item") else sample["idx"]
            )
            train_idx_1.append(idx_value)

        train_idx_2 = []
        for i in range(len(train_loader_2.dataset)):
            sample = train_loader_2.dataset[i]
            idx_value = (
                int(sample["idx"]) if hasattr(sample["idx"], "item") else sample["idx"]
            )
            train_idx_2.append(idx_value)

        self.assertEqual(
            train_idx_1,
            train_idx_2,
            "Train splits do not match for the same seed (content check).",
        )

        val_idx_1 = []
        for i in range(len(val_loader_1.dataset)):
            sample = val_loader_1.dataset[i]
            idx_value = (
                int(sample["idx"]) if hasattr(sample["idx"], "item") else sample["idx"]
            )
            val_idx_1.append(idx_value)

        val_idx_2 = []
        for i in range(len(val_loader_2.dataset)):
            sample = val_loader_2.dataset[i]
            idx_value = (
                int(sample["idx"]) if hasattr(sample["idx"], "item") else sample["idx"]
            )
            val_idx_2.append(idx_value)

        self.assertEqual(
            val_idx_1,
            val_idx_2,
            "Validation splits do not match for the same seed (content check).",
        )

    def test_different_seeds_produce_different_splits(self):
        """Test that different seeds produce different splits."""
        train_data_fraction = 0.05
        seed1 = 123
        seed2 = 4560

        train_loader_1, val_loader_1 = get_dataloaders(
            [dataset_path],
            train_data_fraction=train_data_fraction,
            batch_size=10,
            batch_padded=True,
            seed=seed1,
            val_data_fraction=0.1,
        )

        train_loader_2, val_loader_2 = get_dataloaders(
            [dataset_path],
            train_data_fraction=train_data_fraction,
            batch_size=10,
            batch_padded=True,
            seed=seed2,
            val_data_fraction=0.1,
        )

        # Check that at least one of the splits is different by comparing their "idx" fields
        train_idx_1 = []
        for i in range(len(train_loader_1.dataset)):
            sample = train_loader_1.dataset[i]
            idx_value = (
                int(sample["idx"]) if hasattr(sample["idx"], "item") else sample["idx"]
            )
            train_idx_1.append(idx_value)

        train_idx_2 = []
        for i in range(len(train_loader_2.dataset)):
            sample = train_loader_2.dataset[i]
            idx_value = (
                int(sample["idx"]) if hasattr(sample["idx"], "item") else sample["idx"]
            )
            train_idx_2.append(idx_value)

        # If the splits are truly different, these lists should not be completely identical
        self.assertNotEqual(
            train_idx_1,
            train_idx_2,
            "Different seeds produced identical train splits (content check).",
        )

        val_idx_1 = []
        for i in range(len(val_loader_1.dataset)):
            sample = val_loader_1.dataset[i]
            idx_value = (
                int(sample["idx"]) if hasattr(sample["idx"], "item") else sample["idx"]
            )
            val_idx_1.append(idx_value)

        val_idx_2 = []
        for i in range(len(val_loader_2.dataset)):
            sample = val_loader_2.dataset[i]
            idx_value = (
                int(sample["idx"]) if hasattr(sample["idx"], "item") else sample["idx"]
            )
            val_idx_2.append(idx_value)

        self.assertNotEqual(
            val_idx_1,
            val_idx_2,
            "Different seeds produced identical validation splits (content check).",
        )

    def test_custom_collate_fn_dataset_padded_dimensions(self):
        """Test that custom_collate_fn_dataset_padded() returns the correct dimensions."""
        train_data_fraction = 0.01
        seed = 42

        train_loader, _ = get_dataloaders(
            [dataset_path],
            train_data_fraction=train_data_fraction,
            batch_size=10,
            batch_padded=False,
            seed=seed,
            val_data_fraction=0.1,
        )

        batch = next(iter(train_loader))
        atoms_dim = batch["atomic_numbers"].size(1)
        max_n_atoms = dataset.max_n_atoms
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
            [dataset_path],
            train_data_fraction=0.01,
            batch_size=10,
            batch_padded=False,
            seed=42,
            val_data_fraction=0.1,
            graph=False,
            factorize=False,
        )
        batch = next(iter(train_loader))
        expected_keys = {
            "atomic_numbers",
            "positions",
            "distance_matrix",
            "energy",
            "forces",
            "stress",
        }
        self.assertTrue(
            expected_keys.issubset(batch.keys()),
            "Batch dictionary is missing keys in non-graph mode.",
        )

        # Check that "idx" and "symbols" are present when directly indexing the dataset
        expected_keys.update({"idx", "symbols"})
        x = dataset[0]
        self.assertTrue(
            expected_keys.issubset(x.keys()),
            "Dataset dictionary is missing keys in non-graph mode.",
        )

    def test_batch_keys_graph_true(self):
        """Test that the PyG Data object contains all expected attributes when graph=True."""
        dataset_graph = OMat24Dataset(dataset_paths=[dataset_path], graph=True)
        train_loader, _ = get_dataloaders(
            [dataset_path],
            train_data_fraction=0.01,
            batch_size=10,
            seed=42,
            val_data_fraction=0.1,
            graph=True,
        )
        batch = next(iter(train_loader))
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
        actual_keys = (
            set(batch.keys()) if hasattr(batch, "keys") else set(batch.__dict__.keys())
        )
        self.assertTrue(
            expected_attrs.issubset(actual_keys),
            "PyG Data object is missing attributes in graph mode.",
        )

    def test_multi_dataset_fractions(self):
        """Test that when multiple datasets are provided, each is split individually so that the combined dataset
        contains the sum of per-dataset splits. Additionally, check that the debug information ('source')
        indicates that each sample comes from the correct original dataset.
        """
        # Load second dataset
        dataset_name_2 = "aimd-from-PBE-3000-nvt"
        dataset_path_2 = Path(f"datasets/{split_name}/{dataset_name_2}")
        if not dataset_path_2.exists():
            download_dataset(dataset_name_2, split_name)

        dataset_1 = OMat24Dataset(dataset_paths=[dataset_path], debug=True)
        dataset_2 = OMat24Dataset(dataset_paths=[dataset_path_2], debug=True)

        # Set fractions: 1% from each dataset (after reserving 10% for validation)
        train_data_fraction = 0.01
        val_data_fraction = 0.1

        val_1 = int(len(dataset_1) * val_data_fraction)
        train_1 = int((len(dataset_1) - val_1) * train_data_fraction)
        val_2 = int(len(dataset_2) * val_data_fraction)
        train_2 = int((len(dataset_2) - val_2) * train_data_fraction)

        train_loader, val_loader = get_dataloaders(
            [dataset_path, dataset_path_2],
            train_data_fraction=train_data_fraction,
            batch_size=10,
            seed=42,
            val_data_fraction=val_data_fraction,
            graph=False,
            batch_padded=True,
        )

        self.assertEqual(
            len(train_loader.dataset),
            train_1 + train_2,
            "Combined training set size mismatch.",
        )
        self.assertEqual(
            len(val_loader.dataset),
            val_1 + val_2,
            "Combined validation set size mismatch.",
        )


if __name__ == "__main__":
    unittest.main()
