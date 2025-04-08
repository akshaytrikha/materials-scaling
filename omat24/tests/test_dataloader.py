# External
from pathlib import Path
import unittest
import torch

# Internal
from data import get_dataloaders, OMat24Dataset
from data_utils import download_dataset, DATASET_INFO
from matrix import rotate_atom_positions, random_rotate_atom_positions, rotate_stress

# Load dataset
split_name = "val"
dataset_name = "rattled-300-subsampled"

dataset_path = Path(f"datasets/{split_name}/{dataset_name}")
if not dataset_path.exists():
    download_dataset(dataset_name, split_name)
# Create dataset using a list with a single path (as a string)
dataset = OMat24Dataset(dataset_paths=[dataset_path], architecture="Transformer")


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
            architecture="Transformer",
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
            seed=seed,
            architecture="FCN",
            val_data_fraction=0.1,
        )

        train_loader_2, val_loader_2 = get_dataloaders(
            [dataset_path],
            train_data_fraction=train_data_fraction,
            batch_size=10,
            seed=seed,
            architecture="Transformer",
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
            seed=seed1,
            architecture="FCN",
            val_data_fraction=0.1,
        )

        train_loader_2, val_loader_2 = get_dataloaders(
            [dataset_path],
            train_data_fraction=train_data_fraction,
            batch_size=10,
            seed=seed2,
            architecture="Transformer",
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

    # def test_custom_collate_fn_dataset_padded_dimensions(self):
    #     """Test that custom_collate_fn_dataset_padded() returns the correct dimensions."""
    #     train_data_fraction = 0.01
    #     seed = 42

    #     train_loader, _ = get_dataloaders(
    #         [dataset_path],
    #         train_data_fraction=train_data_fraction,
    #         batch_size=10,
    #         batch_padded=False,
    #         seed=seed,
    #         architecture="FCN",
    #         val_data_fraction=0.1,
    #     )

    #     batch = next(iter(train_loader))
    #     atoms_dim = batch["atomic_numbers"].size(1)

    #     split_name = dataset_path.parent.name
    #     max_n_atoms = max(
    #         info["max_n_atoms"] for info in DATASET_INFO[split_name].values()
    #     )
    #     self.assertEqual(
    #         atoms_dim,
    #         max_n_atoms,
    #         "Batch size is not padded to max_n_atoms for dataset.",
    #     )

    # def test_batch_keys_graph_false(self):
    # """Test that the batch dictionary contains all expected keys when graph=False.

    # Though the dataset's __getitem__() returns a dict containing the keys "idx" and "symbols",
    # the batch dictionary returned by the DataLoader does not contain these keys because of the
    # custom collate functions.
    # """
    # train_loader, _ = get_dataloaders(
    #     [dataset_path],
    #     train_data_fraction=0.01,
    #     batch_size=10,
    #     batch_padded=False,
    #     seed=42,
    #     architecture="FCN",
    #     val_data_fraction=0.1,
    #     graph=False,
    #     factorize=False,
    # )
    # batch = next(iter(train_loader))
    # expected_keys = {
    #     "atomic_numbers",
    #     "positions",
    #     "distance_matrix",
    #     "energy",
    #     "forces",
    #     "stress",
    # }
    # self.assertTrue(
    #     expected_keys.issubset(batch.keys()),
    #     "Batch dictionary is missing keys in non-graph mode.",
    # )

    # # Check that "idx" and "symbols" are present when directly indexing the dataset
    # expected_keys.update({"idx", "symbols"})
    # x = dataset[0]
    # self.assertTrue(
    #     expected_keys.issubset(x.keys()),
    #     "Dataset dictionary is missing keys in non-graph mode.",
    # )

    def test_batch_keys_graph_true(self):
        """Test that the PyG Data object contains all expected attributes when graph=True."""
        train_loader, _ = get_dataloaders(
            [dataset_path],
            train_data_fraction=0.01,
            batch_size=10,
            seed=42,
            architecture="SchNet",
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

        dataset_1 = OMat24Dataset(
            dataset_paths=[dataset_path], architecture="FCN", debug=True
        )
        dataset_2 = OMat24Dataset(
            dataset_paths=[dataset_path_2], architecture="FCN", debug=True
        )

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
            architecture="FCN",
            val_data_fraction=val_data_fraction,
            graph=False,
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

    def test_rotation_90_degrees_four_times(self):
        """Test that rotating atomic positions, forces, and stresses by 90 degrees four times returns to the original values."""
        # Get a sample from the dataset
        sample = dataset[0]
        positions = sample.pos
        forces = sample.forces
        stress = sample.stress

        # Store original values
        original_positions = positions.clone()
        original_forces = forces.clone()
        original_stress = stress.clone()

        # Apply four 90-degree rotations around the z-axis
        rotated_positions = positions
        rotated_forces = forces
        rotated_stress = stress

        for _ in range(4):
            rotated_positions, R = rotate_atom_positions(
                rotated_positions, 90, axis=(0, 0, 1)
            )
            rotated_forces = rotated_forces @ R.T
            rotated_stress = rotate_stress(rotated_stress, R)

        # Check if we're back to the original positions (within numerical precision)
        max_pos_diff = torch.max(torch.abs(original_positions - rotated_positions))
        self.assertLess(
            max_pos_diff,
            1e-5,
            "Four 90-degree rotations didn't return to original positions",
        )

        # Check if we're back to the original forces (within numerical precision)
        max_forces_diff = torch.max(torch.abs(original_forces - rotated_forces))
        self.assertLess(
            max_forces_diff,
            1e-5,
            "Four 90-degree rotations didn't return to original forces",
        )

        # Check if we're back to the original stresses (within numerical precision)
        max_stress_diff = torch.max(torch.abs(original_stress - rotated_stress))
        self.assertLess(
            max_stress_diff,
            1e-5,
            "Four 90-degree rotations didn't return to original stress tensor",
        )

    def test_rotation_matrix_properties(self):
        """Test that the rotation matrix has proper mathematical properties."""
        # Get a sample from the dataset
        sample = dataset[0]
        positions = sample.pos

        # Test different angles
        for angle in [30, 45, 60, 90, 180]:
            _, R = rotate_atom_positions(positions, angle, axis=(0, 0, 1))

            # Test determinant is 1 (volume preserving)
            det = torch.det(R)
            self.assertAlmostEqual(
                det.item(),
                1.0,
                delta=1e-6,
                msg=f"Rotation matrix for {angle} degrees doesn't have determinant 1",
            )

            # Test orthogonality (R^T * R = I)
            I = R.T @ R
            expected = torch.eye(3, dtype=torch.float, device=positions.device)
            max_diff = torch.max(torch.abs(I - expected))
            self.assertLess(
                max_diff, 1e-5, f"Rotation matrix for {angle} degrees is not orthogonal"
            )

    def test_stress_rotation_properties(self):
        """Test that stress tensor rotations preserve tensor invariants."""
        # Get a sample from the dataset
        sample = dataset[0]
        stress = sample.stress

        # Handle the stress tensor properly whether it's 1D or 2D
        stress_unbatched = stress.squeeze(0) if stress.dim() > 1 else stress

        # Convert stress to 3x3 matrix for calculating invariants
        stress_matrix = torch.tensor(
            [
                [
                    stress_unbatched[0],
                    stress_unbatched[5],
                    stress_unbatched[4],
                ],  # xx, xy, xz
                [
                    stress_unbatched[5],
                    stress_unbatched[1],
                    stress_unbatched[3],
                ],  # xy, yy, yz
                [
                    stress_unbatched[4],
                    stress_unbatched[3],
                    stress_unbatched[2],
                ],  # xz, yz, zz
            ],
            dtype=torch.float,
        )

        # Calculate invariants before rotation
        trace_original = torch.trace(stress_matrix)
        det_original = torch.det(stress_matrix)

        # Test different angles
        for angle in [30, 45, 60, 90, 180]:
            # Create rotation matrix
            _, R = rotate_atom_positions(torch.zeros(1, 3), angle, axis=(0, 0, 1))

            # Rotate stress tensor
            rotated_stress = rotate_stress(stress, R)

            # Handle the rotated stress tensor properly whether it's 1D or 2D
            rotated_stress_unbatched = (
                rotated_stress.squeeze(0)
                if rotated_stress.dim() > 1
                else rotated_stress
            )

            # Convert rotated stress to 3x3 matrix
            rotated_stress_matrix = torch.tensor(
                [
                    [
                        rotated_stress_unbatched[0],
                        rotated_stress_unbatched[5],
                        rotated_stress_unbatched[4],
                    ],
                    [
                        rotated_stress_unbatched[5],
                        rotated_stress_unbatched[1],
                        rotated_stress_unbatched[3],
                    ],
                    [
                        rotated_stress_unbatched[4],
                        rotated_stress_unbatched[3],
                        rotated_stress_unbatched[2],
                    ],
                ],
                dtype=torch.float,
            )

            # Calculate invariants after rotation
            trace_rotated = torch.trace(rotated_stress_matrix)
            det_rotated = torch.det(rotated_stress_matrix)

            # Check that invariants are preserved
            self.assertAlmostEqual(
                trace_original.item(),
                trace_rotated.item(),
                delta=1e-5,
                msg=f"Trace not preserved for {angle} degree rotation",
            )

            self.assertAlmostEqual(
                det_original.item(),
                det_rotated.item(),
                delta=1e-5,
                msg=f"Determinant not preserved for {angle} degree rotation",
            )

    def test_random_rotation_preserves_distances(self):
        """Test that random rotations preserve distances between atoms."""
        # Get a sample from the dataset
        sample = dataset[0]
        positions = sample.pos

        # Calculate pairwise distances before rotation
        n_atoms = positions.shape[0]
        original_distances = torch.zeros((n_atoms, n_atoms), device=positions.device)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                original_distances[i, j] = torch.norm(positions[i] - positions[j])
                original_distances[j, i] = original_distances[i, j]

        # Apply random rotation
        rotated_positions, _ = random_rotate_atom_positions(positions)

        # Calculate pairwise distances after rotation
        rotated_distances = torch.zeros((n_atoms, n_atoms), device=positions.device)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                rotated_distances[i, j] = torch.norm(
                    rotated_positions[i] - rotated_positions[j]
                )
                rotated_distances[j, i] = rotated_distances[i, j]

        # Verify distances are preserved
        max_diff = torch.max(torch.abs(original_distances - rotated_distances))
        self.assertLess(
            max_diff, 1e-5, "Random rotation didn't preserve inter-atomic distances"
        )


if __name__ == "__main__":
    unittest.main()
