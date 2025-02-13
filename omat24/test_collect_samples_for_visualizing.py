import unittest
import torch
import numpy as np
import torch.nn as nn

from log_utils import collect_samples_for_visualizing


# Dummy dataset returns a fixed sample.
class DummyDataset:
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return {
            "idx": index,
            "symbols": ["H", "He", "Li", "Be"],
            "atomic_numbers": torch.tensor([1, 2, 0, 0], dtype=torch.long),
            "positions": torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=torch.float,
            ),
            "factorized_matrix": torch.tensor(
                [[0.1, 0.2], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float
            ),
            "forces": torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=torch.float,
            ),
            "energy": torch.tensor(1.0, dtype=torch.float),
            "stress": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float),
        }


# Dummy DataLoader that only provides a 'dataset' attribute.
class DummyDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset


# Dummy model that returns constant predictions.
class DummyModel(nn.Module):
    def forward(self, atomic_numbers, positions, factorized_distances, mask):
        batch_size, n_atoms = atomic_numbers.shape
        pred_forces = torch.ones((batch_size, n_atoms, 3))
        pred_energy = torch.full((batch_size,), 2.0)
        pred_stress = torch.ones((batch_size, 6))
        return pred_forces, pred_energy, pred_stress


class TestCollectTrainValSamples(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = DummyModel()
        self.train_dataset = DummyDataset(n_samples=2)
        self.val_dataset = DummyDataset(n_samples=2)
        self.train_loader = DummyDataLoader(self.train_dataset)
        self.val_loader = DummyDataLoader(self.val_dataset)
        self.num_visualization_samples = 3

    def test_structure_and_content(self):
        """Test that the collected samples have the expected structure and content."""
        samples = collect_samples_for_visualizing(
            self.model,
            self.train_loader,
            self.val_loader,
            self.device,
            self.num_visualization_samples,
        )
        # Expect 2 samples per split since dataset length is 2.
        self.assertEqual(len(samples["train"]), 2)
        self.assertEqual(len(samples["val"]), 2)

        for split in ["train", "val"]:
            for sample in samples[split]:
                # Check top-level keys.
                for key in [
                    "idx",
                    "symbols",
                    "atomic_numbers",
                    "positions",
                    "true",
                    "pred",
                ]:
                    self.assertIn(key, sample)

                # Integers can be compared directly.
                self.assertEqual(sample["atomic_numbers"], [1, 2])
                # Use numpy's allclose for float comparisons.
                np.testing.assert_allclose(
                    sample["positions"],
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    rtol=1e-5,
                    atol=1e-6,
                )

                true_vals = sample["true"]
                np.testing.assert_allclose(
                    true_vals["forces"],
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    rtol=1e-5,
                    atol=1e-6,
                )
                self.assertAlmostEqual(true_vals["energy"], 1.0, places=4)
                np.testing.assert_allclose(
                    true_vals["stress"],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    rtol=1e-5,
                    atol=1e-6,
                )

                pred_vals = sample["pred"]
                np.testing.assert_allclose(
                    pred_vals["forces"], [[1, 1, 1], [1, 1, 1]], rtol=1e-5, atol=1e-6
                )
                self.assertAlmostEqual(pred_vals["energy"], 2.0, places=4)
                np.testing.assert_allclose(
                    pred_vals["stress"], [1, 1, 1, 1, 1, 1], rtol=1e-5, atol=1e-6
                )

    def test_num_visualization_samples_limit(self):
        dataset = DummyDataset(5)
        loader = DummyDataLoader(dataset)
        samples = collect_samples_for_visualizing(
            self.model, loader, loader, self.device, 3
        )
        self.assertEqual(len(samples["train"]), 3)
        self.assertEqual(len(samples["val"]), 3)


if __name__ == "__main__":
    unittest.main()
