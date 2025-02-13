# External
import subprocess
import json
import re
import numpy as np
import os
import torch
import torch.nn as nn
import unittest

# Internal
from models.fcn import FCNModel, MetaFCNModels


# A helper module that always returns zeros, to “remove” the effect of inner layers.
class ZeroLayer(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class TestFCN(unittest.TestCase):
    def setUp(self):
        """Initialize dummy data for FCNModel tests, including atomic numbers, positions, distance matrix, and mask."""
        # Define a small dummy batch.
        self.batch_size = 2
        self.n_atoms = 4
        self.vocab_size = 119

        # Create dummy atomic numbers.
        # Here, a zero indicates a padded atom.
        self.atomic_numbers = torch.tensor(
            [[1, 2, 3, 0], [4, 5, 6, 0]], dtype=torch.long
        )

        # Random positions: shape [batch_size, n_atoms, 3]
        self.positions = torch.randn(self.batch_size, self.n_atoms, 3)

        # For factorized mode, dummy distance matrix: shape [batch_size, n_atoms, 5]
        self.distance_matrix = torch.randn(self.batch_size, self.n_atoms, 5)

        # Mask: valid (nonzero) atoms are True.
        self.mask = self.atomic_numbers != 0

    def test_train_job(self):
        """Test that a minimal training job executes successfully and produces expected configuration and loss values."""
        result = subprocess.run(
            [
                "python3",
                "train.py",
                "--architecture",
                "FCN",
                "--epochs",
                "1",
                "--data_fraction",
                "0.00001",
                "--val_data_fraction",
                "0.001",
                "--batch_size",
                "1",
                "--lr",
                "0.001",
                "--vis_every",
                "1",
            ],
            capture_output=True,
            text=True,
        )

        # Extract results path from stdout
        match = re.search(
            r"Results will be saved to (?P<results_path>.+)", result.stdout
        )
        self.assertIsNotNone(match, "Results path not found in output")
        results_path = match.group("results_path")

        try:
            with open(results_path, "r") as f:
                result_json = json.load(f)
            config = result_json["1"][0]["config"]
            first_val_loss = result_json["1"][0]["losses"]["0"]["val_loss"]
            first_train_loss = result_json["1"][0]["losses"]["1"]["train_loss"]

            self.assertEqual(config["embedding_dim"], 6)
            self.assertEqual(config["depth"], 4)
            self.assertEqual(config["num_params"], 1060)

            np.testing.assert_allclose(first_train_loss, 21.387577056884766, rtol=0.1)
            np.testing.assert_allclose(first_val_loss, 38.86819725036621, rtol=0.1)
        finally:
            if os.path.exists(results_path):
                os.remove(results_path)

    def test_forward_non_factorized_output_shapes(self):
        """Verify that the FCNModel's forward pass without factorized distances returns outputs with correct shapes."""
        model = FCNModel(
            vocab_size=self.vocab_size,
            embedding_dim=6,
            hidden_dim=6,
            depth=4,
            use_factorized=False,
        )
        forces, energy, stress = model(
            self.atomic_numbers, self.positions, distance_matrix=None, mask=self.mask
        )

        self.assertEqual(forces.shape, (self.batch_size, self.n_atoms, 3))
        self.assertEqual(energy.shape, (self.batch_size,))
        self.assertEqual(stress.shape, (self.batch_size, 6))

    def test_forward_factorized_output_shapes(self):
        """Verify that the FCNModel's forward pass with factorized distances returns outputs with correct shapes."""
        model = FCNModel(
            vocab_size=self.vocab_size,
            embedding_dim=6,
            hidden_dim=6,
            depth=4,
            use_factorized=True,
        )
        forces, energy, stress = model(
            self.atomic_numbers,
            self.positions,
            distance_matrix=self.distance_matrix,
            mask=self.mask,
        )

        self.assertEqual(forces.shape, (self.batch_size, self.n_atoms, 3))
        self.assertEqual(energy.shape, (self.batch_size,))
        self.assertEqual(stress.shape, (self.batch_size, 6))

    def test_masking_and_edge_cases(self):
        """
        Test model behavior for masked atoms and edge-case inputs.
        This includes:
          - Verifying that outputs corresponding to padded atoms are zero.
          - Ensuring that a fully padded input yields zero outputs.
          - Checking that a mismatched input shape (positions tensor) raises an exception.
        """
        model = FCNModel(
            vocab_size=self.vocab_size,
            embedding_dim=6,
            hidden_dim=6,
            depth=4,
            use_factorized=False,
        )
        model.eval()
        with torch.no_grad():
            # Verify that outputs for padded atoms are zero.
            forces, energy, stress = model(
                self.atomic_numbers,
                self.positions,
                distance_matrix=None,
                mask=self.mask,
            )
            for i in range(self.batch_size):
                for j in range(self.n_atoms):
                    if self.atomic_numbers[i, j] == 0:
                        self.assertTrue(
                            torch.allclose(forces[i, j], torch.zeros(3), atol=1e-6),
                            f"Force at batch {i}, atom {j} is not zero.",
                        )

            # Fully padded input: all atomic numbers are zero.
            all_padded = torch.zeros_like(self.atomic_numbers)
            all_mask = all_padded != 0
            forces_pad, energy_pad, stress_pad = model(
                all_padded, self.positions, distance_matrix=None, mask=all_mask
            )
            self.assertTrue(
                torch.allclose(forces_pad, torch.zeros_like(forces_pad), atol=1e-6),
                "Forces are not zero for a fully padded input.",
            )
            self.assertTrue(
                torch.allclose(energy_pad, torch.zeros_like(energy_pad), atol=1e-6),
                "Energy is not zero for a fully padded input.",
            )
            self.assertTrue(
                torch.allclose(stress_pad, torch.zeros_like(stress_pad), atol=1e-6),
                "Stress is not zero for a fully padded input.",
            )

            # Mismatched input shapes: positions tensor does not match atomic_numbers.
            wrong_positions = torch.randn(self.batch_size, self.n_atoms + 1, 3)
            with self.assertRaises(Exception):
                _ = model(
                    self.atomic_numbers,
                    wrong_positions,
                    distance_matrix=None,
                    mask=self.mask,
                )

    def test_gradient_flow(self):
        """Verify that gradients flow back through the model during backpropagation."""
        model = FCNModel(
            vocab_size=self.vocab_size,
            embedding_dim=6,
            hidden_dim=6,
            depth=4,
            use_factorized=False,
        )
        model.train()
        model.zero_grad()
        forces, energy, stress = model(
            self.atomic_numbers, self.positions, distance_matrix=None, mask=self.mask
        )
        loss = (forces**2).sum() + (energy**2).sum() + (stress**2).sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None.")
                grad_norm = param.grad.abs().sum().item()
                self.assertGreater(grad_norm, 0, f"Gradient for {name} is zero.")

    def test_residual_connections_effect(self):
        """Test that the residual connections in the inner layers have an effect.
        This is done by comparing the output of a normal model versus a model where
        the inner layers are replaced with zero-output layers.
        """
        model_normal = FCNModel(
            vocab_size=self.vocab_size,
            embedding_dim=6,
            hidden_dim=6,
            depth=4,
            use_factorized=False,
        )
        model_no_res = FCNModel(
            vocab_size=self.vocab_size,
            embedding_dim=6,
            hidden_dim=6,
            depth=4,
            use_factorized=False,
        )
        model_no_res.inner_layers = nn.ModuleList(
            [ZeroLayer() for _ in range(model_no_res.depth)]
        )

        model_normal.eval()
        model_no_res.eval()
        with torch.no_grad():
            forces_normal, energy_normal, stress_normal = model_normal(
                self.atomic_numbers,
                self.positions,
                distance_matrix=None,
                mask=self.mask,
            )
            forces_no_res, energy_no_res, stress_no_res = model_no_res(
                self.atomic_numbers,
                self.positions,
                distance_matrix=None,
                mask=self.mask,
            )
        self.assertFalse(
            torch.allclose(forces_normal, forces_no_res, atol=1e-6),
            "Forces outputs are identical; residual connections may not be effective.",
        )
        self.assertFalse(
            torch.allclose(energy_normal, energy_no_res, atol=1e-6),
            "Energy outputs are identical; residual connections may not be effective.",
        )
        self.assertFalse(
            torch.allclose(stress_normal, stress_no_res, atol=1e-6),
            "Stress outputs are identical; residual connections may not be effective.",
        )

    def test_factorized_vs_non_factorized(self):
        """Verify that the model's first linear layer (fc1) has the correct input feature size for factorized and non-factorized modes."""
        model_factorized = FCNModel(
            vocab_size=self.vocab_size,
            embedding_dim=6,
            hidden_dim=6,
            depth=4,
            use_factorized=True,
        )
        model_non_factorized = FCNModel(
            vocab_size=self.vocab_size,
            embedding_dim=6,
            hidden_dim=6,
            depth=4,
            use_factorized=False,
        )
        self.assertEqual(
            model_factorized.fc1.in_features,
            6 + 5,
            "Factorized mode fc1 input feature size incorrect.",
        )
        self.assertEqual(
            model_non_factorized.fc1.in_features,
            6 + 3,
            "Non-factorized mode fc1 input feature size incorrect.",
        )

    def test_meta_model_iteration(self):
        """
        Test that iterating over MetaFCNModels yields the expected number of models,
        and that each returned model has the correct configuration (e.g. parameter count).
        """
        meta_models = MetaFCNModels(vocab_size=self.vocab_size, use_factorized=False)
        models_list = list(iter(meta_models))
        self.assertEqual(
            len(models_list),
            len(meta_models),
            "Number of models returned by iteration does not match expected count.",
        )
        expected_params = [1060, 9770, 98378]
        for model, expected in zip(models_list, expected_params):
            self.assertEqual(
                model.num_params,
                expected,
                f"Model parameter count mismatch: expected {expected}, got {model.num_params}",
            )
