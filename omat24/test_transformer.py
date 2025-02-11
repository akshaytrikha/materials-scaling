import subprocess
import json
import re
import numpy as np
import os
import torch
import unittest
from models.transformer_models import XTransformerModel, MetaTransformerModels


class TestTransformer(unittest.TestCase):
    def setUp(self):
        """Initialize dummy data for TransformerModel tests."""
        self.batch_size = 2
        self.n_atoms = 4
        self.vocab_size = 119

        # Create dummy atomic numbers (0 indicates a padded atom)
        self.atomic_numbers = torch.tensor(
            [[1, 2, 3, 0], [4, 5, 6, 0]], dtype=torch.long
        )

        # Random positions: shape [batch_size, n_atoms, 3]
        self.positions = torch.randn(self.batch_size, self.n_atoms, 3)

        # Dummy factorized distance matrix: shape [batch_size, n_atoms, 5]
        self.distance_matrix = torch.randn(self.batch_size, self.n_atoms, 5)

        # Mask for valid (nonzero) atoms
        self.mask = self.atomic_numbers != 0

    def test_overfit_one_sample(self):
        """Test that a minimal training job with the Transformer architecture
        executes successfully and produces a valid configuration and finite loss values.
        """
        result = subprocess.run(
            [
                "python3",
                "train.py",
                "--architecture",
                "Transformer",
                "--epochs",
                "500",
                "--data_fraction",
                "0.00001",
                "--val_data_fraction",
                "0.001",
                "--batch_size",
                "1",
                "--lr",
                "0.05",
                "--val_every",
                "500",
                "--vis_every",
                "500",
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

        # Ground truth dimensions
        DIMS = [
            {
                "embedding_dim": 1,
                "depth": 1,
                "num_params": 1729,
            },
            {
                "embedding_dim": 4,
                "depth": 2,
                "num_params": 9061,
            },
            {
                "embedding_dim": 8,
                "depth": 8,
                "num_params": 109059,
            },
        ]

        # Ground truth loss values
        EXPECTED_LOSSES = [
            {
                "first_train_loss": 11.630836486816406,
                "first_val_loss": 25.706957708086286,
                "last_train_loss": 0.15662512183189392,
                "last_val_loss": 15.45500339099339,
            },
            {
                "first_train_loss": 8.382429122924805,
                "first_val_loss": 21.924051897866384,
                "last_train_loss": 0.027594711631536484,
                "last_val_loss": 14.812024521827698,
            },
            {
                "first_train_loss": 8.800992965698242,
                "first_val_loss": 21.239415659223283,
                "last_train_loss": 0.6234482526779175,
                "last_val_loss": 14.360544357981,
            },
        ]

        try:
            with open(results_path, "r") as f:
                result_json = json.load(f)

            for i in range(3):
                # Get the config and loss information
                config = result_json["1"][i]["config"]
                first_val_loss = result_json["1"][i]["losses"]["0"]["val_loss"]
                first_train_loss = result_json["1"][i]["losses"]["1"]["train_loss"]
                last_val_loss = result_json["1"][i]["losses"]["500"]["val_loss"]
                last_train_loss = result_json["1"][i]["losses"]["500"]["train_loss"]

                # For the Transformer, the first configuration (from MetaTransformerModels) is expected to be:
                self.assertEqual(config["embedding_dim"], DIMS[i]["embedding_dim"])
                self.assertEqual(config["depth"], DIMS[i]["depth"])
                self.assertEqual(config["num_params"], DIMS[i]["num_params"])

                np.testing.assert_allclose(
                    first_train_loss, EXPECTED_LOSSES[i]["first_train_loss"], rtol=0.1
                )
                np.testing.assert_allclose(
                    first_val_loss, EXPECTED_LOSSES[i]["first_val_loss"], rtol=0.1
                )
                np.testing.assert_allclose(
                    last_train_loss, EXPECTED_LOSSES[i]["last_train_loss"], rtol=0.1
                )
                np.testing.assert_allclose(
                    last_val_loss, EXPECTED_LOSSES[i]["last_val_loss"], rtol=0.1
                )
        finally:
            if os.path.exists(results_path):
                os.remove(results_path)

    def test_forward_non_factorized_output_shapes(self):
        """
        Verify that the TransformerModel's forward pass in non-factorized (concatenated) mode
        returns outputs with the correct shapes.
        """
        # Use concatenated mode (non-factorized): positions are provided.
        model = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=6,
            depth=4,
            n_heads=2,
            d_ff_mult=2,
            concatenated=True,
            use_factorized=False,
        )
        forces, energy, stress = model(
            self.atomic_numbers, self.positions, distance_matrix=None, mask=self.mask
        )
        self.assertEqual(forces.shape, (self.batch_size, self.n_atoms, 3))
        self.assertEqual(energy.shape, (self.batch_size,))
        self.assertEqual(stress.shape, (self.batch_size, 6))

    def test_forward_factorized_output_shapes(self):
        """Verify that the TransformerModel's forward pass in factorized mode
        returns outputs with the correct shapes."""
        model = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=6,
            depth=4,
            n_heads=2,
            d_ff_mult=2,
            concatenated=True,
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
          - Verifying that outputs for padded atoms are zero.
          - Ensuring that a fully padded input yields zero outputs.
          - Checking that a mismatched input shape raises an exception.
        """
        model = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=6,
            depth=4,
            n_heads=2,
            d_ff_mult=2,
            concatenated=True,
            use_factorized=False,
        )
        model.eval()
        with torch.no_grad():
            forces, energy, stress = model(
                self.atomic_numbers,
                self.positions,
                distance_matrix=None,
                mask=self.mask,
            )
            # Check that forces for padded atoms are zero.
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

            # Mismatched input shapes: positions tensor has a wrong shape.
            wrong_positions = torch.randn(self.batch_size, self.n_atoms + 1, 3)
            with self.assertRaises(Exception):
                _ = model(
                    self.atomic_numbers,
                    wrong_positions,
                    distance_matrix=None,
                    mask=self.mask,
                )

    def test_gradient_flow(self):
        """Verify that gradients flow back through the Transformer model during backpropagation."""
        model = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=6,
            depth=4,
            n_heads=2,
            d_ff_mult=2,
            concatenated=True,
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
            # Only check parameters that require gradients.
            if param.requires_grad:
                if "project_emb" in name or "to_logits" in name:
                    continue
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None.")
                grad_norm = param.grad.abs().sum().item()
                self.assertGreater(grad_norm, 0, f"Gradient for {name} is zero.")

    def test_factorized_vs_non_factorized(self):
        """
        Verify that the force prediction layer's input feature size differs between factorized
        and non-factorized modes.
        """
        model_factorized = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=4,
            depth=2,
            n_heads=2,
            d_ff_mult=2,
            concatenated=False,
            use_factorized=True,
        )
        model_non_factorized = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=4,
            depth=2,
            n_heads=2,
            d_ff_mult=2,
            concatenated=True,
            use_factorized=False,
        )
        # For factorized mode: additional_dim should be 5, so input = d_model + 5.
        # For non-factorized (concatenated) mode: additional_dim should be 3, so input = d_model + 3.
        expected_in_features_factorized = 4 + 5
        expected_in_features_non_factorized = 4 + 3

        self.assertEqual(
            model_factorized.force_output.in_features,
            expected_in_features_factorized,
            "Factorized mode force_output input feature size incorrect.",
        )
        self.assertEqual(
            model_non_factorized.force_output.in_features,
            expected_in_features_non_factorized,
            "Non-factorized mode force_output input feature size incorrect.",
        )

    def test_meta_model_iteration(self):
        """
        Test that iterating over MetaTransformerModels yields the expected number of models,
        and that each returned model has the expected parameter count.
        Expected parameter counts (as per comments): [1848, 9537, 110011]
        """
        meta_models = MetaTransformerModels(
            vocab_size=self.vocab_size,
            max_seq_len=10,
            concatenated=True,
            use_factorized=False,
        )
        models_list = list(iter(meta_models))
        self.assertEqual(
            len(models_list),
            len(meta_models),
            "Number of models returned by iteration does not match expected count.",
        )
        expected_params = [1729, 9061, 109059]
        for model, expected in zip(models_list, expected_params):
            self.assertEqual(
                model.num_params,
                expected,
                f"Model parameter count mismatch: expected {expected}, got {model.num_params}",
            )
