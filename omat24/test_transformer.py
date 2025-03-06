# External
import subprocess
import json
import re
import numpy as np
import os
import sys
import shutil
import io
from contextlib import redirect_stdout
import torch
import unittest
from unittest.mock import patch
from pathlib import Path
import random

# Internal
from models.transformer_models import XTransformerModel
from train import main as train_main


class TestTransformer(unittest.TestCase):
    def set_seed(self):
        SEED = 1024
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cpu")

    def setUp(self):
        """Initialize dummy data for TransformerModel tests."""
        self.set_seed()

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

    def test_fixed_transformer_overfit(self):
        """Test that a minimal training job with the Transformer architecture
        executes successfully and produces a valid configuration and finite loss values.
        """
        self.set_seed()

        # Create a fixed transformer model with desired hyperparameters.
        fixed_model = XTransformerModel(
            num_tokens=119,
            d_model=1,
            depth=1,
            n_heads=1,
            d_ff_mult=1,
            use_factorized=False,
        )
        # Patch MetaTransformerModels so that iterating over it yields only our fixed_model.
        with patch("train.MetaTransformerModels") as MockMeta:
            instance = MockMeta.return_value
            instance.__iter__.return_value = iter([fixed_model])

            # Simulate command-line arguments for a minimal training run.
            test_args = [
                "train.py",
                "--architecture",
                "Transformer",
                "--epochs",
                "500",
                "--data_fraction",
                "0.0001",
                "--val_data_fraction",
                "0.0001",
                "--batch_size",
                "2",
                "--lr",
                "0.05",
                "--val_every",
                "500",
                "--vis_every",
                "500",
            ]
            with patch.object(sys, "argv", test_args):
                # Patch subprocess.run inside train.py to intercept the call to model_prediction_evolution.py.
                # This prevents the production code from trying to read an empty results file.
                with patch("train.subprocess.run") as mock_subproc_run:
                    mock_subproc_run.return_value = subprocess.CompletedProcess(
                        args=["python3", "model_prediction_evolution.py"],
                        returncode=0,
                        stdout="dummy output",
                        stderr="",
                    )
                    # Capture stdout from train_main() to retrieve the generated results filename.

                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        train_main()
                    output = buf.getvalue()

                    # Extract the experiment JSON filename from the output.
                    match = re.search(
                        r"Results will be saved to (?P<results_path>.+)", output
                    )
                    self.assertIsNotNone(
                        match, "Could not find results filename in output"
                    )
                    results_filename = match.group(1).strip()
                    print("Captured results filename:", results_filename)

            visualization_filepath = None
            try:
                # ---------- Test loss values and config ----------
                with open(results_filename, "r") as f:
                    result_json = json.load(f)

                # Get the config and loss information
                config = result_json["3"][0]["config"]
                first_train_loss = result_json["3"][0]["losses"]["1"]["train_loss"]
                first_val_loss = result_json["3"][0]["losses"]["0"]["val_loss"]
                first_flops = result_json["3"][0]["losses"]["0"]["flops"]
                second_flops = result_json["3"][0]["losses"]["1"]["flops"]
                last_train_loss = result_json["3"][0]["losses"]["500"]["train_loss"]
                last_val_loss = result_json["3"][0]["losses"]["500"]["val_loss"]
                last_flops = result_json["3"][0]["losses"]["500"]["flops"]

                # For the Transformer, the first configuration (from MetaTransformerModels) is expected to be:
                self.assertEqual(config["embedding_dim"], 1)
                self.assertEqual(config["depth"], 1)
                self.assertEqual(config["num_params"], 1670)

                np.testing.assert_allclose(first_train_loss, 1029.019196, rtol=0.1)
                np.testing.assert_allclose(first_val_loss, 210.055866, rtol=0.1)
                np.testing.assert_allclose(first_flops, 0, rtol=0.1)
                np.testing.assert_allclose(second_flops, 65028096, rtol=0.1)
                np.testing.assert_allclose(last_train_loss, 219.792595, rtol=0.1)
                np.testing.assert_allclose(last_flops, 32514048000, rtol=0.1)
                if os.getenv("IS_CI", False):
                    np.testing.assert_allclose(last_val_loss, 2181.785034, rtol=0.1)
                else:
                    np.testing.assert_allclose(last_val_loss, 1198.087219, rtol=0.1)

                # ---------- Test visualization was created ----------
                result = subprocess.run(
                    [
                        "python3",
                        "model_prediction_evolution.py",
                        str(results_filename),
                        "--split",
                        "train",
                    ],
                    capture_output=True,
                    text=True,
                )

                visualization_filepath = Path(f"figures/{Path(results_filename).stem}")
                assert visualization_filepath.exists(), "Visualization was not created."

            finally:
                if os.path.exists(results_filename):
                    os.remove(results_filename)
                if visualization_filepath and os.path.exists(visualization_filepath):
                    shutil.rmtree(visualization_filepath)

    def test_forward_non_factorized_output_shapes(self):
        """
        Verify that the TransformerModel's forward pass in non-factorized mode
        returns outputs with the correct shapes.
        """
        self.set_seed()

        # Use non-factorized mode: positions are provided.
        model = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=6,
            depth=4,
            n_heads=2,
            d_ff_mult=2,
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
        self.set_seed()

        model = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=6,
            depth=4,
            n_heads=2,
            d_ff_mult=2,
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
        self.set_seed()

        model = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=6,
            depth=4,
            n_heads=2,
            d_ff_mult=2,
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
        self.set_seed()

        model = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=6,
            depth=4,
            n_heads=2,
            d_ff_mult=2,
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
        self.set_seed()

        model_factorized = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=4,
            depth=2,
            n_heads=2,
            d_ff_mult=2,
            use_factorized=True,
        )
        model_non_factorized = XTransformerModel(
            num_tokens=self.vocab_size,
            d_model=4,
            depth=2,
            n_heads=2,
            d_ff_mult=2,
            use_factorized=False,
        )

        # For factorized mode: additional_dim should be 5, so input = d_model + 5.
        # For non-factorized mode: additional_dim should be 3, so input = d_model + 3.
        expected_in_features_factorized = 4 + 5
        expected_in_features_non_factorized = 4 + 3

        self.assertEqual(
            model_factorized.forces_output.in_features,
            expected_in_features_factorized,
            "Factorized mode forces_output input feature size incorrect.",
        )
        self.assertEqual(
            model_non_factorized.forces_output.in_features,
            expected_in_features_non_factorized,
            "Non-factorized mode forces_output input feature size incorrect.",
        )
