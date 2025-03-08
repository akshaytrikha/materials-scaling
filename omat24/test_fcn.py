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
from torch import nn

# Internal
from models.fcn import FCNModel, MetaFCNModels
from train import main as train_main


# A helper module that always returns zeros, to “remove” the effect of inner layers.
class ZeroLayer(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class TestFCN(unittest.TestCase):
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
        """Initialize dummy data for FCNModel tests, including atomic numbers, positions, distance matrix, and mask."""
        self.set_seed()

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

    def test_fixed_fcn_overfit(self):
        """Test that a minimal training job executes successfully and produces expected configuration and loss values."""
        self.set_seed()

        # Create a fixed FCN model with desired hyperparameters.
        fixed_model = FCNModel(
            vocab_size=119,
            embedding_dim=2,
            hidden_dim=2,
            depth=2,
            use_factorized=False,
        )

        # Patch MetaFCNModels so that iterating over it yields only our fixed_model.
        with patch("train.MetaFCNModels") as MockMeta:
            instance = MockMeta.return_value
            instance.__iter__.return_value = iter([fixed_model])

            test_args = [
                "train.py",
                "--architecture",
                "FCN",
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
                last_train_loss = result_json["3"][0]["losses"]["500"]["train_loss"]
                last_val_loss = result_json["3"][0]["losses"]["500"]["val_loss"]

                # For the FCN, the first configuration (from MetaFCNModels) is expected to be:
                self.assertEqual(config["embedding_dim"], 2)
                self.assertEqual(config["depth"], 2)
                self.assertEqual(config["num_params"], 62)

                np.testing.assert_allclose(
                    first_train_loss, 299.64583587646484, rtol=0.1
                )
                np.testing.assert_allclose(first_val_loss, 292.1522979736328, rtol=0.1)
                np.testing.assert_allclose(
                    last_train_loss, 137.36277389526367, rtol=0.1
                )
                if os.getenv("IS_CI", False):
                    np.testing.assert_allclose(last_val_loss, 129.64451027, rtol=0.1)
                else:
                    np.testing.assert_allclose(last_val_loss, 108.93801498413086, rtol=0.1)

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

        meta_models.configurations = [
            # 138 params
            {"embedding_dim": 4, "hidden_dim": 4, "depth": 2},
            # 1274 params
            {"embedding_dim": 8, "hidden_dim": 16, "depth": 3},
            # 99338 params
            {"embedding_dim": 64, "hidden_dim": 64, "depth": 22},
        ]

        models_list = list(iter(meta_models))
        self.assertEqual(
            len(models_list),
            len(meta_models),
            "Number of models returned by iteration does not match expected count.",
        )
        expected_params = [138, 1274, 99338]
        for model, expected in zip(models_list, expected_params):
            self.assertEqual(
                model.num_params,
                expected,
                f"Model parameter count mismatch: expected {expected}, got {model.num_params}",
            )
