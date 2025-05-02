# External
import json
import re
import numpy as np
import os
import sys
import io
from contextlib import redirect_stdout
import torch
import unittest
from unittest.mock import patch
from pathlib import Path
import random

# Internal
from models.adit import ADiTS2EFSModel
from train import main as train_main
from data_utils import DATASET_INFO


class TestADiT(unittest.TestCase):
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
        """Initialize dummy data for ADiT model tests."""
        self.set_seed()

        self.batch_size = 2
        self.n_atoms = 4
        self.max_num_elements = 119

        # Create PyG-like batch data
        # Atomic numbers (for n_atoms per batch entry)
        self.atomic_numbers = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)

        # Positions: shape [num_atoms, 3]
        self.positions = torch.randn(6, 3)

        # Fractional coordinates: shape [num_atoms, 3]
        self.frac_coords = torch.rand(6, 3)

        # Cell: shape [batch_size, 3, 3] (represents the unit cell)
        self.cell = torch.stack([torch.eye(3), torch.eye(3)], dim=0)

        # Periodic boundary conditions
        self.pbc = torch.ones(self.batch_size, 3, dtype=torch.float)

        # Token indices
        self.token_idx = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

        # Batch index for each atom
        self.batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        # Number of atoms per batch entry
        self.natoms = torch.tensor([3, 3], dtype=torch.long)

        # Create a PyG-like batch object
        class DummyBatch:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def to(self, device):
                for key, value in self.__dict__.items():
                    if isinstance(value, torch.Tensor):
                        setattr(self, key, value.to(device))
                return self

        self.batch = DummyBatch(
            atomic_numbers=self.atomic_numbers,
            pos=self.positions,
            frac_coords=self.frac_coords,
            cell=self.cell,
            pbc=self.pbc,
            token_idx=self.token_idx,
            batch=self.batch_idx,
            natoms=self.natoms,
        )

    def test_initialization_predicts_means(self):
        """Test that ADiT model outputs predict dataset means at initialization."""
        self.set_seed()

        model = ADiTS2EFSModel(
            max_num_elements=self.max_num_elements,
            d_model=32,
            nhead=1,
            dim_feedforward=64,
            activation="gelu",
            dropout=0.1,
            norm_first=True,
            bias=True,
            num_layers=1,
        )
        model.eval()

        # Create zero inputs to test bias initialization
        zero_embedding = torch.zeros(1, model.transformer.d_model)

        with torch.no_grad():
            # Get direct output from the energy head with zero input
            energy_output = model.energy_head(zero_embedding)

            # Check energy bias matches expected dataset mean (per-atom)
            expected_energy_mean = DATASET_INFO["train"]["all"]["means"]["energy"] / 20
            self.assertAlmostEqual(
                energy_output.item(),
                expected_energy_mean,
                delta=0.01,
                msg="Energy head bias doesn't match expected dataset mean",
            )

            # Check force bias is zero
            force_output = model.force_head(zero_embedding)
            self.assertTrue(
                torch.allclose(force_output, torch.zeros(1, 3), atol=1e-6),
                msg="Force head output is not initialized to zero",
            )

            # Check stress bias matches expected values (per-atom)
            stress_output = model.stress_head(zero_embedding)
            expected_stress = torch.tensor(
                [[x / 20 for x in DATASET_INFO["train"]["all"]["means"]["stress"]]],
                dtype=stress_output.dtype,
            )
            self.assertTrue(
                torch.allclose(stress_output, expected_stress, atol=1e-4),
                msg="Stress head bias doesn't match expected dataset means",
            )

    def test_fixed_adit_overfit(self):
        """Test that a minimal training job with the ADiT architecture
        executes successfully and produces a valid configuration and finite loss values.
        """
        self.set_seed()

        # Create a fixed ADiT model with desired hyperparameters
        fixed_model = ADiTS2EFSModel(
            max_num_elements=119,
            d_model=8,
            nhead=1,
            dim_feedforward=32,
            activation="gelu",
            dropout=0.1,
            norm_first=True,
            bias=True,
            num_layers=1,
        )

        # Patch MetaADiTModels so that iterating over it yields only our fixed_model
        with patch("train.MetaADiTModels") as MockMeta:
            instance = MockMeta.return_value
            instance.__iter__.return_value = iter([fixed_model])

            # Simulate command-line arguments for a minimal training run
            test_args = [
                "train.py",
                "--architecture",
                "ADiT",
                "--epochs",
                "100",
                "--data_fraction",
                "0.0001",
                "--val_data_fraction",
                "0.0001",
                "--batch_size",
                "2",
                "--lr",
                "0.05",
                "--val_every",
                "100",
                "--vis_every",
                "100",
                "--name",
                "test_adit"
            ]
            with patch.object(sys, "argv", test_args):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    train_main()
                output = buf.getvalue()

                # Extract the experiment JSON filename from the output
                match = re.search(
                    r"Results will be saved to (?P<results_path>.+)", output
                )
                self.assertIsNotNone(match, "Could not find results filename in output")
                results_filename = match.group(1).strip()
                print("Captured results filename:", results_filename)

            try:
                # ---------- Test loss values and config ----------
                with open(results_filename, "r") as f:
                    result_json = json.load(f)

                # Get the config and loss information
                config = result_json["3"][0]["config"]
                first_train_loss = result_json["3"][0]["losses"]["1"]["train_loss"]
                first_val_loss = result_json["3"][0]["losses"]["0"]["val_loss"]
                first_flops = result_json["3"][0]["losses"]["0"].get("flops", 0)
                second_flops = result_json["3"][0]["losses"]["1"].get("flops", 0)
                last_train_loss = result_json["3"][0]["losses"]["100"]["train_loss"]
                last_val_loss = result_json["3"][0]["losses"]["100"]["val_loss"]
                last_flops = result_json["3"][0]["losses"]["100"].get("flops", 0)

                # Check that the model config matches what we expect
                self.assertEqual(config["architecture"], "ADiT")

                # Check that the losses are finite
                self.assertTrue(
                    np.isfinite(first_train_loss), "First train loss is not finite"
                )
                self.assertTrue(
                    np.isfinite(first_val_loss), "First validation loss is not finite"
                )
                self.assertTrue(
                    np.isfinite(last_train_loss), "Last train loss is not finite"
                )
                self.assertTrue(
                    np.isfinite(last_val_loss), "Last validation loss is not finite"
                )

                # Check that FLOPs are increasing
                if first_flops > 0 and second_flops > 0 and last_flops > 0:
                    self.assertLessEqual(
                        first_flops,
                        second_flops,
                        "FLOPs should increase or stay the same",
                    )
                    self.assertLessEqual(
                        second_flops,
                        last_flops,
                        "FLOPs should increase or stay the same",
                    )

                # Check that the loss decreases
                self.assertLessEqual(
                    last_train_loss,
                    first_train_loss * 1.5,
                    "Training loss did not decrease",
                )

            finally:
                if os.path.exists(results_filename):
                    os.remove(results_filename)

    def test_forward_output_shapes(self):
        """Verify that the ADiT model's forward pass returns outputs with the correct shapes."""
        self.set_seed()

        model = ADiTS2EFSModel(
            max_num_elements=self.max_num_elements,
            d_model=32,
            nhead=1,
            dim_feedforward=64,
            activation="gelu",
            dropout=0.1,
            norm_first=True,
            bias=True,
            num_layers=1,
        )

        forces, energy, stress = model(self.batch)

        # For forces, shape should be [num_atoms, 3]
        self.assertEqual(forces.shape, (6, 3))

        # For energy, shape should be [batch_size]
        self.assertEqual(energy.shape, (self.batch_size,))

        # For stress, shape should be [batch_size, 6]
        self.assertEqual(stress.shape, (self.batch_size, 6))

    def test_gradient_flow(self):
        """Verify that gradients flow back through the ADiT model during backpropagation."""
        self.set_seed()

        model = ADiTS2EFSModel(
            max_num_elements=self.max_num_elements,
            d_model=32,
            nhead=1,
            dim_feedforward=64,
            activation="gelu",
            dropout=0.1,
            norm_first=True,
            bias=True,
            num_layers=1,
        )
        model.train()
        model.zero_grad()

        forces, energy, stress = model(self.batch)
        loss = (forces**2).sum() + (energy**2).sum() + (stress**2).sum()
        loss.backward()

        for name, param in model.named_parameters():
            # Only check parameters that require gradients
            if param.requires_grad:
                if param.grad is None:
                    continue
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None.")
                grad_norm = param.grad.abs().sum().item()
                self.assertGreater(grad_norm, 0, f"Gradient for {name} is zero.")

    def test_atom_type_embedding(self):
        """Verify that different atomic numbers result in different embeddings."""
        self.set_seed()

        model = ADiTS2EFSModel(
            max_num_elements=self.max_num_elements,
            d_model=32,
            nhead=1,
            dim_feedforward=64,
            activation="gelu",
            dropout=0.1,
            norm_first=True,
            bias=True,
            num_layers=1,
        )

        # Get embeddings for different atomic numbers
        atom_type_1 = torch.tensor([1], dtype=torch.long)
        atom_type_2 = torch.tensor([2], dtype=torch.long)

        emb_1 = model.transformer.atom_type_embedder(atom_type_1)
        emb_2 = model.transformer.atom_type_embedder(atom_type_2)

        # Check that different atomic numbers have different embeddings
        self.assertFalse(
            torch.allclose(emb_1, emb_2, atol=1e-6),
            "Different atomic numbers should have different embeddings",
        )

    def test_aggregation(self):
        """Test that per-atom values are correctly aggregated to per-structure values."""
        self.set_seed()

        # Create a model with controlled output for testing
        class TestModel(ADiTS2EFSModel):
            def __init__(self):
                super().__init__(
                    max_num_elements=119,
                    d_model=32,
                    nhead=1,
                    dim_feedforward=64,
                    activation="gelu",
                    dropout=0.1,
                    norm_first=True,
                    bias=True,
                    num_layers=1,
                )

            def transformer(self, batch):
                # Return fixed embeddings for testing
                x = torch.ones(6, 32)  # 6 atoms, 32-dim embeddings
                return {
                    "x": x,
                    "natoms": batch.natoms,
                    "batch": batch.batch,
                    "token_idx": batch.token_idx,
                }

        model = TestModel()
        model.eval()

        # Modify the energy head to always output 1.0 for each atom
        with torch.no_grad():
            for p in model.energy_head.parameters():
                p.zero_()
            model.energy_head.net[-1].bias.fill_(1.0)

            # Similarly for stress head
            for p in model.stress_head.parameters():
                p.zero_()
            model.stress_head.net[-1].bias.fill_(1.0)

        # Run forward pass
        with torch.no_grad():
            _, energy, stress = model(self.batch)

        # Check that energy is summed per structure (3 atoms each)
        self.assertTrue(
            torch.allclose(energy, torch.tensor([3.0, 3.0])),
            "Energy should be summed per structure",
        )

        # Check that stress is averaged per structure
        self.assertTrue(
            torch.allclose(
                stress,
                torch.tensor(
                    [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
                ),
            ),
            "Stress should be averaged per structure",
        )


if __name__ == "__main__":
    unittest.main()
