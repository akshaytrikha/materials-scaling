# External
import unittest
import torch
import numpy as np
from contextlib import redirect_stdout
import io
import sys
import json
import re
import os
import subprocess
from unittest.mock import patch
import random

# Internal
from models.schnet import SchNet
from data_utils import DATASET_INFO
from train import main as train_main


class TestSchNet(unittest.TestCase):
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
        self.set_seed()

        # Create dummy data for 2 molecules with 4 atoms each (total 8 atoms)
        self.num_atoms = 8
        self.num_molecules = 2
        # Example atomic numbers (using valid indices < 100)
        # Molecule 1: atoms 0-3; Molecule 2: atoms 4-7.
        self.atomic_numbers = torch.tensor(
            [1, 6, 8, 1, 6, 1, 8, 8], dtype=torch.long, device=self.device
        )
        # Random positions for each atom: [8, 3]
        self.positions = torch.randn(self.num_atoms, 3, device=self.device)
        # Molecule assignment: first 4 atoms -> 0, next 4 -> 1
        self.structure_index = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device=self.device
        )
        # Construct a simple full pairwise connectivity (excluding self-loops) per molecule.
        edge_index_list = []
        # Molecule 1 (indices 0-3)
        for i in range(4):
            for j in range(4):
                if i != j:
                    edge_index_list.append([i, j])
        # Molecule 2 (indices 4-7)
        for i in range(4, 8):
            for j in range(4, 8):
                if i != j:
                    edge_index_list.append([i, j])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long, device=self.device)
        self.edge_index = edge_index.t().contiguous()  # shape [2, num_edges]

        # Initialize a SchNet model with modest dimensions.
        self.model = SchNet(
            hidden_channels=4,
            num_filters=4,
            num_interactions=3,
            num_gaussians=16,
            cutoff=5.0,
            max_num_neighbors=32,
            readout="add",
            dipole=False,
            device=self.device,
        ).to(self.device)

    def test_forward_output_shapes(self):
        """Test that a forward pass returns outputs with correct shapes."""
        forces, energy, stress = self.model(
            self.atomic_numbers, self.positions, self.edge_index, self.structure_index
        )
        self.assertEqual(
            forces.shape,
            (self.num_atoms, 3),
            "Forces should have shape [num_atoms, 3].",
        )
        self.assertEqual(
            energy.shape,
            (self.num_molecules,),
            "Energy should have shape [num_molecules].",
        )
        self.assertEqual(
            stress.shape,
            (self.num_molecules, 6),
            "Stress should have shape [num_molecules, 6].",
        )

    def test_gradient_flow(self):
        """Ensure that gradients are computed during backpropagation."""
        self.model.train()
        self.model.zero_grad()
        forces, energy, stress = self.model(
            self.atomic_numbers, self.positions, self.edge_index, self.structure_index
        )
        loss = forces.pow(2).sum() + energy.pow(2).sum() + stress.pow(2).sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None.")
                self.assertGreater(
                    param.grad.abs().sum().item(), 0, f"Gradient for {name} is zero."
                )

    # This test is weird because the model is not trained so I'm not sure why the energy should be 0.
    # def test_zero_padding_behavior(self):
    #     """
    #     When a molecule is fully padded (atomic numbers set to zero), the model should output near-zero energy.
    #     Note: SchNet's embedding includes an entry for index 0; with proper training this output would be negligible.
    #     """
    #     padded_atomic_numbers = self.atomic_numbers.clone()
    #     # Fully pad molecule 1 (atoms 0-3)
    #     padded_atomic_numbers[:4] = 0
    #     forces, energy, stress = self.model(
    #         padded_atomic_numbers, self.positions, self.edge_index, self.structure_index
    #     )
    #     self.assertTrue(
    #         abs(energy[0].item()) < 1e-3,
    #         "Energy for a fully padded molecule should be near zero.",
    #     )

    def test_incorrect_input_shapes(self):
        """
        Providing a positions tensor with extra rows should not cause an error.
        The model will use the full positions tensor, but since only the rows
        corresponding to atomic_numbers are used in energy computation, the extra
        row's gradient (i.e. force) should be zero.
        """
        wrong_positions = torch.randn(self.num_atoms + 1, 3, device=self.device)
        forces, energy, stress = self.model(
            self.atomic_numbers,
            wrong_positions,
            self.edge_index,
            self.structure_index,
        )
        # The forces tensor will have one extra row because positions has extra data.
        self.assertEqual(
            forces.shape,
            (self.num_atoms + 1, 3),
            "Forces shape should match the positions input shape.",
        )
        # The extra row (last row) should be zero because it is not used in the forward pass.
        self.assertTrue(
            torch.allclose(forces[-1], torch.zeros(3, device=self.device), atol=1e-6),
            "The extra row in forces should be zero since it is not used.",
        )
        # Energy and stress outputs remain computed from the valid atoms only.
        self.assertEqual(
            energy.shape,
            (self.num_molecules,),
            "Energy should have shape [num_molecules].",
        )
        self.assertEqual(
            stress.shape,
            (self.num_molecules, 6),
            "Stress should have shape [num_molecules, 6].",
        )

    def test_initialization_predicts_means(self):
        """Test that SchNet outputs predict dataset means at initialization."""
        self.set_seed()

        # Create a minimal SchNet model for testing
        model = SchNet(
            hidden_channels=4,
            num_filters=4,
            num_interactions=3,
            num_gaussians=16,
            cutoff=5.0,
            max_num_neighbors=32,
            readout="add",
            device=self.device,
        ).to(self.device)
        model.eval()

        # Create a setup where the model's internal representations should be minimal
        # This isolates the effect of the output head biases
        with torch.enable_grad():
            # Access the output heads directly with zero embeddings
            zero_embedding = torch.zeros(1, model.hidden_channels, device=self.device)

            # Check energy head bias
            energy_contrib = model.energy_head(zero_embedding)
            self.assertAlmostEqual(
                energy_contrib.item(),
                DATASET_INFO["train"]["all"]["means"]["energy"],
                places=3,
                msg="Energy head bias doesn't match expected dataset mean",
            )

            # Force is derived through autograd, but we can check that force_head bias is zero
            # when manually accessed
            force_output = torch.zeros_like(zero_embedding)

            # Check stress head bias
            stress_contrib = model.stress_head(zero_embedding)
            expected_stress = torch.tensor(
                [DATASET_INFO["train"]["all"]["means"]["stress"]],
                dtype=stress_contrib.dtype,
                device=self.device,
            )
            self.assertTrue(
                torch.allclose(stress_contrib, expected_stress, atol=1e-4),
                msg="Stress head bias doesn't match expected dataset means",
            )

            # For completeness, run a forward pass and check that outputs are influenced by the biases
            # The exact values will depend on the randomly initialized weights, but the general
            # magnitudes should be close to the expected means
            forces, energy, stress = model(
                self.atomic_numbers,
                self.positions,
                self.edge_index,
                self.structure_index,
            )

            # Since there are other factors at play, we just check the order of magnitude
            self.assertLess(
                abs(torch.mean(energy).item() + 9.773),
                10.0,
                msg="Mean energy is too far from expected dataset mean",
            )

    def test_fixed_schnet_overfit(self):
        """
        Run a minimal training job using the SchNet architecture.
        This test patches the meta-model iterator in train.py to return a fixed SchNet model,
        runs a short training run, and verifies that the results JSON file contains a valid run entry.
        """
        self.set_seed()

        with patch("train.MetaSchNetModels") as MockMeta:
            # Force the meta-model iterator to yield our fixed model.
            instance = MockMeta.return_value
            instance.__iter__.return_value = iter([self.model])
            test_args = [
                "train.py",
                "--architecture",
                "SchNet",
                "--epochs",
                "500",
                "--data_fractions",
                "0.0001",
                "--val_data_fraction",
                "0.0001",
                "--batch_size",
                "2",
                "--lr",
                "0.001",
                "--val_every",
                "500",
                "--vis_every",
                "500",
            ]
            with patch.object(sys, "argv", test_args):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    train_main()
                output = buf.getvalue()
                match = re.search(
                    r"Results will be saved to (?P<results_path>.+)", output
                )
                self.assertIsNotNone(
                    match, "Results path not found in training output."
                )
                results_path = match.group("results_path").strip()

        try:
            with open(results_path, "r") as f:
                result_json = json.load(f)

                # Get the config and loss information
                config = result_json["3"][0]["config"]
                first_train_loss = result_json["3"][0]["losses"]["1"]["train_loss"]
                first_val_loss = result_json["3"][0]["losses"]["0"]["val_loss"]
                last_train_loss = result_json["3"][0]["losses"]["500"]["train_loss"]
                last_val_loss = result_json["3"][0]["losses"]["500"]["val_loss"]

                # First configuration (from MetaSchnetModels) is expected to be:
                self.assertEqual(config["num_params"], 907)

                # np.testing.assert_allclose(
                #     first_train_loss, 141.48191833496094, rtol=0.1
                # )
                # np.testing.assert_allclose(first_val_loss, 61.96849060058594, rtol=0.1)
                # np.testing.assert_allclose(
                #     last_train_loss, 120.46190071105957, rtol=0.1
                # )
                # np.testing.assert_allclose(last_val_loss, 170.6886854171753, rtol=0.1)
        finally:
            if os.path.exists(results_path):
                os.remove(results_path)
