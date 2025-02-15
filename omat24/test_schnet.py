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

# Internal
from models.schnet import SchNet


class TestSchNet(unittest.TestCase):
    def setUp(self):
        # Set reproducible seed
        torch.manual_seed(1024)
        np.random.seed(1024)
        self.device = torch.device("cpu")
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

    def test_fixed_schnet_overfit(self):
        """
        Run a minimal training job using the SchNet architecture.
        This test patches the meta-model iterator in train.py to return a fixed SchNet model,
        runs a short training run, and verifies that the results JSON file contains a valid run entry.
        """
        from train import main as train_main

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
                with patch("train.subprocess.run") as mock_subproc_run:
                    mock_subproc_run.return_value = subprocess.CompletedProcess(
                        args=["python3", "model_prediction_evolution.py"],
                        returncode=0,
                        stdout="dummy output",
                        stderr="",
                    )
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

                # For the Transformer, the first configuration (from MetaTransformerModels) is expected to be:
                self.assertEqual(config["num_params"], 875)

                np.testing.assert_allclose(
                    first_train_loss, 19.543967723846436, rtol=0.1
                )
                np.testing.assert_allclose(first_val_loss, 5.409457623958588, rtol=0.1)
                np.testing.assert_allclose(last_train_loss, 18.04733633995056, rtol=0.1)
                np.testing.assert_allclose(last_val_loss, 574.0395112037659, rtol=0.1)
        finally:
            if os.path.exists(results_path):
                os.remove(results_path)
