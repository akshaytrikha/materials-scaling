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
import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch

# Internal
from models.equiformer_v2 import EquiformerS2EF
from fairchem.core.models.equiformer_v2.so3 import SO3_Embedding
from train import main as train_main


class TestEquiformerV2(unittest.TestCase):
    def set_seed(self):
        """Set a fixed seed for reproducibility."""
        SEED = 1024
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)

    def create_dummy_data(self):
        # Create two identical structures with 3 atoms each
        positions1 = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # atom 1
                [1.0, 0.0, 0.0],  # atom 2
                [0.0, 1.0, 0.0],  # atom 3
            ],
            dtype=torch.float,
            device=self.device,
        )

        positions2 = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # atom 1
                [1.0, 0.0, 0.0],  # atom 2
                [0.0, 1.0, 0.0],  # atom 3
            ],
            dtype=torch.float,
            device=self.device,
        )

        # H, C, O atoms for each structure
        atomic_numbers1 = torch.tensor([1, 6, 8], dtype=torch.long, device=self.device)
        atomic_numbers2 = torch.tensor([1, 6, 8], dtype=torch.long, device=self.device)

        # Create two PyG data objects
        data1 = PyGData(
            pos=positions1,
            atomic_numbers=atomic_numbers1,
            energy=torch.tensor([1.0], device=self.device),
            forces=torch.randn(3, 3, device=self.device),
            stress=torch.randn(6, device=self.device),
        )
        data1.natoms = torch.tensor([3], dtype=torch.long, device=self.device)
        data1.cell = torch.eye(3, device=self.device).unsqueeze(0)  # 1×3×3 unit cell
        data1.pbc = torch.ones(
            3, dtype=torch.bool, device=self.device
        )  # Periodic in all dimensions

        data2 = PyGData(
            pos=positions2,
            atomic_numbers=atomic_numbers2,
            energy=torch.tensor([2.0], device=self.device),
            forces=torch.randn(3, 3, device=self.device),
            stress=torch.randn(6, device=self.device),
        )
        data2.natoms = torch.tensor([3], dtype=torch.long, device=self.device)
        data2.cell = torch.eye(3, device=self.device).unsqueeze(0)  # 1×3×3 unit cell
        data2.pbc = torch.ones(
            3, dtype=torch.bool, device=self.device
        )  # Periodic in all dimensions

        # Create a batch from the two data objects
        self.batch = Batch.from_data_list([data1, data2])

        # Add batch_full attribute expected by the model
        self.batch.batch_full = torch.tensor(
            [0, 0, 0, 1, 1, 1], dtype=torch.long, device=self.device
        )

        # Add additional attributes that might be required
        self.batch.atomic_numbers_full = self.batch.atomic_numbers
        self.batch.edge_index = torch.tensor(
            [
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],  # Source nodes
                [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4],  # Target nodes
            ],
            dtype=torch.long,
            device=self.device,
        )

        # Edge distances (can be computed from positions, but providing directly for simplicity)
        self.batch.edge_distance = torch.ones(12, device=self.device)

        # Edge distance vectors (vector from source to target)
        edge_vecs = []
        for i, j in zip(self.batch.edge_index[0], self.batch.edge_index[1]):
            source_pos = self.batch.pos[i]
            target_pos = self.batch.pos[j]
            edge_vecs.append(target_pos - source_pos)
        self.batch.edge_distance_vec = torch.stack(edge_vecs)

    def setUp(self):
        self.set_seed()
        self.device = torch.device("cpu")

        # Create a minimal backbone configuration
        config = {
            "regress_forces": True,
            "use_pbc": True,
            "otf_graph": True,
            "max_neighbors": 10,
            "max_radius": 5.0,
            "num_layers": 1,
            "sphere_channels": 2,  # Increased from 1
            "attn_hidden_channels": 2,  # Increased from 1
            "num_heads": 2,  # Increased from 1
            "attn_alpha_channels": 2,  # Increased from 1
            "attn_value_channels": 2,  # Increased from 1
            "ffn_hidden_channels": 2,  # Increased from 1
            "norm_type": "rms_norm_sh",  # Changed from layer_norm_sh
            "lmax_list": [1],  # Increased from [0]
            "mmax_list": [1],  # Increased from [0]
            "grid_resolution": 4,  # Set to a positive value instead of None
            "num_sphere_samples": 2,  # Increased from 1
            "edge_channels": 2,  # Increased from 1
            "max_num_elements": 119,
        }

        self.model = EquiformerS2EF(config).to(self.device)

        # Create dummy data
        self.create_dummy_data()

    def test_forward_output_shapes(self):
        # Run the forward pass
        try:
            forces, energy, stress = self.model(self.batch)

            # Check shapes if successful
            self.assertEqual(forces.shape, (6, 3))  # 6 atoms, 3 dimensions
            self.assertEqual(energy.shape, (2,))  # 2 structures
            self.assertEqual(stress.shape, (2, 6))  # 2 structures, 6 stress components
        except Exception as e:
            self.fail(f"Forward pass failed with exception: {e}")

    def test_gradient_flow(self):
        # Forward pass
        self.model.train()
        self.model.zero_grad()
        forces, energy, stress = self.model(self.batch)

        # Backward pass
        loss = forces.pow(2).sum() + energy.pow(2).sum() + stress.pow(2).sum()
        loss.backward()

        # Check gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None.")
                self.assertGreater(
                    param.grad.abs().sum().item(),
                    0,
                    f"Gradient for {name} is zero.",
                )

    def test_fixed_equiformer_overfit(self):
        """Test a minimal training run using EquiformerV2 architecture overfits and yields expected config and loss values."""
        self.set_seed()

        # Patch the MetaEquiformerV2Models so that its iterator yields our fixed_model.
        with patch("train.MetaEquiformerV2Models") as MockMeta:
            instance = MockMeta.return_value
            instance.__iter__.return_value = iter([self.model])

            # Set minimal training arguments.
            test_args = [
                "train.py",
                "--architecture",
                "EquiformerV2",
                "--epochs",
                "500",
                "--data_fraction",
                "0.00001",
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
                with patch("train.subprocess.run") as mock_subproc_run:
                    # Prevent the actual generation of prediction evolution plots.
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
                        match, "Could not find results filename in output"
                    )
                    results_filename = match.group("results_path").strip()
                    print("Captured results filename:", results_filename)

            visualization_filepath = None
            try:
                with open(results_filename, "r") as f:
                    result_json = json.load(f)

                config = result_json["1"][0]["config"]
                first_train_loss = result_json["1"][0]["losses"]["1"]["train_loss"]
                first_val_loss = result_json["1"][0]["losses"]["0"]["val_loss"]
                last_train_loss = result_json["1"][0]["losses"]["500"]["train_loss"]
                last_val_loss = result_json["1"][0]["losses"]["500"]["val_loss"]

                # Our fixed minimal Equiformer (with dummy backbone) should only include the parameters
                # from the MLP readouts. For in_dim=1, the readouts contribute 4, 8, and 14 params respectively,
                # yielding a total of 26.
                self.assertEqual(config["num_params"], 4224)

                # The expected loss values below are chosen based on a prior minimal overfit run.
                np.testing.assert_allclose(
                    first_train_loss, 105.53826141357422, rtol=0.1
                )
                np.testing.assert_allclose(first_val_loss, 78.86790084838867, rtol=0.1)
                np.testing.assert_allclose(
                    last_train_loss, 0.27151069045066833, rtol=0.1
                )
                np.testing.assert_allclose(last_val_loss, 144.41654205322266, rtol=0.1)

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
                self.assertTrue(
                    visualization_filepath.exists(),
                    "Visualization was not created.",
                )
            finally:
                if os.path.exists(results_filename):
                    os.remove(results_filename)
                if visualization_filepath and os.path.exists(visualization_filepath):
                    shutil.rmtree(visualization_filepath)


if __name__ == "__main__":
    unittest.main()
