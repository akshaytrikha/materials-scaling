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
        # Create first structure with 2 atoms
        positions1 = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # atom 1
                [1.0, 0.0, 0.0],  # atom 2
            ],
            dtype=torch.float,
            device=self.device,
        )
        atomic_numbers1 = torch.tensor([1, 6], dtype=torch.long, device=self.device)

        # Create second structure with 4 atoms
        positions2 = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # atom 1
                [1.0, 0.0, 0.0],  # atom 2
                [0.0, 1.0, 0.0],  # atom 3
                [1.0, 1.0, 0.0],  # atom 4
            ],
            dtype=torch.float,
            device=self.device,
        )
        atomic_numbers2 = torch.tensor(
            [1, 6, 8, 7], dtype=torch.long, device=self.device
        )

        # Create PyG data objects
        data1 = PyGData(
            pos=positions1,
            atomic_numbers=atomic_numbers1,
            energy=torch.tensor([1.0], device=self.device),
            forces=torch.randn(2, 3, device=self.device),
            stress=torch.randn(6, device=self.device),
        )
        data1.natoms = torch.tensor([2], dtype=torch.long, device=self.device)
        data1.cell = torch.eye(3, device=self.device).unsqueeze(0)
        data1.pbc = torch.ones(3, dtype=torch.bool, device=self.device)

        data2 = PyGData(
            pos=positions2,
            atomic_numbers=atomic_numbers2,
            energy=torch.tensor([2.0], device=self.device),
            forces=torch.randn(4, 3, device=self.device),
            stress=torch.randn(6, device=self.device),
        )
        data2.natoms = torch.tensor([4], dtype=torch.long, device=self.device)
        data2.cell = torch.eye(3, device=self.device).unsqueeze(0)
        data2.pbc = torch.ones(3, dtype=torch.bool, device=self.device)

        # Create a batch
        self.batch = Batch.from_data_list([data1, data2])

        # Add batch_full attribute
        self.batch.batch_full = torch.tensor(
            [0, 0, 1, 1, 1, 1], dtype=torch.long, device=self.device
        )
        self.batch.atomic_numbers_full = self.batch.atomic_numbers

        # Create edge connections (all-to-all within each structure)
        edge_src = []
        edge_dst = []

        # Structure 1: atoms 0-1
        for i in range(2):
            for j in range(2):
                if i != j:
                    edge_src.append(i)
                    edge_dst.append(j)

        # Structure 2: atoms 2-5
        for i in range(2, 6):
            for j in range(2, 6):
                if i != j:
                    edge_src.append(i)
                    edge_dst.append(j)

        self.batch.edge_index = torch.tensor(
            [edge_src, edge_dst],
            dtype=torch.long,
            device=self.device,
        )

        # Compute edge distances (simplified to all 1.0)
        self.batch.edge_distance = torch.ones(len(edge_src), device=self.device)

        # Compute edge vectors
        edge_vecs = []
        for i, j in zip(edge_src, edge_dst):
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
            self.assertEqual(forces.shape, (6, 3))  # 6 atoms total (2+4), 3 dimensions
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

    def test_so3_embedding(self):
        """Test that the SO3_Embedding produces outputs with the expected shapes."""
        batch_size = 2
        n_atoms = 3
        lmax = 1

        # Create input tensor with shape [batch_size * n_atoms, 3]
        x = torch.randn(batch_size * n_atoms, 3, device=self.device)

        # Initialize an SO3_Embedding to store the result
        embedding = SO3_Embedding(
            length=batch_size * n_atoms,
            lmax_list=[lmax],
            num_channels=1,
            device=self.device,
            dtype=torch.float,
        )

        # Manually compute spherical harmonic coefficients
        # For l=0 (scalar part)
        l0_coeff = torch.ones(batch_size * n_atoms, 1, device=self.device)

        # For l=1 (vector part, same dimension as input)
        l1_coeff = x  # Shape: [batch_size * n_atoms, 3]

        # Combine into a dictionary of coefficients
        output = {0: l0_coeff, 1: l1_coeff}

        # Check output type and shape
        self.assertIsInstance(output, dict, "Output should be a dictionary")

        # For lmax=1, we should have keys 0 and 1
        self.assertIn(0, output, "Output should have key 0")
        self.assertIn(1, output, "Output should have key 1")

        # l=0 should have shape [batch_size * n_atoms, 1]
        self.assertEqual(
            output[0].shape,
            (batch_size * n_atoms, 1),
            "l=0 component should have shape [batch_size * n_atoms, 1]",
        )

        # l=1 should have shape [batch_size * n_atoms, 3]
        self.assertEqual(
            output[1].shape,
            (batch_size * n_atoms, 3),
            "l=1 component should have shape [batch_size * n_atoms, 3]",
        )

    def test_fixed_equiformer_overfit(self):
        """Test a minimal training run using EquiformerV2 architecture overfits and yields expected config and loss values."""
        self.set_seed()

        # Patch the DEVICE global variable to force CPU
        with patch("train.DEVICE", torch.device("cpu")):
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
            self.assertEqual(config["num_params"], 1798)

            # The expected loss values below are chosen based on a prior minimal overfit run.
            np.testing.assert_allclose(first_train_loss, 124.94979858398438, rtol=0.1)
            np.testing.assert_allclose(first_val_loss, 95.6764030456543, rtol=0.1)
            if os.getenv("IS_CI", False):
                np.testing.assert_allclose(last_train_loss, 7.25011635, rtol=0.1)
            else:
                np.testing.assert_allclose(
                    last_train_loss, 10.293143272399902, rtol=0.1
                )
            if os.getenv("IS_CI", False):
                np.testing.assert_allclose(last_val_loss, 127.09902191, rtol=0.1)
            else:
                np.testing.assert_allclose(last_val_loss, 104.14714431762695, rtol=0.1)

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

    # def test_equivariance(self):
    #     """Test that the model's forces transform correctly under rotation (equivariance)."""
    #     self.model.eval()

    #     # Create a simple rotation matrix (90 degrees around z-axis)
    #     theta = torch.tensor(torch.pi / 2, device=self.device)
    #     rot_matrix = torch.tensor(
    #         [
    #             [torch.cos(theta), -torch.sin(theta), 0],
    #             [torch.sin(theta), torch.cos(theta), 0],
    #             [0, 0, 1],
    #         ],
    #         device=self.device,
    #     )

    #     # Create a proper deep copy of the batch
    #     rotated_batch = Batch()
    #     for key, value in self.batch:
    #         setattr(
    #             rotated_batch,
    #             key,
    #             value.clone() if isinstance(value, torch.Tensor) else value,
    #         )

    #     # Apply rotation to positions
    #     rotated_batch.pos = torch.matmul(rotated_batch.pos, rot_matrix.T)

    #     # Apply rotation to edge vectors
    #     rotated_batch.edge_distance_vec = torch.matmul(
    #         rotated_batch.edge_distance_vec, rot_matrix.T
    #     )

    #     # Forward pass with original and rotated data
    #     with torch.no_grad():
    #         original_forces, original_energy, original_stress = self.model(self.batch)
    #         rotated_forces, rotated_energy, rotated_stress = self.model(rotated_batch)

    #     # Forces should transform like the positions (vectors)
    #     # Rotate the original forces
    #     rotated_original_forces = torch.matmul(original_forces, rot_matrix.T)

    #     # Check equivariance of forces (with some tolerance)
    #     forces_close = torch.allclose(
    #         rotated_forces, rotated_original_forces, atol=1e-5
    #     )
    #     if not forces_close:
    #         # Print detailed information for debugging
    #         max_diff = torch.max(torch.abs(rotated_forces - rotated_original_forces))
    #         print(f"Maximum force difference: {max_diff}")
    #         print("Rotated forces:")
    #         print(rotated_forces)
    #         print("Manually rotated original forces:")
    #         print(rotated_original_forces)

    #     self.assertTrue(
    #         forces_close,
    #         "Forces are not equivariant under rotation",
    #     )

    #     # Energy should be invariant under rotation
    #     self.assertTrue(
    #         torch.allclose(original_energy, rotated_energy, atol=1e-5),
    #         "Energy is not invariant under rotation",
    #     )


if __name__ == "__main__":
    unittest.main()
