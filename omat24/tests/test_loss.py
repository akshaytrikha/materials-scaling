import unittest
import torch
from fairchem.core.modules.loss import L2NormLoss


class TestL2NormLoss(unittest.TestCase):
    def test_equivalence_graph_vs_non_graph(self):
        """
        Verifies that L2NormLoss outputs match when using graph-like [N,3] vs.
        batch-like [B, N_atoms, 3] (then reshaped to [B*N_atoms, 3]).
        """
        # Fix the random seed for reproducibility
        torch.manual_seed(1234)

        # Suppose we have a batch of 4 structures, each with 5 atoms => total 20 atoms
        batch_size = 4
        n_atoms = 5
        n_total = batch_size * n_atoms

        # Graph-like input => shape [N, 3]
        pred_graph = torch.randn(n_total, 3)
        tgt_graph = torch.randn(n_total, 3)

        # Batch-like input => shape [B, N_atoms, 3]
        # We'll just reshape the same data to confirm the numeric equivalence
        pred_batched = pred_graph.view(batch_size, n_atoms, 3)
        tgt_batched = tgt_graph.view(batch_size, n_atoms, 3)

        # Construct a dummy natoms vector (not used by L2NormLoss, but needed for forward signature)
        # For "graph" mode, that shape might be [N] (each row is an atom).
        # For "non-graph" mode, that shape might be [B] or any placeholder.
        natoms_graph = torch.ones(n_total)
        natoms_batch = torch.ones(batch_size)

        # Instantiate the L2NormLoss
        loss_fn = L2NormLoss()

        # (1) Graph path: shape [N, 3]
        # The L2NormLoss output is a vector [N], so we sum() it to get a single scalar
        loss_vec_graph = loss_fn(pred_graph, tgt_graph, natoms_graph)
        loss_graph_scalar = loss_vec_graph.sum()

        # (2) Non-graph path: shape [B, N_atoms, 3] => flatten to [B*N_atoms, 3]
        pred_flat = pred_batched.view(-1, 3)
        tgt_flat = tgt_batched.view(-1, 3)
        loss_vec_non_graph = loss_fn(pred_flat, tgt_flat, natoms_graph)
        loss_non_graph_scalar = loss_vec_non_graph.sum()

        # Compare final scalar losses. They should be the same within a tolerance.
        self.assertTrue(
            torch.allclose(loss_graph_scalar, loss_non_graph_scalar, atol=1e-6),
            msg="L2NormLoss differs between graph-like [N,3] vs. flattened batch-like [B,N_atoms,3]!",
        )


if __name__ == "__main__":
    unittest.main()
