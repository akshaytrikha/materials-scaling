import torch.nn as nn


class MLPReadout(nn.Module):
    """A simple MLP readout for transforming node embeddings to per-atom energies, forces, & stress."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, node_emb):
        return self.net(node_emb)
