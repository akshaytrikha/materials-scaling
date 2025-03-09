import torch
import torch.nn as nn


def initialize_weights(module):
    """
    Generic weight initialization for different layer types.

    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Embedding):
        # Standard initialization for embeddings
        nn.init.normal_(module.weight, mean=0.0, std=0.01)

    elif isinstance(module, nn.Linear):
        # Kaiming initialization for linear layers (good with ReLU/LeakyReLU)
        nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.LayerNorm):
        # Standard initialization for LayerNorm
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0)


def initialize_output_heads(energy_head, force_head=None, stress_head=None):
    """
    Initialize output heads to predict dataset means.

    Args:
        energy_head: Output head for energy prediction
        force_head: Output head for force prediction (optional)
        stress_head: Output head for stress prediction (optional)
    """
    # Initialize energy head
    if energy_head is not None:
        nn.init.normal_(energy_head.net[-1].weight, mean=0, std=0.01)
        energy_head.net[-1].bias.data.fill_(-9.773)

    # Initialize force head
    if force_head is not None:
        nn.init.normal_(force_head.net[-1].weight, mean=0, std=0.01)
        force_head.net[-1].bias.data.zero_()

    # Initialize stress head
    if stress_head is not None:
        nn.init.normal_(stress_head.net[-1].weight, mean=0, std=0.01)
        stress_head.net[-1].bias.data.copy_(
            torch.tensor(
                [-0.03071, -0.03048, -0.03014, 2.67e-6, -9.82e-6, -1.06e-4],
                device=stress_head.net[-1].bias.device,
            )
        )


def apply_initialization(model):
    """
    Apply initialization to an entire model.

    Args:
        model: PyTorch model to initialize
    """
    # Initialize most layers using the generic function
    for name, module in model.named_modules():
        # Skip output heads and complete modules (they'll be initialized separately)
        if (
            isinstance(module, nn.Sequential)
            or isinstance(module, nn.ModuleList)
            or "energy_head" in name
            or "force_head" in name
            or "stress_head" in name
        ):
            continue

        initialize_weights(module)

    # Initialize output heads if they exist
    if hasattr(model, "energy_head"):
        initialize_output_heads(
            model.energy_head,
            getattr(model, "force_head", None),
            getattr(model, "stress_head", None),
        )


class MLPOutput(nn.Module):
    """A simple MLP output head for transforming node embeddings to per-atom energies, forces, & stress."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_dim, out_dim),
        )

        # Apply standardized initialization to each layer
        for module in self.net:
            if isinstance(module, nn.Linear):
                initialize_weights(module)

    def forward(self, node_emb):
        return self.net(node_emb)
