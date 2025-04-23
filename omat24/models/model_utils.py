# External
import torch
import torch.nn as nn

# Internal
from data_utils import DATASET_INFO


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


def initialize_output_heads(
    energy_head, force_head=None, stress_head=None, per_atom: bool = True
):
    """
    Initialize output heads to predict dataset means.

    Args:
        energy_head: Output head for energy prediction
        force_head: Output head for force prediction (optional)
        stress_head: Output head for stress prediction (optional)
    """
    if per_atom:
        # Divide energies and stresses by 20 atoms to match per-atom values
        DATASET_INFO["train"]["all"]["means"]["energy"] /= 20
        # also divide stress by 10
        DATASET_INFO["train"]["all"]["means"]["stress"] = [
            x / 20 for x in DATASET_INFO["train"]["all"]["means"]["stress"]
        ]

    # Initialize energy head
    if energy_head is not None:
        nn.init.normal_(energy_head.net[-1].weight, mean=0, std=0.01)
        energy_head.net[-1].bias.data.fill_(
            DATASET_INFO["train"]["all"]["means"]["energy"]
        )

    # Initialize force head
    if force_head is not None:
        nn.init.normal_(force_head.net[-1].weight, mean=0, std=0.01)
        force_head.net[-1].bias.data.zero_()

    # Initialize stress head
    if stress_head is not None:
        nn.init.normal_(stress_head.net[-1].weight, mean=0, std=0.01)
        stress_head.net[-1].bias.data.copy_(
            torch.tensor(
                DATASET_INFO["train"]["all"]["means"]["stress"],
                device=stress_head.net[-1].bias.device,
            )
        )


def initialize_eqV2_output_heads(
    energy_head, force_head=None, stress_head=None, per_atom: bool = True
):
    """
    Initialize EquiformerV2 output heads to predict dataset means.

    Args:
        energy_head: EqV2ScalarHead for energy prediction
        force_head: EqV2VectorHead for force prediction
        stress_head: Stress prediction head (likely rank2_symmetric_head)
        per_atom: Whether to scale by number of atoms (default=True)
    """
    if per_atom:
        # Divide energies and stresses by 20 atoms to match per-atom values
        DATASET_INFO["train"]["all"]["means"]["energy"] /= 20
        # also divide stress by 20
        DATASET_INFO["train"]["all"]["means"]["stress"] = [
            x / 20 for x in DATASET_INFO["train"]["all"]["means"]["stress"]
        ]

    energy_mean = DATASET_INFO["train"]["all"]["means"]["energy"]
    stress_means = DATASET_INFO["train"]["all"]["means"]["stress"]
    device = next(energy_head.parameters()).device if energy_head is not None else None

    # Initialize energy head
    if energy_head is not None:
        # EquiformerV2 heads typically have a 'proj' layer as their output
        if hasattr(energy_head, "proj"):
            if hasattr(energy_head.proj, "weight"):
                nn.init.normal_(energy_head.proj.weight, mean=0, std=0.01)
            if hasattr(energy_head.proj, "bias") and energy_head.proj.bias is not None:
                energy_head.proj.bias.data.fill_(energy_mean)
        # Some might use a different structure
        elif hasattr(energy_head, "energy_block") and hasattr(
            energy_head.energy_block, "so3_linear_2"
        ):
            if hasattr(energy_head.energy_block.so3_linear_2, "weight"):
                nn.init.normal_(
                    energy_head.energy_block.so3_linear_2.weight, mean=0, std=0.01
                )
            if hasattr(energy_head.energy_block.so3_linear_2, "bias"):
                energy_head.energy_block.so3_linear_2.bias.data.fill_(energy_mean)

    # Initialize force head
    if force_head is not None:
        if hasattr(force_head, "proj"):
            if hasattr(force_head.proj, "weight"):
                nn.init.normal_(force_head.proj.weight, mean=0, std=0.01)
            if hasattr(force_head.proj, "bias") and force_head.proj.bias is not None:
                force_head.proj.bias.data.zero_()
        elif hasattr(force_head, "force_block") and hasattr(
            force_head.force_block, "so3_linear_2"
        ):
            if hasattr(force_head.force_block.so3_linear_2, "weight"):
                nn.init.normal_(
                    force_head.force_block.so3_linear_2.weight, mean=0, std=0.01
                )
            if hasattr(force_head.force_block.so3_linear_2, "bias"):
                force_head.force_block.so3_linear_2.bias.data.zero_()

    # Initialize stress head
    if stress_head is not None:
        # Stress head might decompose stress into isotropic and anisotropic parts
        if hasattr(stress_head, "isotropic_net") and hasattr(
            stress_head, "anisotropic_net"
        ):
            # Handle isotropic component (scalar)
            if hasattr(stress_head.isotropic_net[-1], "weight"):
                nn.init.normal_(stress_head.isotropic_net[-1].weight, mean=0, std=0.01)
            if hasattr(stress_head.isotropic_net[-1], "bias"):
                # Use mean for isotropic component (hydrostatic pressure)
                stress_head.isotropic_net[-1].bias.data.fill_(
                    sum(stress_means) / len(stress_means)
                )

            # Handle anisotropic component (traceless tensor)
            if hasattr(stress_head.anisotropic_net[-1], "weight"):
                nn.init.normal_(
                    stress_head.anisotropic_net[-1].weight, mean=0, std=0.01
                )
            if hasattr(stress_head.anisotropic_net[-1], "bias"):
                # This depends on how many components the anisotropic part has
                # Typically 5 components for the traceless part of the stress tensor
                bias_tensor = torch.zeros_like(stress_head.anisotropic_net[-1].bias)
                if bias_tensor.shape[0] == len(stress_means):
                    # If the shapes match, use the full stress means
                    stress_tensor = torch.tensor(stress_means, device=device)
                    stress_head.anisotropic_net[-1].bias.data.copy_(stress_tensor)
                else:
                    # Otherwise just initialize to zero
                    stress_head.anisotropic_net[-1].bias.data.zero_()
        # Or it might have a simple proj layer
        elif hasattr(stress_head, "proj"):
            if hasattr(stress_head.proj, "weight"):
                nn.init.normal_(stress_head.proj.weight, mean=0, std=0.01)
            if hasattr(stress_head.proj, "bias") and stress_head.proj.bias is not None:
                stress_tensor = torch.tensor(stress_means, device=device)
                if stress_head.proj.bias.shape[0] == len(stress_means):
                    stress_head.proj.bias.data.copy_(stress_tensor)
                else:
                    stress_head.proj.bias.data.zero_()


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
            nn.SiLU(),
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

        # Apply standardized initialization to each layer
        for module in self.net:
            if isinstance(module, nn.Linear):
                initialize_weights(module)

    def forward(self, node_emb):
        return self.net(node_emb)
