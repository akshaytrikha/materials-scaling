import torch
import torch.nn as nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, prediction, target):
        # We return 1 - mean cosine similarity, so that perfect alignment = 0 loss.
        cosine_sim = self.cos(prediction, target)
        # mean_cosine_sim = cosine_sim.mean(dim=1)
        loss = 1 - cosine_sim
        return loss


# from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/modules/loss.py
class PerAtomMAELoss(nn.Module):
    """Simply divide a loss by the number of atoms/nodes in the graph.
    Currently this loss is intended for scalar values, not vectors or higher tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(pred / natoms, target / natoms)


# from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/modules/loss.py
class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        self.loss.reduction = "none"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(pred, target)


# from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/modules/loss.py
class L2NormLoss(nn.Module):
    """Currently this loss is intended to be used with vectors."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        if target.dim() == 3:
            # shape: [batch_size, n_atoms, vector_dim]
            # We compute the per-atom norm of the difference
            diff = pred - target
            return torch.linalg.vector_norm(diff, ord=2, dim=-1)
        else:
            # shape: [batch_size, n_atoms] or [batch_size]
            return (pred - target).abs()  # or norm if needed


def unvoigt_stress(voigt_stress_batch):
    """Separates stress tensors in Voigt notation into isotropic and anisotropic components for a batch.
    voigt_stress_batch: [batch_size, 6]
    """
    if voigt_stress_batch.shape[-1] != 6:
        raise ValueError("Input voigt_stress_batch must have shape [N, 6].")

    # Compute mean (hydrostatic) stress
    p = voigt_stress_batch[:, :3].mean(dim=-1, keepdim=True)
    isotropic_stress = torch.cat(
        [p, p, p, torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(p)], dim=-1
    )
    anisotropic_stress = voigt_stress_batch - isotropic_stress

    return isotropic_stress, anisotropic_stress


def compute_loss(
    pred_forces,
    pred_energy,
    pred_stress,
    true_forces,
    true_energy,
    true_stress,
    mask,
    device,
    natoms=None,
    use_mask=True,
    naive=False,
):
    """Compute composite loss for forces, energy, and stress.

    Args:
        pred_forces (Tensor): Predicted forces [batch, n_atoms, 3].
        true_forces (Tensor): True forces [batch, n_atoms, 3].
        pred_energy (Tensor): Predicted energies [batch].
        true_energy (Tensor): True energies [batch].
        pred_stress (Tensor): Predicted stress [batch, 6].
        true_stress (Tensor): True stress [batch, 6].
        mask (Tensor): Mask for atoms [batch, n_atoms].
        device: torch device.
        natoms (Tensor): Number of atoms per system [batch].
        use_mask (bool): If True, apply mask to forces.
        convert_forces_to_magnitudes (bool): Whether to convert forces to magnitudes if naive=True.
        naive (bool): If True, use only magnitude-based force loss. If False, also incorporate direction.

    Returns:
        torch.Tensor: The composite loss.
    """
    # Handle natoms if not provided
    if natoms is None:
        natoms = torch.tensor(
            [len(pred_forces[i]) for i in range(len(pred_forces))],
            device=device,
        )

    # Apply mask if requested
    if use_mask:
        mask = mask.unsqueeze(-1)  # [batch, n_atoms, 1]
        pred_forces = pred_forces * mask.float()
        true_forces = true_forces * mask.float()

    # Energy loss (per-atom)
    energy_loss = PerAtomMAELoss()(pred=pred_energy, target=true_energy, natoms=natoms)

    # Stress losses
    true_isotropic_stress, true_anisotropic_stress = unvoigt_stress(true_stress)
    pred_isotropic_stress, pred_anisotropic_stress = unvoigt_stress(pred_stress)

    stress_isotropic_loss = MAELoss()(
        pred=torch.sum(pred_isotropic_stress, dim=1),
        target=torch.sum(true_isotropic_stress, dim=1),
    )

    stress_anisotropic_loss = MAELoss()(
        pred=torch.sum(pred_anisotropic_stress, dim=1),
        target=torch.sum(true_anisotropic_stress, dim=1),
    )

    # Forces loss
    # If naive is True, we do what we did before (optionally convert to magnitudes).
    # If naive is False, we incorporate direction by using the full vector and also a cosine similarity loss.
    if naive:
        # Use full vector difference but no direction loss
        force_magnitude_loss = L2NormLoss()(
            pred=pred_forces, target=true_forces, natoms=natoms
        ).mean(dim=1)

        total_loss = torch.mean(
            2.5 * energy_loss
            + 20 * force_magnitude_loss
            + 5 * stress_isotropic_loss
            + 5 * stress_anisotropic_loss
        )

    else:
        force_magnitude_loss = L2NormLoss()(
            pred=pred_forces, target=true_forces, natoms=natoms
        )
        cosine_loss = CosineSimilarityLoss(dim=2)(pred_forces, true_forces)
        force_loss_per_structure = (force_magnitude_loss + cosine_loss).mean(dim=1)

        total_loss = torch.mean(
            2.5 * energy_loss
            + 20 * force_loss_per_structure
            + 5 * stress_isotropic_loss
            + 5 * stress_anisotropic_loss
        )

    return total_loss
