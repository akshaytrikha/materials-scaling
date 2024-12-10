import torch
import torch.nn as nn

# This is from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/modules/loss.py
class PerAtomMAELoss(nn.Module):
    """
    Simply divide a loss by the number of atoms/nodes in the graph.
    Current this loss is intened to used with scalar values, not vectors or higher tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        _natoms = torch.reshape(natoms, target.shape)
        # check if target is a scalar
        assert target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1)
        # check per_atom shape
        assert (target / _natoms).shape == target.shape
        return self.loss(pred / _natoms, target / _natoms)

# This is from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/modules/loss.py
class L2NormLoss(nn.Module):
    """
    Currently this loss is intened to used with vectors.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        assert target.dim() == 2
        assert target.shape[1] != 1
        return torch.linalg.vector_norm(pred - target, ord=2, dim=-1)

# This is from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/modules/loss.py
class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(pred, target)

def unvoigt_stress(voigt_stress):
    """Separates a stress tensor in Voigt notation into isotropic and anisotropic components.

    Parameters:
    - voigt_stress (Tensor): A 6-element tensor representing the stress tensor in Voigt notation.
                             Order: [sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy]

    Returns:
    - isotropic_stress (Tensor): 6-element tensor of the isotropic stress component.
    - anisotropic_stress (Tensor): 6-element tensor of the anisotropic stress component.
    """
    voigt_stress = torch.as_tensor(voigt_stress, dtype=torch.float32, device='cpu')

    if voigt_stress.shape != (6,):
        raise ValueError("Input voigt_stress must be a 6-element tensor.")

    # Compute the mean (hydrostatic) stress
    p = (voigt_stress[0] + voigt_stress[1] + voigt_stress[2]) / 3.0

    # Construct isotropic stress in Voigt notation
    isotropic_stress = torch.tensor([p, p, p, 0.0, 0.0, 0.0], dtype=torch.float32)

    # Compute anisotropic (deviatoric) stress
    anisotropic_stress = voigt_stress - isotropic_stress

    return isotropic_stress, anisotropic_stress

def compute_mae_loss(
    pred_forces, pred_energy, pred_stress, true_forces, true_energy, true_stress, mask
):
    """Compute the MAE loss for forces, energy, and stress, considering the mask."""
    per_atom_mae_loss = PerAtomMAELoss()
    l2_norm_loss = L2NormLoss()
    mae_loss = MAELoss()
    # Mask out padded atoms
    mask = mask.unsqueeze(-1)  # Shape: [batch_size, max_atoms, 1]
    pred_forces = (pred_forces * mask.float())
    true_forces = (true_forces * mask.float())
    nmolecules = pred_forces.shape[0]
    natoms = pred_forces.shape[1]
    # Compute losses
    total_loss = 0
    for i in range(nmolecules):
        energy_loss = per_atom_mae_loss(pred=pred_energy[i].item(), target=true_energy[i].item(), natoms=natoms)
        force_loss = sum(l2_norm_loss(pred=pred_forces[i], target=true_forces[i], natoms=natoms))
        true_isotropic_stress, true_anisotropic_stress = unvoigt_stress(true_stress[i])
        pred_isotropic_stress, pred_anisotropic_stress = unvoigt_stress(pred_stress[i])
        stress_isotropic_loss = sum(mae_loss(pred=pred_isotropic_stress, target=true_isotropic_stress, natoms=natoms))
        stress_anisotropic_loss = sum(mae_loss(pred=pred_anisotropic_stress, target=true_anisotropic_stress, natoms=natoms))
        total_loss += 2.5 * energy_loss + 20 * force_loss + 5 * stress_isotropic_loss + 5 * stress_anisotropic_loss
    return total_loss

class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, prediction, target):
        cosine_sim = self.cos(prediction, target)
        mean_cosine_sim = cosine_sim.mean()
        loss = 1 - mean_cosine_sim
        return loss
