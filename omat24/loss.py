import torch
import torch.nn as nn


def unvoigt_stress(voigt_stress):
    """Separates a stress tensor in Voigt notation into isotropic and anisotropic components.

    Parameters:
    - voigt_stress (Tensor): A 6-element tensor representing the stress tensor in Voigt notation.
                             Order: [sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy]

    Returns:
    - isotropic_stress (Tensor): 6-element tensor of the isotropic stress component.
    - anisotropic_stress (Tensor): 6-element tensor of the anisotropic stress component.
    """
    voigt_stress = torch.as_tensor(voigt_stress, dtype=torch.float32)

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
    # Mask out padded atoms
    mask = mask.unsqueeze(-1)  # Shape: [batch_size, max_atoms, 1]
    pred_forces = pred_forces * mask.float()
    true_forces = true_forces * mask.float()
    # Compute losses
    energy_loss = nn.L1Loss()(pred_energy, true_energy)
    force_loss = nn.MSELoss()(pred_forces, true_forces)
    true_isotropic_stress, true_anisotropic_stress = unvoigt_stress(true_stress)
    pred_isotropic_stress, pred_anisotropic_stress = unvoigt_stress(pred_stress)
    stress_isotropic_loss = nn.L1Loss()(pred_isotropic_stress, true_isotropic_stress)
    stress_anisotropic_loss = nn.MSELoss()(
        pred_anisotropic_stress, true_anisotropic_stress
    )
    return (
        2.5 * energy_loss
        + 20 * force_loss
        + 5 * stress_isotropic_loss
        + 5 * stress_anisotropic_loss
    )


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, prediction, target):
        cosine_sim = self.cos(prediction, target)
        mean_cosine_sim = cosine_sim.mean()
        loss = 1 - mean_cosine_sim
        return loss
