import torch
import torch.nn as nn


# This is from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/modules/loss.py
class PerAtomMAELoss(nn.Module):
    """Simply divide a loss by the number of atoms/nodes in the graph.
    Currently this loss is intened to used with scalar values, not vectors or higher tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(pred / natoms, target / natoms)


# This is from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/modules/loss.py
class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(pred, target)


class L2NormLoss(nn.Module):
    """Currently this loss is intened to used with vectors."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        assert target.dim() == 2
        assert target.shape[1] != 1
        return torch.linalg.vector_norm(pred - target, ord=2, dim=-1)


def unvoigt_stress(voigt_stress_batch):
    """Separates stress tensors in Voigt notation into isotropic and anisotropic components for a batch.

    Parameters:
    - voigt_stress_batch (Tensor): A [32, 6] tensor where each row represents a stress tensor in Voigt notation.
                                   Order: [sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy]

    Returns:
    - isotropic_stress (Tensor): A [32, 6] tensor of the isotropic stress components for each sample.
    - anisotropic_stress (Tensor): A [32, 6] tensor of the anisotropic stress components for each sample.
    """
    if voigt_stress_batch.shape[-1] != 6:
        raise ValueError("Input voigt_stress_batch must have shape [N, 6].")

    # Compute the mean (hydrostatic) stress for each sample
    p = voigt_stress_batch[:, :3].mean(dim=-1, keepdim=True)  # Shape [32, 1]

    # Construct isotropic stress in Voigt notation for each sample
    isotropic_stress = torch.cat(
        [p, p, p, torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(p)], dim=-1
    )

    # Compute anisotropic (deviatoric) stress for each sample
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
    force_magnitude=False,
):
    """Compute composite loss for forces, energy, and stress, considering the mask.

    Args:
        pred_forces (Tensor): Predicted forces. If `convert_forces_to_magnitudes` is True,
            shape is [batch_size, molecule_size, 3]; otherwise, shape is [batch_size, molecule_size].
        true_forces (Tensor): True forces. If `convert_forces_to_magnitudes` is True,
            shape is [batch_size, molecule_size, 3]; otherwise, shape is [batch_size, molecule_size].
        pred_energy (Tensor): Predicted energy with shape [batch_size].
        true_energy (Tensor): True energy with shape [batch_size].
        pred_stress (Tensor): Predicted stress with shape [batch_size, 6].
        true_stress (Tensor): True stress with shape [batch_size, 6].
        convert_forces_to_magnitudes (bool): Whether to convert the 3-dimensional forces to their magnitudes.
        natoms (Tensor, optional): Number of atoms per molecule. If provided, shape is [batch_size].
        mask (Tensor, optional): A mask to filter the input data.
        device (torch.device): Device to use for computation.
        use_mask (bool, optional): Whether to use the mask to filter the input data.
        force_magnitude (bool, optional): Whether to compute the force loss using L2NormLoss or nn.MSELoss.

    Returns:
        dict: A dictionary containing the computed MAE losses for forces, energy, and stress.
    """
    # Mask out padded atoms
    if natoms is None:
        natoms = torch.tensor(
            data=[len(pred_forces[i]) for i in range(len(pred_forces))], device=device
        )
    if use_mask:
        mask = mask.unsqueeze(-1)  # Shape: [batch_size, max_atoms, 1]
        pred_forces = pred_forces * mask.float()
        true_forces = true_forces * mask.float()

    # Compute losses
    energy_loss_fn = PerAtomMAELoss()
    energy_loss = energy_loss_fn(pred=pred_energy, target=true_energy, natoms=natoms)

    if force_magnitude:
        force_loss_fn = L2NormLoss()
        force_loss = force_loss_fn(pred=pred_forces, target=true_forces, natoms=natoms)
    else:
        # Use reduction="none" to compute the loss per atom
        force_loss_fn = nn.MSELoss(reduction="none")
        force_loss = force_loss_fn(pred_forces, true_forces)
        force_loss = force_loss.sum(dim=(2,1)) / (3 * natoms)  # [B, N, 3] -> [B] / natoms
        # # Then take the mean over the directions and then atoms [B, N, 3] -> [B]
        # force_loss = force_loss.mean(dim=(2, 1))

    true_isotropic_stress, true_anisotropic_stress = unvoigt_stress(true_stress)
    pred_isotropic_stress, pred_anisotropic_stress = unvoigt_stress(pred_stress)
    stress_loss_fn = MAELoss()
    stress_isotropic_loss = stress_loss_fn(
        pred=torch.sum(pred_isotropic_stress, dim=1),
        target=torch.sum(true_isotropic_stress, dim=1),
    )
    stress_anisotropic_loss = stress_loss_fn(
        pred=torch.sum(pred_anisotropic_stress, dim=1),
        target=torch.sum(true_anisotropic_stress, dim=1),
    )

    return torch.mean(
        energy_loss
        + force_loss
        + stress_isotropic_loss
        + stress_anisotropic_loss
    )


# Not in use
# class CosineSimilarityLoss(nn.Module):
#     def __init__(self, dim=1):
#         super(CosineSimilarityLoss, self).__init__()
#         self.cos = nn.CosineSimilarity(dim=dim)

#     def forward(self, prediction, target):
#         cosine_sim = self.cos(prediction, target)
#         mean_cosine_sim = cosine_sim.mean()
#         loss = 1 - mean_cosine_sim
#         return loss
