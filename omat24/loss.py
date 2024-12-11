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
        assert (target.to("cpu") / _natoms).shape == target.shape
        return self.loss(pred.to("cpu") / _natoms, target.to("cpu") / _natoms)


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


def unvoigt_stress(voigt_stress_batch):
    """
    Separates stress tensors in Voigt notation into isotropic and anisotropic components for a batch.

    Parameters:
    - voigt_stress_batch (Tensor): A [32, 6] tensor where each row represents a stress tensor in Voigt notation.
                                   Order: [sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy]

    Returns:
    - isotropic_stress (Tensor): A [32, 6] tensor of the isotropic stress components for each sample.
    - anisotropic_stress (Tensor): A [32, 6] tensor of the anisotropic stress components for each sample.
    """
    voigt_stress_batch = torch.as_tensor(
        voigt_stress_batch, dtype=torch.float32, device="cpu"
    )

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


def unvoigt_stress(voigt_stress_batch):
    """Separates a stress tensor in Voigt notation into isotropic and anisotropic components.
    Parameters:
    - voigt_stress (Tensor): A 6-element tensor representing the stress tensor in Voigt notation.
                             Order: [sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy]
    Returns:
    - isotropic_stress (Tensor): 6-element tensor of the isotropic stress component.
    - anisotropic_stress (Tensor): 6-element tensor of the anisotropic stress component.
    """
    voigt_stress_batch = torch.as_tensor(
        voigt_stress_batch, dtype=torch.float32, device="cpu"
    )
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


def compute_mae_loss(
    pred_forces,
    pred_energy,
    pred_stress,
    true_forces,
    true_energy,
    true_stress,
    mask,
    natoms=None,
    use_mask=True,
    convert_forces_to_magnitudes=False,
):
    """Compute the Mean Absolute Error (MAE) loss for forces, energy, and stress, considering the mask.

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

    Returns:
        dict: A dictionary containing the computed MAE losses for forces, energy, and stress.
    """
    per_atom_mae_loss = PerAtomMAELoss()
    l2_norm_loss = L2NormLoss()
    mae_loss = MAELoss()

    # Mask out padded atoms
    if natoms is None:
        natoms = torch.tensor(
            data=[len(pred_forces[i]) for i in range(len(pred_forces))]
        )
    if use_mask:
        mask = mask.unsqueeze(-1)  # Shape: [batch_size, max_atoms, 1]
        pred_forces = pred_forces * mask.float()
        true_forces = true_forces * mask.float()

    # Compute losses
    energy_loss = per_atom_mae_loss(pred=pred_energy, target=true_energy, natoms=natoms)
    if convert_forces_to_magnitudes:
        force_loss = l2_norm_loss(
            pred=torch.linalg.norm(pred_forces, dim=2),
            target=torch.linalg.norm(true_forces, dim=2),
            natoms=natoms,
        )
    else:
        force_loss = l2_norm_loss(pred=pred_forces, target=true_forces, natoms=natoms)

    true_isotropic_stress, true_anisotropic_stress = unvoigt_stress(true_stress)
    pred_isotropic_stress, pred_anisotropic_stress = unvoigt_stress(pred_stress)
    stress_isotropic_loss = mae_loss(
        pred=torch.sum(pred_isotropic_stress, dim=1),
        target=torch.sum(true_isotropic_stress, dim=1),
        natoms=natoms,
    )
    stress_anisotropic_loss = mae_loss(
        pred=torch.sum(pred_anisotropic_stress, dim=1),
        target=torch.sum(true_anisotropic_stress, dim=1),
        natoms=natoms,
    )
    return torch.mean(
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
