import torch
from einops import rearrange
from fairchem.core.modules.loss import DDPLoss


ENERGY_COEF = 2.5
FORCE_COEF = 20
STRESS_COEF = 5


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
    graph=False,
    structure_index=[],
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
        mask (Tensor, optional): A mask to filter the input data.
        device (torch.device): Device to use for computation.
        natoms (Tensor, optional): Number of atoms per molecule. If provided, shape is [batch_size].
        graph (bool): Whether or not the input data is in graph format.
        structure_index (list, optional): A list of structure indices for each atom in the graph batch.

    Returns:
        dict: A dictionary containing the computed MAE losses for forces, energy, and stress.
    """
    if natoms is None:
        raise ValueError("natoms is None")

    # Initialize loss components as specified in the eqV2-S config
    # https://github.com/FAIR-Chem/fairchem/blob/main/configs/omat24/all/eqV2_31M.yml
    energy_loss_fn = DDPLoss("per_atom_mae", reduction="mean")
    forces_loss_fn = DDPLoss("l2mae", reduction="mean")
    stress_loss_fn = DDPLoss("mae", reduction="mean")

    energy_loss = energy_loss_fn(pred_energy, true_energy, natoms)

    if graph == False:
        if mask is None:
            raise ValueError("mask is None")
        pred_flat = rearrange(pred_forces, "b n d -> (b n) d")
        true_flat = rearrange(true_forces, "b n d -> (b n) d")
        flat_mask = rearrange(mask, "b n -> (b n)")  # shape [batch_size * max_atoms]
        # Obtain indices of non-zero elements in the mask
        # Zeroes in the mask indicate padded atoms
        nonzero_indices = flat_mask.nonzero().squeeze(-1)  # shape [num_nonzero]
        pred_forces = pred_flat[nonzero_indices]
        true_forces = true_flat[nonzero_indices]

    force_loss = forces_loss_fn(pred_forces, true_forces, natoms)

    true_iso_stress, true_aniso_stress = unvoigt_stress(true_stress)
    pred_iso_stress, pred_aniso_stress = unvoigt_stress(pred_stress)
    stress_iso_loss = stress_loss_fn(pred_iso_stress, true_iso_stress, natoms)
    stress_aniso_loss = stress_loss_fn(pred_aniso_stress, true_aniso_stress, natoms)

    total_loss = (
        ENERGY_COEF * energy_loss
        + FORCE_COEF * force_loss
        + STRESS_COEF * stress_iso_loss
        + STRESS_COEF * stress_aniso_loss
    )

    loss_dict = {
        "total_loss": total_loss,
        "energy_loss": ENERGY_COEF * energy_loss,
        "force_loss": FORCE_COEF * force_loss,
        "stress_iso_loss": STRESS_COEF * stress_iso_loss,
        "stress_aniso_loss": STRESS_COEF * stress_aniso_loss,
    }

    return loss_dict
