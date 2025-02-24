import torch
import torch.nn as nn
from torch_scatter import scatter


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


def compute_graph_force_loss(pred_forces, true_forces, structure_index, natoms):
    """
    Args:
        pred_forces (Tensor): Predicted forces [N, 3].
        true_forces (Tensor): True forces [N, 3].
        structure_index (list): A list of structure indices for each atom in the graph batch.
        natoms (Tensor): Number of atoms per structure [B].
    """
    # 1) MSE "per-atom" (sum over x,y,z)
    #    shape => [N]
    force_loss_fn = nn.MSELoss(reduction="none")
    force_loss_per_atom = force_loss_fn(pred_forces, true_forces)

    # 2) Scatter across structures
    #    shape => [B]
    #    We sum per structure if we want to do the same normalization
    #    that the FCN/Transformer does.
    force_sum_struct = scatter(
        force_loss_per_atom, structure_index, dim=0, reduce="sum"
    )

    # 3) Divide per structure by (3 * number_of_atoms_in_that_structure)
    #    shape => [B]
    #    This matches the “force_loss = sum over atoms / (3 * natoms)”
    force_avg_struct = force_sum_struct / (3.0 * natoms.unsqueeze(1))

    # 4) Finally, average over the B structures
    force_loss = force_avg_struct.mean()

    return force_loss


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
    # Mask out padded atoms
    if natoms is None:
        natoms = torch.tensor(
            data=[len(pred_forces[i]) for i in range(len(pred_forces))], device=device
        )
    if mask is not None:
        mask = mask.unsqueeze(-1)  # Shape: [batch_size, max_atoms, 1]
        pred_forces = pred_forces * mask.float()
        true_forces = true_forces * mask.float()

    # Compute losses
    energy_loss_fn = PerAtomMAELoss()
    energy_loss = energy_loss_fn(pred=pred_energy, target=true_energy, natoms=natoms)
    energy_loss = torch.mean(energy_loss)  # Take mean over batch

    # Use reduction="none" to compute the loss per atom
    force_loss_fn = nn.MSELoss(reduction="none")
    force_loss = force_loss_fn(pred_forces, true_forces)

    if graph:
        force_loss = compute_graph_force_loss(
            pred_forces, true_forces, structure_index, natoms
        )
    else:
        force_loss = force_loss.sum(dim=(2, 1)) / (
            3 * natoms
        )  # [B, N, 3] -> [B] / natoms
        # Then take the mean over the directions and then atoms [B, N, 3] -> [B]
        force_loss = torch.mean(force_loss)

    true_isotropic_stress, true_anisotropic_stress = unvoigt_stress(true_stress)
    pred_isotropic_stress, pred_anisotropic_stress = unvoigt_stress(pred_stress)
    stress_loss_fn = MAELoss()
    stress_isotropic_loss = stress_loss_fn(
        pred=pred_isotropic_stress, target=true_isotropic_stress
    ).mean(
        dim=-1
    )  # Mean over components
    stress_isotropic_loss = torch.mean(stress_isotropic_loss)  # Mean over batch

    stress_anisotropic_loss = stress_loss_fn(
        pred=pred_anisotropic_stress, target=true_anisotropic_stress
    ).mean(
        dim=-1
    )  # Mean over components
    stress_anisotropic_loss = torch.mean(stress_anisotropic_loss)  # Mean over batch

    total_loss = (
        energy_loss + force_loss + stress_isotropic_loss + stress_anisotropic_loss
    )

    loss_dict = {
        "total_loss": total_loss,
        "energy_loss": energy_loss,
        "force_loss": force_loss,
        "stress_iso_loss": stress_isotropic_loss,
        "stress_aniso_loss": stress_anisotropic_loss,
    }

    return loss_dict
