import torch.nn as nn


def compute_mae_loss(
    pred_forces, pred_energy, pred_stress, true_forces, true_energy, true_stress, mask
):
    """Compute the MAE loss for forces, energy, and stress, considering the mask."""
    # Mask out padded atoms
    mask = mask.unsqueeze(-1)  # Shape: [batch_size, max_atoms, 1]
    pred_forces = pred_forces * mask.float()
    true_forces = true_forces * mask.float()
    # Compute MSE losses
    energy_loss = nn.L1Loss()(pred_energy, true_energy)
    force_loss = nn.L1Loss()(pred_forces, true_forces)
    stress_loss = nn.L1Loss()(pred_stress, true_stress)
    return 20 * energy_loss + 10 * force_loss + stress_loss


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, prediction, target):
        cosine_sim = self.cos(prediction, target)
        mean_cosine_sim = cosine_sim.mean()
        loss = 1 - mean_cosine_sim
        return loss
