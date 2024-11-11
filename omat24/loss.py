import torch.nn as nn


def compute_mse_loss(
    pred_forces, pred_energy, pred_stress, true_forces, true_energy, true_stress, mask
):
    """Compute the MSE loss for forces, energy, and stress, considering the mask."""
    # Mask out padded atoms
    mask = mask.unsqueeze(-1)  # Shape: [batch_size, max_atoms, 1]
    pred_forces = pred_forces * mask.float()
    true_forces = true_forces * mask.float()

    # Compute MSE losses
    force_loss = nn.MSELoss()(pred_forces, true_forces)
    energy_loss = nn.MSELoss()(pred_energy, true_energy)
    stress_loss = nn.MSELoss()(pred_stress, true_stress)

    return force_loss + energy_loss + stress_loss


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, prediction, target):
        cosine_sim = self.cos(prediction, target)
        mean_cosine_sim = cosine_sim.mean()
        loss = 1 - mean_cosine_sim
        return loss
