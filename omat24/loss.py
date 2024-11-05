import torch.nn as nn


def compute_mse_loss(pred_forces, pred_energy, pred_stress, batch):
    """Compute the MSE loss for forces, energy, and stress and cosine similarity loss for forces."""
    # Compute MSE loss for forces, energy, and stress
    force_loss = nn.MSELoss()(pred_forces, batch["forces"])
    energy_loss = nn.MSELoss()(pred_energy, batch["energy"])
    stress_loss = nn.MSELoss()(pred_stress, batch["stress"])

    return force_loss + energy_loss + stress_loss


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1):
        """
        Args:
            dim (int): Dimension along which to compute cosine similarity.
        """
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, prediction, rotated_prediction):
        """Forward pass to compute the loss.

        Args:
            output (torch.Tensor): Predicted force matrix of shape (n_atoms, 3).
            target (torch.Tensor): Target force matrix of shape (n_atoms, 3).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute cosine similarity for each corresponding row
        cosine_sim = self.cos(prediction, rotated_prediction)  # Shape: (n_atoms,)

        # Compute the mean cosine similarity
        mean_cosine_sim = cosine_sim.mean()

        # Define loss as 1 - mean cosine similarity
        loss = 1 - mean_cosine_sim

        return loss
