import torch.nn as nn


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
