import numpy as np
import torch
from typing import Tuple


def random_rotate_atoms(positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a random rotation around each axis (x, y, z) to a set of atomic positions using PyTorch.

    Args:
        positions: torch tensor of shape (N, 3), where each row represents the (x, y, z) coordinates of an atom.

    Returns:
        rotated_positions (torch.Tensor): rotated coordinates with shape (N, 3)
        R (torch.Tensor): rotation matrix with shape  (3, 3).
    """
    # Generate random rotation angles in radians for each axis
    theta_x = torch.rand(1) * 2 * torch.pi  # random angle for x-axis
    theta_y = torch.rand(1) * 2 * torch.pi  # random angle for y-axis
    theta_z = torch.rand(1) * 2 * torch.pi  # random angle for z-axi

    # Define the rotation matrices around each axis
    R_x = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(theta_x), -torch.sin(theta_x)],
            [0, torch.sin(theta_x), torch.cos(theta_x)],
        ],
        dtype=positions.dtype,
        device=positions.device,
    )

    R_y = torch.tensor(
        [
            [torch.cos(theta_y), 0, torch.sin(theta_y)],
            [0, 1, 0],
            [-torch.sin(theta_y), 0, torch.cos(theta_y)],
        ],
        dtype=positions.dtype,
        device=positions.device,
    )

    R_z = torch.tensor(
        [
            [torch.cos(theta_z), -torch.sin(theta_z), 0],
            [torch.sin(theta_z), torch.cos(theta_z), 0],
            [0, 0, 1],
        ],
        dtype=positions.dtype,
        device=positions.device,
    )

    # Combine the rotations: first apply x, then y, then z
    R = R_z @ R_y @ R_x

    # Apply the combined rotation matrix to each atomic position
    rotated_positions = positions @ R.T

    return rotated_positions, R


def compute_distance_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise Euclidean / Cartesian distance matrix for a set of atomic positions.

    Args:
        positions (torch.Tensor): Tensor of shape [N_atoms, 3] representing atomic positions.

    Returns:
        torch.Tensor: Pairwise distance matrix of shape [N_atoms, N_atoms].
    """
    if positions.dim() != 2 or positions.size(1) != 3:
        raise ValueError("Positions tensor must be of shape [N_atoms, 3].")

    # Compute pairwise distances using torch.cdist
    distance_matrix = torch.cdist(positions, positions, p=2)

    return distance_matrix


def factorize_matrix(D: torch.Tensor) -> torch.Tensor:
    """Factorize the distance matrix using Singular Value Decomposition (SVD).
    
    Args:
        D (torch.Tensor): Distance matrix tensor of shape [N_atoms, N_atoms].
        
    Returns:
        torch.Tensor: Left matrix (U * sqrt(Sigma)) of shape [N_atoms, k=5]
    """
    if D.dim() != 2 or D.size(0) != D.size(1):
        raise ValueError("Distance matrix must be a square 2D tensor.")
        
    # Create inverse distance matrix with zeros on diagonal
    D_inv = torch.zeros_like(D)
    mask = ~torch.eye(D.shape[0], dtype=bool, device=D.device)
    D_inv[mask] = 1.0 / D[mask]
    
    # Fix k=5 and compute SVD
    k = min(5, D.size(0))
    U, s, Vt = torch.linalg.svd(D_inv)
    
    # Take first k components
    U_k = U[:, :k]  # n x k matrix
    s_k = s[:k]     # k singular values
    
    # Compute left matrix: U * sqrt(Sigma)
    s_sqrt = torch.sqrt(s_k)
    left_matrix = U_k * s_sqrt[None, :]  # Broadcasting to multiply each column
    
    return left_matrix
