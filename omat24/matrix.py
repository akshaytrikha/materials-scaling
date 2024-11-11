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


def factorize_matrix(distance_matrix: torch.Tensor) -> tuple:
    """Factorize the distance matrix using Singular Value Decomposition (SVD).

    Args:
        distance_matrix (torch.Tensor): Tensor of shape [N_atoms, N_atoms].

    Returns:
        tuple:
            - U is a [N_atoms, N_atoms] orthogonal matrix,
            - S is a [N_atoms] vector of singular values,
            - Vh is a [N_atoms, N_atoms] orthogonal matrix (transpose of V).
    """
    if distance_matrix.dim() != 2 or distance_matrix.size(0) != distance_matrix.size(1):
        raise ValueError("Distance matrix must be a square 2D tensor.")

    # Perform SVD
    U, S, Vh = torch.linalg.svd(distance_matrix, full_matrices=True)

    return U, S, Vh


def low_rank_approximation(distance_matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Generate a low-rank approximation of the distance matrix using singular value decomposition (SVD).

    Args:
        distance_matrix (torch.Tensor): Tensor of shape [N_atoms, N_atoms].
        rank (int): The target rank for the approximation.

    Returns:
        torch.Tensor: Low-rank approximated distance matrix of shape [N_atoms, N_atoms].
    """
    if rank <= 0 or rank > distance_matrix.size(0):
        raise ValueError(f"Rank must be between 1 and {distance_matrix.size(0)}.")

    # Factorize the matrix
    U, S, Vh = factorize_matrix(distance_matrix)

    # Truncate to the desired rank
    U_reduced = U[:, :rank]
    S_reduced = S[:rank]
    Vh_reduced = Vh[:rank, :]

    # Reconstruct the low-rank approximation
    # Note: S needs to be a diagonal matrix for reconstruction
    S_reduced_diag = torch.diag(S_reduced)
    low_rank_matrix = U_reduced @ S_reduced_diag @ Vh_reduced

    return low_rank_matrix
