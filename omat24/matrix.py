import torch


def rotate_matrix(
    matrices: torch.Tensor, k: int = 1, dims: tuple = (0, 1)
) -> torch.Tensor:
    """
    Rotate a matrix or a batch of matrices by multiples of 90 degrees.

    Args:
        matrices (torch.Tensor):
            - For a single matrix, shape should be [N, M].
            - For a batch of matrices, shape should be [B, N, M], where B is the batch size.
        k (int, optional): Number of times to rotate the matrix by 90 degrees.
                           Positive values rotate counter-clockwise, negative values clockwise. Defaults to 1.
        dims (tuple, optional): The two dimensions to rotate.
                                For 2D matrices, the default is (0, 1).
                                For higher-dimensional tensors, adjust accordingly.
                                Defaults to (0, 1).

    Returns:
        torch.Tensor: Rotated matrix or batch of rotated matrices with the same shape as input.
    """
    if matrices.dim() == 2:
        # Single matrix: [N, M]
        rotated = torch.rot90(matrices, k=k, dims=dims)
    elif matrices.dim() == 3:
        # Move batch dimension to the front if not already
        rotated = torch.rot90(matrices, k=k, dims=dims)
    else:
        raise ValueError("Input tensor must be 2D or 3D (batch of matrices).")

    return rotated


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
