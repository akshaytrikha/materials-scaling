import torch


def rotate_atom_positions(positions, angle_deg, axis=(0, 0, 1)):
    """Rotate atomic positions by a specified angle around a specified axis.

    Args:
        positions (torch.Tensor): Tensor of shape [N_atoms, 3] containing atomic positions
        angle_deg (float): Rotation angle in degrees
        axis (tuple): Normalized rotation axis (default is z-axis)

    Returns:
        tuple: (rotated_positions, rotation_matrix)
    """
    # Convert angle to radians
    angle_rad = torch.tensor(angle_deg * (torch.pi / 180.0), device=positions.device)

    # Convert axis to tensor and normalize
    axis = torch.tensor(axis, dtype=torch.float, device=positions.device)
    axis = axis / torch.norm(axis)

    # Rodrigues' rotation formula components using quaternions
    a = torch.cos(angle_rad / 2)
    b, c, d = -axis * torch.sin(angle_rad / 2)

    # Quaternion to rotation matrix conversion
    R = torch.tensor(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
        ],
        dtype=torch.float,
        device=positions.device,
    )

    # Apply rotation
    rotated_positions = positions @ R.T

    return rotated_positions, R


def random_rotate_atom_positions(positions):
    """Apply a random rotation to atomic positions.

    Args:
        positions (torch.Tensor): Tensor of shape [N_atoms, 3] containing atomic positions

    Returns:
        tuple: (rotated_positions, rotation_matrix)
    """
    # Generate random rotation axis using PyTorch's random number generators
    phi = torch.rand(1, device=positions.device) * 2 * torch.pi
    costheta = torch.rand(1, device=positions.device) * 2 - 1
    theta = torch.acos(costheta)

    # Convert spherical to Cartesian coordinates
    axis = (
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    )

    # Random angle (0 to 360 degrees)
    angle_deg = torch.rand(1, device=positions.device) * 360

    # Use the deterministic rotation function
    return rotate_atom_positions(positions, angle_deg.item(), axis)


def rotate_stress(stress, R):
    """Rotate a stress tensor by applying a rotation matrix.
    Args:
        stress (torch.Tensor): Stress tensor in Voigt notation [xx, yy, zz, yz, xz, xy]
                              or batch of stress tensors with shape [batch_size, 6]
        R (torch.Tensor): 3x3 rotation matrix
    Returns:
        torch.Tensor: Rotated stress tensor in Voigt notation with same shape as input
    """
    # Handle case where stress is a batch of tensors (shape [batch, 6])
    input_is_batched = stress.dim() > 1
    if input_is_batched:
        # Extract the first (and only) tensor from the batch
        stress_unbatched = stress.squeeze(0)
    else:
        stress_unbatched = stress

    # Convert from Voigt notation to 3x3 matrix
    stress_matrix = torch.tensor(
        [
            [
                stress_unbatched[0],
                stress_unbatched[5],
                stress_unbatched[4],
            ],  # xx, xy, xz
            [
                stress_unbatched[5],
                stress_unbatched[1],
                stress_unbatched[3],
            ],  # xy, yy, yz
            [
                stress_unbatched[4],
                stress_unbatched[3],
                stress_unbatched[2],
            ],  # xz, yz, zz
        ],
        dtype=torch.float,
        device=stress.device if hasattr(stress, "device") else None,
    )

    # Apply the tensor transformation: R @ stress @ R.T
    stress_matrix = R @ stress_matrix @ R.T

    # Convert back to Voigt notation
    rotated_stress = torch.tensor(
        [
            stress_matrix[0, 0],  # xx
            stress_matrix[1, 1],  # yy
            stress_matrix[2, 2],  # zz
            stress_matrix[1, 2],  # yz
            stress_matrix[0, 2],  # xz
            stress_matrix[0, 1],  # xy
        ],
        dtype=torch.float,
        device=stress.device if hasattr(stress, "device") else None,
    )

    # If input was a batch tensor, return a batch tensor
    if input_is_batched:
        rotated_stress = rotated_stress.unsqueeze(0)

    return rotated_stress


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

    # # Create inverse distance matrix with zeros on diagonal
    # EPS = 1e-8
    # D_inv = 1.0 / (D + EPS)

    # # Compute SVD
    # U, s, Vt = torch.linalg.svd(D_inv)

    # # Take first k components
    # k = min(1, D.size(0))
    # U_k = U[:, :k]  # n x k matrix
    # s_k = s[:k]  # k singular values

    # # Compute left matrix: U * sqrt(Sigma)
    # s_sqrt = torch.sqrt(s_k)
    # left_matrix = U_k * s_sqrt[None, :]  # Broadcasting to multiply each column

    N_atoms = D.shape[0]
    return torch.zeros((N_atoms, 5))
