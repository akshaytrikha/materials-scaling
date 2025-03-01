"""
Gaunt Tensor Product Accelerator for EquiformerV2.

This module implements the Gaunt Tensor Product approach to accelerate equivariant operations
in EquiformerV2. The approach converts spherical harmonic representations to 2D Fourier bases,
performs fast convolution using FFT, and converts back to spherical harmonics.

This reduces the computational complexity of tensor products from O(L^6) to O(L^3),
where L is the maximum degree of spherical harmonics.
"""

# External
import torch
import torch.fft as fft
import torch.nn as nn
import numpy as np
from typing import List


def precompute_sh2f_bases(lmax: int, device=None) -> torch.Tensor:
    """
    Precompute the transformation matrices from spherical harmonics to 2D Fourier bases.

    Args:
        lmax (int): Maximum degree of spherical harmonics
        device: Device to place the precomputed bases on

    Returns:
        torch.Tensor: Precomputed bases with shape (L, 2L-1, 2L-1, 2)
    """
    # Placeholder for a more complex calculation - in practice this would use
    # mathematical formulas to compute the actual transformation matrices
    L = lmax + 1
    grid_size = 2 * L - 1

    # This would typically involve mathematical calculations based on
    # spherical harmonic coefficients and their relation to Fourier series
    bases = torch.zeros((L, grid_size, grid_size, 2), device=device)

    # For demonstration, we'll initialize with random values
    # In practice, these would be specific mathematical values
    for l in range(L):
        # Generate different patterns for different l values
        for i in range(grid_size):
            for j in range(grid_size):
                # Create structured patterns that would be based on spherical harmonics
                bases[l, i, j, 0] = (
                    0.5
                    * np.sin(np.pi * l * i / grid_size)
                    * np.cos(np.pi * l * j / grid_size)
                )
                bases[l, i, j, 1] = (
                    0.5
                    * np.cos(np.pi * l * i / grid_size)
                    * np.sin(np.pi * l * j / grid_size)
                )

    return bases


def precompute_f2sh_bases(lmax: int, device=None) -> torch.Tensor:
    """
    Precompute the transformation matrices from 2D Fourier bases to spherical harmonics.

    Args:
        lmax (int): Maximum degree of spherical harmonics
        device: Device to place the precomputed bases on

    Returns:
        torch.Tensor: Precomputed bases with shape (L, 2L-1, 2L-1, 2)
    """
    # In a full implementation, f2sh would be the mathematical inverse of sh2f
    # Here we're creating a placeholder that approximately represents this relationship
    L = lmax + 1
    grid_size = 2 * L - 1

    # Initialize bases with pseudo-inverse values
    bases = torch.zeros((L, grid_size, grid_size, 2), device=device)

    # For demonstration, approximate inverse relationship
    for l in range(L):
        for i in range(grid_size):
            for j in range(grid_size):
                bases[l, i, j, 0] = (
                    0.5
                    * np.sin(np.pi * l * i / grid_size)
                    * np.cos(np.pi * l * j / grid_size)
                )
                bases[l, i, j, 1] = (
                    0.5
                    * np.cos(np.pi * l * i / grid_size)
                    * np.sin(np.pi * l * j / grid_size)
                )

    # Normalize to make it closer to an inverse
    norm = torch.sum(bases * bases, dim=(1, 2), keepdim=True)
    bases = bases / (norm + 1e-7)

    return bases


def sh2f(sh_coeff: torch.Tensor, sh2f_bases: torch.Tensor) -> torch.Tensor:
    """
    Convert from spherical harmonics to 2D Fourier bases.

    Args:
        sh_coeff: Coefficients of spherical harmonics, shape (L, 2L-1)
        sh2f_bases: Precomputed bases, shape (L, 2L-1, 2L-1, 2)

    Returns:
        torch.Tensor: Coefficients of 2D Fourier bases, shape (2L-1, 2L-1)
    """
    sum_along_L = (sh_coeff.unsqueeze(-1).unsqueeze(-1) * sh2f_bases).sum(
        dim=0
    )  # (2L-1, 2L-1, 2)
    res = ((sum_along_L[:, :, 0] + sum_along_L[:, :, 1].flip(dims=[0]))).permute(
        1, 0
    )  # (2L-1, 2L-1)
    return res


def sh2f_channel(sh_coeff: torch.Tensor, sh2f_bases: torch.Tensor) -> torch.Tensor:
    """
    Convert from spherical harmonics to 2D Fourier bases with multiple channels.

    Args:
        sh_coeff: Coefficients of spherical harmonics, shape (C, L, 2L-1)
        sh2f_bases: Precomputed bases, shape (L, 2L-1, 2L-1, 2)

    Returns:
        torch.Tensor: Coefficients of 2D Fourier bases, shape (C, 2L-1, 2L-1)
    """
    sum_along_L = (sh_coeff.unsqueeze(-1).unsqueeze(-1) * sh2f_bases.unsqueeze(0)).sum(
        dim=1
    )  # (C, 2L-1, 2L-1, 2)
    res = ((sum_along_L[:, :, :, 0] + sum_along_L[:, :, :, 1].flip(dims=[1]))).permute(
        0, 2, 1
    )  # (C, 2L-1, 2L-1)
    return res


def f2sh(fourier_coef: torch.Tensor, f2sh_bases: torch.Tensor) -> torch.Tensor:
    """
    Convert from 2D Fourier bases to spherical harmonics.

    Args:
        fourier_coef: Coefficients of 2D Fourier bases, shape (2L-1, 2L-1)
        f2sh_bases: Precomputed bases, shape (L, 2L-1, 2L-1, 2)

    Returns:
        torch.Tensor: Coefficients of spherical harmonics, shape (L, 2L-1)
    """
    fourier_coef_t_first = fourier_coef.permute(1, 0)
    sum_positive = (fourier_coef_t_first.unsqueeze(0) * f2sh_bases[:, :, :, 0]).sum(
        dim=-1
    )  # (L, 2L-1)
    sum_negative = (
        fourier_coef_t_first.flip(dims=[0]).unsqueeze(0) * f2sh_bases[:, :, :, 1]
    ).sum(
        dim=-1
    )  # (L, 2L-1)
    res = sum_positive + sum_negative
    return res


def f2sh_channel(fourier_coef: torch.Tensor, f2sh_bases: torch.Tensor) -> torch.Tensor:
    """
    Convert from 2D Fourier bases to spherical harmonics with multiple channels.

    Args:
        fourier_coef: Coefficients of 2D Fourier bases, shape (C, 2L-1, 2L-1)
        f2sh_bases: Precomputed bases, shape (L, 2L-1, 2L-1, 2)

    Returns:
        torch.Tensor: Coefficients of spherical harmonics, shape (C, L, 2L-1)
    """
    fourier_coef_t_first = fourier_coef.permute(0, 2, 1)
    sum_positive = (
        fourier_coef_t_first.unsqueeze(1) * f2sh_bases[:, :, :, 0].unsqueeze(0)
    ).sum(
        dim=-1
    )  # (C, L, 2L-1)
    sum_negative = (
        fourier_coef_t_first.flip(dims=[1]).unsqueeze(1)
        * f2sh_bases[:, :, :, 1].unsqueeze(0)
    ).sum(
        dim=-1
    )  # (C, L, 2L-1)
    res = sum_positive + sum_negative
    return res


def fft_convolution(
    fourier_coef1: torch.Tensor, fourier_coef2: torch.Tensor, return_real: bool = False
) -> torch.Tensor:
    """
    Perform 2D convolution via Fast Fourier Transform.

    Args:
        fourier_coef1: First set of Fourier coefficients, shape (2L-1, 2L-1)
        fourier_coef2: Second set of Fourier coefficients, shape (2L-1, 2L-1)
        return_real: Whether to return only the real part

    Returns:
        torch.Tensor: Result of the convolution, shape (2L-1 + 2L-1 - 1, 2L-1 + 2L-1 - 1)
    """
    # Step 0: preparation
    in_shape1, in_shape2 = fourier_coef1.shape[0], fourier_coef2.shape[0]
    out_shape = in_shape1 + in_shape2 - 1
    in1 = torch.zeros(
        (out_shape, out_shape), dtype=fourier_coef1.dtype, device=fourier_coef1.device
    )
    in2 = torch.zeros(
        (out_shape, out_shape), dtype=fourier_coef2.dtype, device=fourier_coef2.device
    )
    in1[:in_shape1, :in_shape1] = fourier_coef1
    in2[:in_shape2, :in_shape2] = fourier_coef2

    # Step 1: 2D Discrete Fourier Transform: transform into the frequency domain
    fourier_coef1_freq, fourier_coef2_freq = fft.fft2(in1), fft.fft2(in2)

    # Step 2: Element-wise multiplication in the frequency domain
    res_freq = fourier_coef1_freq * fourier_coef2_freq

    # Step 3: 2D Inverse Discrete Fourier Transform: transform back from the frequency domain
    res = fft.ifft2(res_freq).real if return_real else fft.ifft2(res_freq)

    return res


def fft_convolution_channel(
    fourier_coef1: torch.Tensor, fourier_coef2: torch.Tensor, return_real: bool = False
) -> torch.Tensor:
    """
    Perform 2D convolution via Fast Fourier Transform with multiple channels.

    Args:
        fourier_coef1: First set of Fourier coefficients, shape (C, 2L-1, 2L-1)
        fourier_coef2: Second set of Fourier coefficients, shape (C, 2L-1, 2L-1)
        return_real: Whether to return only the real part

    Returns:
        torch.Tensor: Result of the convolution, shape (C, 2L-1 + 2L-1 - 1, 2L-1 + 2L-1 - 1)
    """
    # Step 0: preparation
    C = fourier_coef1.shape[0]
    in_shape1, in_shape2 = fourier_coef1.shape[1], fourier_coef2.shape[1]
    out_shape = in_shape1 + in_shape2 - 1
    in1 = torch.zeros(
        (C, out_shape, out_shape),
        dtype=fourier_coef1.dtype,
        device=fourier_coef1.device,
    )
    in2 = torch.zeros(
        (C, out_shape, out_shape),
        dtype=fourier_coef2.dtype,
        device=fourier_coef2.device,
    )
    in1[:, :in_shape1, :in_shape1] = fourier_coef1
    in2[:, :in_shape2, :in_shape2] = fourier_coef2

    # Step 1: 2D Discrete Fourier Transform: transform into the frequency domain
    fourier_coef1_freq, fourier_coef2_freq = fft.fft2(in1), fft.fft2(in2)

    # Step 2: Element-wise multiplication in the frequency domain
    res_freq = fourier_coef1_freq * fourier_coef2_freq

    # Step 3: 2D Inverse Discrete Fourier Transform: transform back from the frequency domain
    res = fft.ifft2(res_freq).real if return_real else fft.ifft2(res_freq)

    return res


class GauntTensorProduct(nn.Module):
    """
    Module implementing the Gaunt Tensor Product for accelerating equivariant operations.
    """

    def __init__(
        self, lmax_list: List[int], mmax_list: List[int], channels: int, device=None
    ):
        """
        Initialize the Gaunt Tensor Product module.

        Args:
            lmax_list: List of maximum degrees of spherical harmonics
            mmax_list: List of maximum orders of spherical harmonics
            channels: Number of channels
            device: Device to place the precomputed bases on
        """
        super().__init__()

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.channels = channels
        self.device = device

        # Precompute bases for each lmax
        self.sh2f_bases = nn.ParameterList()
        self.f2sh_bases = nn.ParameterList()

        for lmax in lmax_list:
            sh2f_base = precompute_sh2f_bases(lmax, device)
            f2sh_base = precompute_f2sh_bases(lmax, device)

            # Register as buffers to ensure they're moved to the correct device
            self.register_buffer(f"sh2f_base_{lmax}", sh2f_base)
            self.register_buffer(f"f2sh_base_{lmax}", f2sh_base)

    def get_sh2f_base(self, lmax: int) -> torch.Tensor:
        """Get the precomputed sh2f base for a given lmax"""
        return getattr(self, f"sh2f_base_{lmax}")

    def get_f2sh_base(self, lmax: int) -> torch.Tensor:
        """Get the precomputed f2sh base for a given lmax"""
        return getattr(self, f"f2sh_base_{lmax}")

    def tensor_product(
        self, x: torch.Tensor, y: torch.Tensor, lmax: int
    ) -> torch.Tensor:
        """
        Perform tensor product using the Gaunt Tensor Product approach.

        Args:
            x: First input tensor of spherical harmonic coefficients
            y: Second input tensor of spherical harmonic coefficients
            lmax: Maximum degree of spherical harmonics

        Returns:
            torch.Tensor: Result of the tensor product
        """
        # Get the precomputed bases
        sh2f_base = self.get_sh2f_base(lmax)
        f2sh_base = self.get_f2sh_base(lmax)

        # Convert from spherical harmonics to 2D Fourier basis
        x_fourier = sh2f(x, sh2f_base)
        y_fourier = sh2f(y, sh2f_base)

        # Perform the convolution using FFT
        result_fourier = fft_convolution(x_fourier, y_fourier)

        # Convert back from 2D Fourier basis to spherical harmonics
        result = f2sh(result_fourier, f2sh_base)

        return result

    def tensor_product_channel(
        self, x: torch.Tensor, y: torch.Tensor, lmax: int
    ) -> torch.Tensor:
        """
        Perform tensor product with multiple channels using the Gaunt Tensor Product approach.

        Args:
            x: First input tensor of spherical harmonic coefficients with shape (C, L, 2L-1)
            y: Second input tensor of spherical harmonic coefficients with shape (C, L, 2L-1)
            lmax: Maximum degree of spherical harmonics

        Returns:
            torch.Tensor: Result of the tensor product with shape (C, L, 2L-1)
        """
        # Get the precomputed bases
        sh2f_base = self.get_sh2f_base(lmax)
        f2sh_base = self.get_f2sh_base(lmax)

        # Convert from spherical harmonics to 2D Fourier basis
        x_fourier = sh2f_channel(x, sh2f_base)
        y_fourier = sh2f_channel(y, sh2f_base)

        # Perform the convolution using FFT
        result_fourier = fft_convolution_channel(x_fourier, y_fourier)

        # Convert back from 2D Fourier basis to spherical harmonics
        result = f2sh_channel(result_fourier, f2sh_base)

        return result
