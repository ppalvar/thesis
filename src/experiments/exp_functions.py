import logging
from typing import Tuple

import numpy as np
from src.experiments import SynchrosqueezedResult

IMG_DTYPE = np.float32

log = logging.getLogger()


def threshold_coefficients(
    ssq_result: SynchrosqueezedResult, threshold: float
) -> SynchrosqueezedResult:
    """
    Threshold curvelet coefficients by setting small values to zero.

    Args:
        coefTensor (np.ndarray): Curvelet coefficient tensor from ss_ct2_fwd
        threshold (float): Threshold value for coefficient magnitude

    Modifies:
        Zeroes out coefficients where absolute value < threshold

    Effect:
        - Reduces noise by removing weak coefficients
        - May lose subtle textures/weak edges
        - Improves SNR but risks oversmoothing
    """
    coefTensor = ssq_result.coeff_tensor
    coefTensor = np.where(np.abs(coefTensor) < threshold, 0, coefTensor)
    return ssq_result._replace(coeff_tensor=coefTensor)  # type: ignore


def enhance_energy(
    ssq_result: SynchrosqueezedResult, enhancement_factor: float = 2.0
) -> SynchrosqueezedResult:
    """
    Amplify high-energy regions in synchrosqueezed energy distribution.

    Args:
        ssq_result: SynchrosqueezedResult containing energy distribution and other components
        enhancement_factor: Multiplication factor for energy values (default: 2.0)

    Returns:
        New SynchrosqueezedResult with modified energy values

    Effect:
        - Boosts dominant features (edges, lesions)
        - May create artifacts in homogeneous regions
        - Increases contrast but risks oversaturation
        - Preserves all other components (coeff_cell, coeff_tensor, vecx, vecy)
    """
    if ssq_result.ss_energy.dtype is not IMG_DTYPE:
        log.error(f"The dtype is {ssq_result.ss_energy.dtype}")
    enhanced_energy = ssq_result.ss_energy * np.array(
        enhancement_factor, dtype=IMG_DTYPE
    )
    return ssq_result._replace(ss_energy=enhanced_energy)  # type: ignore


def _apply_mask(matrix, mask):
    return matrix * mask.reshape(*mask.shape, 1, 1)


def mask_frequency_bands(
    ssq_result: SynchrosqueezedResult,
    scale_mask: np.ndarray,
) -> SynchrosqueezedResult:
    """
    Apply frequency band masking to curvelet coefficients in a SynchrosqueezedResult.

    Args:
        ssq_result: SynchrosqueezedResult containing curvelet coefficients
        scale_mask: Boolean mask for target scales (shape: (n_scales,))

    Returns:
        New SynchrosqueezedResult with masked coefficients

    Effect:
        - Isolates features at specific scales
        - Can remove high-frequency noise or low-frequency artifacts
        - Helps analyze scale-specific characteristics
        - Preserves all other components (ss_energy, coeff_cell, vecx, vecy)
    """
    # Create a copy of the coefficient tensor to avoid modifying the original
    masked_coeff_tensor = _apply_mask(
        ssq_result.ss_energy, scale_mask.astype(IMG_DTYPE)
    )

    return ssq_result._replace(ss_energy=masked_coeff_tensor)  # type: ignore


def create_highpass_mask(shape: Tuple[int, int], cutoff_scale: int) -> np.ndarray:
    """
    Generates a high-pass mask preserving the finest (highest frequency) scales up to cutoff_scale.

    Args:
        shape: (N, W) tuple specifying number of angles (W) and scales (N)
        cutoff_scale: All scales finer than this index (0 <= cutoff_scale < N) are preserved

    Returns:
        mask: (N, W) array with 1s in preserved scales, 0s elsewhere

    Improves image quality by:
        - Enhancing edge definition and fine details critical for tumor delineation
        - Increasing visibility of small pathological structures
        - Reducing low-frequency background variations
    """
    mask = np.zeros(shape, dtype=np.float32)
    mask[:cutoff_scale, :] = 1.0
    return mask


def create_bandpass_mask(
    shape: Tuple[int, int], low_scale: int, high_scale: int
) -> np.ndarray:
    """
    Generates a band-pass mask preserving scales between low_scale (inclusive) and high_scale (exclusive).

    Args:
        shape: (N, W) tuple specifying number of angles (W) and scales (N)
        low_scale: Starting scale index (0 <= low_scale < N)
        high_scale: Ending scale index (low_scale < high_scale <= N)

    Returns:
        mask: (N, W) array with 1s in preserved scale range, 0s elsewhere

    Improves image quality by:
        - Isolating mid-frequency components containing tumor texture information
        - Suppressing high-frequency noise and low-frequency artifacts
        - Enhancing contrast in regions showing pathological tissue characteristics
    """
    mask = np.zeros(shape, dtype=np.float32)
    mask[low_scale:high_scale, :] = 1.0
    return mask


def create_gaussian_scale_mask(
    shape: Tuple[int, int], center_scale: int, sigma: float
) -> np.ndarray:
    """
    Generates a Gaussian-weighted mask emphasizing scales around center_scale.

    Args:
        shape: (N, W) tuple specifying number of angles (W) and scales (N)
        center_scale: Scale index receiving maximum weight (0 <= center_scale < N)
        sigma: Standard deviation controlling the width of the Gaussian

    Returns:
        mask: (N, W) array with Gaussian weights applied across scales

    Improves image quality by:
        - Smoothly emphasizing features within a target frequency band
        - Reducing ringing artifacts from abrupt frequency cuts
        - Preserving natural appearance while enhancing lesion contrast
    """
    N, W = shape
    scale_indices = np.arange(N)
    weights = np.exp(-0.5 * ((scale_indices - center_scale) / sigma) ** 2)
    weights /= weights.max()  # Normalize peak to 1.0
    mask = np.zeros(shape, dtype=np.float32)
    for w in range(W):
        mask[:, w] = weights
    return mask


def create_exponential_scale_mask(
    shape: Tuple[int, int], decay_rate: float
) -> np.ndarray:
    """
    Generates a mask with exponentially decreasing weights from finest to coarsest scales.

    Args:
        shape: (N, W) tuple specifying number of angles (W) and scales (N)
        decay_rate: Exponential decay factor (higher values steeper decay)

    Returns:
        mask: (N, W) array with exponential weights in scale dimension

    Improves image quality by:
        - Prioritizing finest scales while softly suppressing coarser ones
        - Maintaining some contribution from lower frequencies for structural context
    """
    N, W = shape
    scales = np.arange(N)
    weights = np.exp(-decay_rate * scales)
    weights /= weights.max()  # Normalize max to 1
    mask = np.zeros(shape, dtype=np.float32)
    for w in range(W):
        mask[:, w] = weights
    return mask
