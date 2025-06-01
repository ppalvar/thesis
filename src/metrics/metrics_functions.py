from typing import Optional
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import torch
from piq import brisque, fsim

loss_fn = lpips.LPIPS(net="alex").eval()


def contrast_improvement_index(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Compute Contrast Improvement Index (CII) for grayscale images.

    Args:
        original: Original grayscale image (H, W) in [0, 255] uint8
        enhanced: Enhanced grayscale image (H, W) in [0, 255] uint8

    Returns:
        CII ratio (enhanced/original). Values >1 indicate improvement.
    """
    return float(np.std(enhanced) / np.std(original))


def laplacian_sharpness_ratio(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Compute sharpness ratio using Laplacian variance for grayscale.

    Args:
        original: Original grayscale image (H, W) in [0, 255] uint8
        enhanced: Enhanced grayscale image (H, W) in [0, 255] uint8

    Returns:
        Sharpness ratio (enhanced/original). Values >1 indicate improvement.
    """

    def _calc_sharpness(img: np.ndarray) -> float:
        return cv2.Laplacian(img, cv2.CV_32F).var()

    return float(_calc_sharpness(enhanced) / _calc_sharpness(original))


def fsim_score(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Compute FSIM for grayscale images using piq.

    Args:
        original: Original grayscale image (H, W) in [0, 255] uint8
        enhanced: Enhanced grayscale image (H, W) in [0, 255] uint8

    Returns:
        FSIM score [0-1]. Higher values indicate better quality.
    """

    original_tensor = (
        torch.tensor(original, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    enhanced_tensor = (
        torch.tensor(enhanced, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )

    original_rgb = original_tensor.repeat(1, 3, 1, 1)
    enhanced_rgb = enhanced_tensor.repeat(1, 3, 1, 1)

    return fsim(original_rgb, enhanced_rgb).item()


def psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Compute PSNR for grayscale images.

    Args:
        original: Original grayscale image (H, W) in [0, 255] uint8
        enhanced: Enhanced grayscale image (H, W) in [0, 255] uint8

    Returns:
        PSNR in dB. Higher values indicate less distortion.
    """
    return float(peak_signal_noise_ratio(original, enhanced, data_range=255))


def ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Compute SSIM for grayscale images.

    Args:
        original: Original grayscale image (H, W) in [0, 255] uint8
        enhanced: Enhanced grayscale image (H, W) in [0, 255] uint8

    Returns:
        SSIM score [0-1]. Values closer to 1 indicate better structural similarity.
    """
    return float(structural_similarity(original, enhanced, data_range=255))  # type: ignore


def lpips_score(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Compute LPIPS for grayscale images (converts to RGB-like format).

    Args:
        original: Original grayscale image (H, W) in [0, 255] uint8
        enhanced: Enhanced grayscale image (H, W) in [0, 255] uint8
        net: Backbone network ('alex', 'vgg', 'squeeze')

    Returns:
        LPIPS distance. Lower values indicate better perceptual similarity.
    """

    def _preprocess(img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 127.5 - 1  # [-1, 1] range
        return (
            torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        )  # Add channel and batch dims

    with torch.no_grad():
        return loss_fn(
            _preprocess(original).repeat(1, 3, 1, 1),  # Repeat grayscale to 3 channels
            _preprocess(enhanced).repeat(1, 3, 1, 1),
        ).item()


def residual_noise_std(
    original: np.ndarray, 
    enhanced: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute residual noise standard deviation for grayscale.
    
    Args:
        original: Original grayscale image (H, W) in [0, 255] uint8
        enhanced: Enhanced grayscale image (H, W) in [0, 255] uint8
        mask: Binary mask indicating smooth regions (H, W)
    
    Returns:
        Standard deviation of noise residuals. Higher values indicate more noise.
    """
    diff = cv2.absdiff(enhanced, original)
    if mask is not None:
        diff = diff[mask]
    return float(np.std(diff)) # type: ignore

def brisque_score(image: np.ndarray) -> float:
    """
    Compute BRISQUE for grayscale images (converts to RGB-like format).
    
    Args:
        image: Grayscale image (H, W) in [0, 255] uint8
    
    Returns:
        BRISQUE score. Higher values indicate more artifacts.
    """
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add channel and batch dimensions
    return brisque(tensor.repeat(1,3,1,1)).item()  # Repeat grayscale to 3 channels