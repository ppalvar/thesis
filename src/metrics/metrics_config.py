from src.metrics import MetricConfig
from src.metrics.metrics_functions import (
    brisque_score,
    contrast_improvement_index,
    fsim_score,
    laplacian_sharpness_ratio,
    lpips_score,
    psnr,
    residual_noise_std,
    ssim,
)


metrics = [
    MetricConfig("Improvement::CII", contrast_improvement_index, True),
    MetricConfig("Improvement::LSR", laplacian_sharpness_ratio, True),
    MetricConfig("Improvement::FSIM", fsim_score, True),
    MetricConfig("Distortion::PSNR", psnr, True),
    MetricConfig("Distortion::SSIM", ssim, True),
    MetricConfig("Distortion::LPIPS", lpips_score, True),
    MetricConfig("Artifacts::RNS", residual_noise_std, True),
    MetricConfig("Artifacts::BRISQUE", brisque_score, False),
]
