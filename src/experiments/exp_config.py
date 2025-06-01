from src.experiments import ExperimentConfig
from src.experiments.exp_functions import (
    create_bandpass_mask,
    create_exponential_scale_mask,
    create_gaussian_scale_mask,
    create_highpass_mask,
    enhance_energy,
    mask_frequency_bands,
    threshold_coefficients,
)


experiments = [
    ExperimentConfig(
        name="Threshold Coefficients at 0.1",
        function=threshold_coefficients,
        params=lambda x: (),
        args=(),
        kwargs={"threshold": 0.1},
    ),
    ExperimentConfig(
        name="Threshold Coefficients at 0.05",
        function=threshold_coefficients,
        params=lambda x: (),
        args=(),
        kwargs={"threshold": 0.05},
    ),
    ExperimentConfig(
        name="Threshold Coefficients at 0.001",
        function=threshold_coefficients,
        params=lambda x: (),
        args=(),
        kwargs={"threshold": 0.001},
    ),
    ExperimentConfig(
        name="Enhance Energy by 3.0",
        function=enhance_energy,
        params=lambda x: (),
        args=(),
        kwargs={"enhancement_factor": 3.0},
    ),
    ExperimentConfig(
        name="Enhance Energy by 1.5",
        function=enhance_energy,
        params=lambda x: (),
        args=(),
        kwargs={"enhancement_factor": 1.5},
    ),
    ExperimentConfig(
        name="Enhance Energy by 10",
        function=enhance_energy,
        params=lambda x: (),
        args=(),
        kwargs={"enhancement_factor": 10},
    ),
    ExperimentConfig(
        name="Highpass Mask with cutoff_scale=5",
        function=mask_frequency_bands,
        params=lambda x: (
            create_highpass_mask(x.ss_energy.shape[:2], x.ss_energy.shape[1] - 5),
        ),
        args=(),
        kwargs={},
    ),
    ExperimentConfig(
        name="Highpass Mask with cutoff_scale=10",
        function=mask_frequency_bands,
        params=lambda x: (
            create_highpass_mask(x.ss_energy.shape[:2], x.ss_energy.shape[1] - 10),
        ),
        args=(),
        kwargs={},
    ),
    ExperimentConfig(
        name="Bandpass Mask with low_scale=5, high_scale=10",
        function=mask_frequency_bands,
        params=lambda x: (
            create_bandpass_mask(x.ss_energy.shape[:2], 5, x.ss_energy.shape[2] - 10),
        ),
        args=(),
        kwargs={},
    ),
    ExperimentConfig(
        name="Bandpass Mask with low_scale=15, high_scale=20",
        function=mask_frequency_bands,
        params=lambda x: (
            create_bandpass_mask(x.ss_energy.shape[:2], 15, x.ss_energy.shape[2] - 20),
        ),
        args=(),
        kwargs={},
    ),
    ExperimentConfig(
        name="Gaussian Scale Mask with center_scale=N/2, sigma=1.0",
        function=mask_frequency_bands,
        params=lambda x: (
            create_gaussian_scale_mask(
                x.ss_energy.shape[:2], x.ss_energy.shape[1] // 2, 1.0
            ),
        ),
        args=(),
        kwargs={},
    ),
    ExperimentConfig(
        name="Gaussian Scale Mask with center_scale=N/2, sigma=4.0",
        function=mask_frequency_bands,
        params=lambda x: (
            create_gaussian_scale_mask(
                x.ss_energy.shape[:2], x.ss_energy.shape[1] // 2, 4.0
            ),
        ),
        args=(),
        kwargs={},
    ),
    ExperimentConfig(
        name="Gaussian Scale Mask with center_scale=N/2, sigma=6.0",
        function=mask_frequency_bands,
        params=lambda x: (
            create_gaussian_scale_mask(
                x.ss_energy.shape[:2], x.ss_energy.shape[1] // 2, 6.0
            ),
        ),
        args=(),
        kwargs={},
    ),
    ExperimentConfig(
        name="Exponential Scale Mask with decay_rate=0.1",
        function=mask_frequency_bands,
        params=lambda x: (create_exponential_scale_mask(x.ss_energy.shape[:2], 0.1),),
        args=(),
        kwargs={},
    ),
    ExperimentConfig(
        name="Exponential Scale Mask with decay_rate=0.01",
        function=mask_frequency_bands,
        params=lambda x: (create_exponential_scale_mask(x.ss_energy.shape[:2], 0.01),),
        args=(),
        kwargs={},
    ),
]
