import gc
import json
import logging

from collections import namedtuple
from typing import Any, Dict, Iterable, List

import numpy as np

from src.curvelet_sst import Synchrosqueezer
from src.image_utils import array_hash, get_windowed_image, normalize_img
from src.experiments import ExperimentConfig, SynchrosqueezedResult, execute_experiment

log = logging.getLogger()

MetricConfig = namedtuple(
    "MetricConfig", ("name", "metric_function", "is_comparison_metric")
)
MetricResult = namedtuple("MetricResult", ("metric_name", "result"))


def calculate_metrics(
    original_image: np.ndarray,
    enhanced_image: np.ndarray,
    metrics_cfg: List[MetricConfig],
) -> List[MetricResult]:
    """
    Calculates a list of metric results for given original and enhanced images using specified metric configurations.
    Args:
        original_image (np.ndarray): The original input image as a NumPy array.
        enhanced_image (np.ndarray): The enhanced or processed image as a NumPy array.
        metrics_cfg (List[MetricConfig]): A list of MetricConfig objects, each specifying a metric to compute and its configuration.
    Returns:
        List[MetricResult]: A list of MetricResult objects containing the name and value of each calculated metric.
    Raises:
        Exception: Propagates any exception raised during metric calculation.
    Logs:
        - Starts and completes the metrics calculation process.
        - Normalizes input images before metric computation.
        - Logs the calculation and result of each metric.
        - Logs errors if metric calculation fails.
    """

    log.info("📊 Starting metrics calculation...")
    results: List[MetricResult] = []

    log.info("🖼️ Normalizing images...")
    original_norm, _, _ = normalize_img(original_image)
    enhanced_norm, _, _ = normalize_img(enhanced_image)

    for metric_cfg in metrics_cfg:
        log.info(
            f"🧮 Calculating metric: {metric_cfg.name} (comparison: {metric_cfg.is_comparison_metric})"
        )
        try:
            if metric_cfg.is_comparison_metric:
                result = metric_cfg.metric_function(original_norm, enhanced_norm)
            else:
                result = metric_cfg.metric_function(enhanced_norm)

            results.append(MetricResult(metric_cfg.name, result))
            log.info(f"✅ Successfully calculated {metric_cfg.name} = {result:.4f}")
        except Exception as e:
            log.error(f"❌ Failed to calculate metric {metric_cfg.name}: {str(e)}")
            raise

    log.info(f"🏁 Completed metrics calculation. Calculated {len(results)} metrics.")
    return results


def run_experiments_and_metrics(
    images: Iterable[np.ndarray],
    experiments_cfg: List[ExperimentConfig],
    metrics_cfg: List[MetricConfig],
) -> Dict[str, Any]:
    """
    Runs a pipeline of experiments and metrics evaluation on a collection of images.
    For each image in the input iterable, this function:
      - Checks if the image has already been processed (using a hash and cache file).
      - Applies a synchrosqueezed curvelet transform to the image.
      - Runs a series of experiments (as specified in `experiments_cfg`) on the transformed image.
      - Calculates metrics (as specified in `metrics_cfg`) for each experiment result.
      - Collects and returns the results for all images and experiments.
    Logging is performed at each major step, and results are cached to avoid redundant computation.
    Args:
        images (Iterable[np.ndarray]): An iterable of images (as numpy arrays) to process.
        experiments_cfg (List[ExperimentConfig]): List of experiment configurations to run on each image.
        metrics_cfg (List[MetricConfig]): List of metric configurations to evaluate for each experiment.
    Returns:
        Dict[str, Any]: A dictionary mapping image hashes to their corresponding experiment and metric results.
    """

    log.info("🔬 Starting experiments and metrics pipeline...")
    log.info(
        f"📋 Config: {len(experiments_cfg)} experiments, {len(metrics_cfg)} metrics"
    )
    experiment_results = {}
    ssq = Synchrosqueezer()

    try:
        for i, image in enumerate(images):
            log.info(f"🖼️ Processing image {i + 1}")

            img_hash = array_hash(image)

            with open("exp_out/out.json", "r") as f:
                if img_hash in json.load(f):
                    log.info("❇️ Image found in cache")
                    continue

            log.info("🌀 Performing synchrosqueezed curvelet transform...")
            image_norm, _ = get_windowed_image(image)
            _ssq_result = ssq.synchrosqueezed_curvelet_transform(image_norm)
            _ssq_result = SynchrosqueezedResult(*_ssq_result)  # type: ignore

            image_experiments_results = {}

            for exp_cfg in experiments_cfg:
                ssq_result = _ssq_result.copy()

                log.info(f"🧪 Running experiment: {exp_cfg.name}")
                try:
                    ssq_result = execute_experiment(
                        exp_cfg, ssq_result, image.shape, ssq
                    )  # type: ignore

                    log.info(f"📈 Calculating metrics for experiment {exp_cfg.name}...")
                    metrics_results = calculate_metrics(image, ssq_result, metrics_cfg)

                    image_experiments_results[exp_cfg.name] = metrics_results
                    log.info(
                        f"🎉 Successfully completed experiment {exp_cfg.name} for image {i + 1}"
                    )
                except Exception as e:
                    log.error(
                        f"🔥 Experiment {exp_cfg.name} failed for image {i + 1}: {str(e)}"
                    )
                    raise
                finally:
                    del ssq_result
                    gc.collect()

            del _ssq_result
            gc.collect()

            experiment_results[img_hash] = image_experiments_results
            log.info(f"✅ Successfully completed all experiments for image {i + 1}")

        log.info(
            f"🏁 All experiments completed. Total results: {len(experiment_results)}"
        )
        return experiment_results

    except Exception as e:
        log.error(f"🔥 Error while running experiments: {e}")
        return experiment_results


def serialize_metrics(data: Dict[str, Dict[str, List[MetricResult]]]) -> str:
    """
    Serializes a nested dictionary of metric results into a JSON-formatted string.
    Args:
        data (Dict[str, Dict[str, List[MetricResult]]]):
            A dictionary where each key is an image hash, mapping to another dictionary.
            The inner dictionary maps experiment names to a list of MetricResult objects.
    Returns:
        str: A JSON-formatted string representing the serialized metrics,
             with NumPy float results converted to native Python floats for JSON compatibility.
    Notes:
        - Assumes each MetricResult object has 'metric_name' and 'result' attributes.
        - Handles NumPy float32 and float64 types by converting them to Python floats.
    """

    serializable_data = {}

    for image_hash, experiments in data.items():
        serializable_data[image_hash] = {}

        for exp_name, metrics in experiments.items():
            metrics_dict = {}
            for metric in metrics:
                if isinstance(metric.result, (np.float32, np.float64)):  # type: ignore
                    metrics_dict[metric.metric_name] = float(metric.result)
                else:
                    metrics_dict[metric.metric_name] = metric.result

            serializable_data[image_hash][exp_name] = metrics_dict

    return json.dumps(serializable_data, indent=4)


def deserialize_metrics(json_str: str) -> Dict[str, Dict[str, List[MetricResult]]]:
    """
    Deserializes a JSON string representing metric results into a nested dictionary structure.
    Args:
        json_str (str): A JSON-formatted string where each top-level key is an image hash,
            mapping to a dictionary of experiment names, which in turn map to dictionaries
            of metric names and their results.
    Returns:
        Dict[str, Dict[str, List[MetricResult]]]: A nested dictionary where each image hash maps to
            a dictionary of experiment names, each containing a list of MetricResult objects
            with metric names and their corresponding float results.
    Raises:
        json.JSONDecodeError: If the input string is not valid JSON.
        TypeError: If the structure of the JSON does not match the expected format.
    """

    data = json.loads(json_str)

    deserialized_data = {}

    for image_hash, experiments in data.items():
        deserialized_data[image_hash] = {}

        for exp_name, metrics_dict in experiments.items():
            # Convert dictionary back to list of MetricResult with Python floats
            metrics_list = [
                MetricResult(metric_name=metric_name, result=float(result))
                for metric_name, result in metrics_dict.items()
            ]

            deserialized_data[image_hash][exp_name] = metrics_list

    return deserialized_data
