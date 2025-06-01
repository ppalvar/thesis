import sys

from src.image_utils import load_nii, get_nii_as_list
from src.experiments.exp_config import experiments
from src.metrics import (
    deserialize_metrics,
    run_experiments_and_metrics,
    serialize_metrics,
)
from src.metrics.metrics_config import metrics

import logging

"""
experiments.py
This script loads a NIfTI (.nii) medical image file, processes it into a list of images, and executes a series of experiments and metrics on each image sequentially. The results are serialized and saved to a JSON file. The experiments are run one by one (rather than in batch) due to stability issues encountered with batch processing.
Modules imported:
- sys, logging: For system operations and logging.
- src.image_utils: For loading and processing NIfTI images.
- src.experiments.exp_config: For experiment configurations.
- src.metrics: For running experiments, serializing, and deserializing metrics.
- src.metrics.metrics_config: For metrics configurations.
Execution flow:
1. Loads a specified .nii file and converts it to a list of images.
2. Iterates through each image, running experiments and metrics.
3. Merges new results with any existing results in the output JSON file.
4. Saves the updated results back to the JSON file.
5. Stops after processing the first successful image (due to the 'break' statement).
Note:
- Experiments are executed one by one due to stability issues with batch processing.
"""


log = logging.getLogger()
log.setLevel(logging.INFO)

if not log.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

file = "datasets/computed-tomography/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1/ct_scans/050.nii"

data = load_nii(file)
data = get_nii_as_list(data)

# result = run_experiments_and_metrics(data, experiments, metrics)
# json_str = serialize_metrics(result)
# with open("exp_out/out.json", "w") as f:
#     f.write(json_str)

log.info("🤖 Starting 🤖")

for index, image in enumerate(data):
    r = run_experiments_and_metrics([image], experiments, metrics)

    if r:
        with open("exp_out/out.json", "r") as f:
            s = deserialize_metrics(f.read())
            r = {**r, **s}
            json_str = serialize_metrics(r)

        with open("exp_out/out.json", "w") as f:
            f.write(json_str)

        break

    log.info(f"✅ Success! Metrics saved for image {index + 1}. 🎉")
