from collections import namedtuple
import logging
from typing import Dict, List, Tuple

import numpy as np

from src.curvelet_sst import Synchrosqueezer

IMG_DTYPE = np.float32

log = logging.getLogger()

ExperimentConfig = namedtuple(
    "ExperimentConfig", ("name", "function", "params", "args", "kwargs")
)


class SynchrosqueezedResult:
    def __init__(
        self,
        ss_energy: np.ndarray,
        coeff_cell: np.ndarray,
        coeff_tensor: np.ndarray,
        vecx: np.ndarray,
        vecy: np.ndarray,
    ) -> None:
        self.ss_energy = ss_energy.astype(IMG_DTYPE)
        self.coeff_cell = coeff_cell
        self.coeff_tensor = coeff_tensor
        self.vecx = vecx
        self.vecy = vecy

    def copy(self: "SynchrosqueezedResult", fields: List[str] = ["ss_energy"]):
        """
        Creates a copy of the current SynchrosqueezedResult instance, optionally copying and converting specified fields.
        Args:
            fields (List[str], optional): List of attribute names to be deeply copied and converted to IMG_DTYPE.
                Defaults to ["ss_energy"].
        Returns:
            SynchrosqueezedResult: A new instance with copied attributes.
        Raises:
            AssertionError: If any specified field does not exist in the instance.
        """
        kwargs = {
            key: value for key, value in self.__dict__.items() if key not in fields
        }

        for field in fields:
            assert field in self.__dict__, f"The property {field} does not exists."

            kwargs[field] = self.__getattribute__(field).copy().astype(IMG_DTYPE)

        return SynchrosqueezedResult(**kwargs)

    def _replace(self: "SynchrosqueezedResult", **kwargs: Dict[str, np.ndarray]):
        """
        Replace attributes of the SynchrosqueezedResult instance with new values.

        Parameters:
            **kwargs (Dict[str, np.ndarray]):
                Keyword arguments where each key is the name of the attribute to replace,
                and each value is the new value (expected to be a numpy ndarray) for that attribute.

        Returns:
            SynchrosqueezedResult:
                The instance with updated attributes.
        """
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        return self


def execute_experiment(
    config: ExperimentConfig,
    ssq_result: SynchrosqueezedResult,
    img_shape: Tuple[int, int],
    ssq: Synchrosqueezer,
) -> np.ndarray:
    """
    Executes a configured experiment using synchrosqueezed transform results and returns the reconstructed image.
    Parameters:
        config (ExperimentConfig): Configuration object containing the experiment function, parameters, and metadata.
        ssq_result (SynchrosqueezedResult): The result object from a previous synchrosqueezed transform.
        img_shape (Tuple[int, int]): The shape of the output image (height, width).
        ssq (Synchrosqueezer): The synchrosqueezer instance used for inverse transformation.
    Returns:
        np.ndarray: The reconstructed image obtained after applying the inverse synchrosqueezed curvelet transform.
    Raises:
        Exception: If the experiment function or the inverse transform fails, the exception is logged and re-raised.
    """

    log.info(f"⚙️ Preparing experiment: {config.name}")

    log.info("🔧 Generating input parameters...")
    input_params = config.params(ssq_result)  # dynamically generated parameters
    input_args = input_params + config.args  # combine with static parameters

    log.info(
        f"🚀 Executing experiment function with {len(input_args)} args and {len(config.kwargs)} kwargs"
    )
    try:
        result: SynchrosqueezedResult = config.function(
            ssq_result, *input_args, **config.kwargs
        )
    except Exception as e:
        log.error(f"💥 Experiment function failed: {str(e)}")
        raise

    log.info("🔄 Performing inverse synchrosqueezed curvelet transform...")
    try:
        inv_img = ssq.inv_synchrosqueezed_curvelet_transform(
            result.ss_energy,
            result.coeff_cell,
            result.coeff_tensor,
            result.vecx,
            result.vecy,
            img_shape,
        )
        log.info("✅ Experiment completed successfully")
        return inv_img
    except Exception as e:
        log.error(f"💢 Inverse transform failed: {str(e)}")
        raise
