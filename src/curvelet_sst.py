import os
import pickle
import logging
from typing import List, Tuple

import oct2py
import numpy as np

from src.image_utils import array_hash

IMG_DTYPE = np.float32

log = logging.getLogger()


class Synchrosqueezer:
    _instance = None
    _initialized = False

    def __new__(cls, octave_src_path="./src/syn_lab"):
        if cls._instance is None:
            cls._instance = super(Synchrosqueezer, cls).__new__(cls)
        return cls._instance

    def __init__(self, octave_src_path="./src/syn_lab"):
        if not self._initialized:
            self.data_path = octave_src_path

            self.oc = oct2py.Oct2Py()
            self.oc.addpath(octave_src_path)

            self._initialized = True

    def __del__(self):
        if hasattr(self, "oc"):
            self.oc.exit()

    def _reset(self):
        self.oc.exit()
        self.oc = oct2py.Oct2Py()
        self.oc.addpath(self.data_path)

    def synchrosqueezed_curvelet_transform(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, List, np.ndarray, np.ndarray, np.ndarray]:
        if not os.path.exists("ssq_cache"):
            os.makedirs("ssq_cache")

        hash = array_hash(img)

        if os.path.isfile(f"ssq_cache/ssct2_{hash}.pkl"):
            log.info(f"Loading cached data for hash {hash}")
            with open(f"ssq_cache/ssct2_{hash}.pkl", "rb") as f:
                ss_energy, coeff_cell, coeff_tensor, vecx, vecy = pickle.load(f)
        else:
            log.info(f"Computing SSQ for hash {hash}")
            ss_energy, coeff_cell, coeff_tensor, vecx, vecy = self.oc.ss_ct2_fwd(
                img, nout=5
            )
            with open(f"ssq_cache/ssct2_{hash}.pkl", "wb") as f:
                pickle.dump((ss_energy, coeff_cell, coeff_tensor, vecx, vecy), f)

        self._reset()

        return ss_energy, coeff_cell, coeff_tensor, vecx, vecy

    def inv_synchrosqueezed_curvelet_transform(
        self, ss_energy, coeff_cell, coeff_tensor, vecx, vecy, shape
    ) -> np.ndarray:
        img_inv = self.oc.ss_ct2_inv(
            ss_energy, coeff_cell, coeff_tensor, vecx, vecy, list(shape)
        )

        self._reset()

        return np.abs(img_inv).astype(IMG_DTYPE)
