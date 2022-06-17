import math
from typing import Optional

import numpy as np


class DecisionMatrix:
    _raw: np.ndarray
    _normalized: Optional[np.ndarray]
    _weighted: Optional[np.ndarray]

    def __init__(self, raw_assessments: np.ndarray):
        self._raw = raw_assessments
        self._normalized = None
        self._weighted = None

    def get_normalized(self):
        return self._normalized

    def get_raw(self):
        return self._raw

    def get_weighted(self) -> np.ndarray:
        return self._weighted

    def normalize(self):
        if self._normalized is not None:
            return
        dividers = np.apply_along_axis(lambda column: math.sqrt((column**2).sum()), 0, self._raw)
        self._normalized = np.around(self._raw / dividers, decimals=4)

    def apply_criteria_weights(self, criteria_weights):
        to_multiply = self._raw if self._normalized is None else self._normalized
        self._weighted = np.around(to_multiply * np.array(criteria_weights), decimals=4)
