import logging
from .utils.config import Config
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame
from .grouping import SpectrumGrouping
from .output import ScoresDataFrame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory
    import pandas as pd
    import numpy as np
    from .utils.abc import SpectrumCollection

logger = logging.getLogger(__name__)


class Experiment:
    if TYPE_CHECKING:
        peptides: "pd.DataFrame"
        predicted_spectra: "SpectrumCollection"
        score_array: "np.ndarray"
        score_df: "pd.DataFrame"
    else:
        peptides = MzIrtDataFrame()
        predicted_spectra = PredictedSpectrumCollection()
        score_array = SpectrumGrouping()
        score_df = ScoresDataFrame()
    _shared_memory: dict[str, "SharedMemory"]

    def __init__(self, config: Config):
        self.config = config
        self._shared_memory = {}

    def __cleanup(self):
        for key in list(self._shared_memory):
            shm = self._shared_memory.pop(key)
            shm.close()
            shm.unlink()
        self.predicted_spectra.close()

    def __del__(self):
        self.__cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__cleanup()
