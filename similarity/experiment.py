import logging
from .utils import Config
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame
from .grouping import SpectrumGrouping
from .output import ScoresDataFrame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory
    import pandas as pd
    import numpy as np
    from .utils import SpectrumIndex

logger = logging.getLogger(__name__)


class Experiment:
    peptides: "pd.DataFrame" = MzIrtDataFrame()
    predicted_spectra: "SpectrumIndex" = PredictedSpectrumCollection()
    score_array: "np.ndarray" = SpectrumGrouping()
    score_df: "pd.DataFrame" = ScoresDataFrame()
    _shared_memory: dict[str, "SharedMemory"]

    def __init__(self, config: Config):
        self.config = config
        self._shared_memory = {}

    def __cleanup(self):
        for key in list(self._shared_memory):
            shm = self._shared_memory.pop(key)
            shm.close()
            shm.unlink()

    def __del__(self):
        self.__cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__cleanup()
