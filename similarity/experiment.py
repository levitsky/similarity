import logging
import traceback
from .utils.config import Config
from .utils.abc import Cache, IndexType
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame
from .grouping import SpectrumGrouping
from .output import ScoresDataFrame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    from .utils.abc import SpectrumCollection, Cache


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

    def __init__(self, config: Config):
        self.config = config
        self.cache: dict[IndexType, "Cache | None"] = {
            index_type: config.cache.value.get_index(index_type, self)
            for index_type in IndexType
        }
        logger.debug(
            "Initialized experiment %d with cache configuration: %s",
            id(self),
            self.cache,
        )

    def __reduce__(self) -> tuple:
        return self.__class__, (self.config,)

    def _cleanup(self):
        MzIrtDataFrame.close(self)
        if PredictedSpectrumCollection.exists(self):
            self.predicted_spectra.close()
        while self.cache:
            _, index = self.cache.popitem()
            if index is not None:
                logger.debug("Closing cache %s for experiment %d", index, id(self))
                index.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        logger.debug(
            "Cleaning up experiment %d. Reason: %s (%s). Traceback: %s",
            id(self),
            exc_type.__name__ if exc_type else "Normal exit",
            exc_value if exc_value else "No exception",
            traceback.format_tb(tb) if tb else "No traceback",
        )
        self._cleanup()
