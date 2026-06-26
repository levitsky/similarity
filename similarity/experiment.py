import logging
import traceback
from .utils.config import Config
from .utils.abc import Cache, IndexType
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame, Offsets
from .grouping import SpectrumGrouping
from .output import ScoresDataFrame
from typing import TYPE_CHECKING, Sequence, cast

if TYPE_CHECKING:
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from .utils.abc import SpectrumCollection, Cache


logger = logging.getLogger(__name__)


class Experiment:
    offsets = cast("Sequence[tuple[int, int]]", Offsets())
    score_array = cast("np.ndarray", SpectrumGrouping())
    score_df = cast("pd.DataFrame", ScoresDataFrame())
    config: Config

    def __init__(
        self,
        config: Config,
    ):
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
            "\n".join(traceback.format_tb(tb)) if tb else "No traceback",
        )
        self._cleanup()


class SingleInputExperiment(Experiment):
    peptides = MzIrtDataFrame()
    predicted_spectra = PredictedSpectrumCollection()

    def __init__(
        self,
        config: Config,
        peptide_table: "Path | str | None" = None,
    ):
        super().__init__(config)
        self.peptide_table = peptide_table

    def __reduce__(self) -> tuple:
        return self.__class__, (self.config, self.peptide_table)

    def _cleanup(self):
        super()._cleanup()
        self.__class__.peptides.close(self)
        if self.__class__.predicted_spectra.exists(self):
            self.predicted_spectra.close()
