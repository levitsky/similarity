import logging
import diskcache
from .utils import Config
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame
from .grouping import SpectrumGrouping
from .scoring import Scores, ScoresDataFrame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diskcache import Cache
    import pandas as pd
    import numpy as np
    from .utils import SpectrumIndex

logger = logging.getLogger(__name__)


class Experiment:
    peptides: "pd.DataFrame" = MzIrtDataFrame()
    pairs: list = SpectrumGrouping()
    predicted_spectra: "SpectrumIndex" = PredictedSpectrumCollection()
    score_array: "np.ndarray" = Scores()
    score_df: "pd.DataFrame" = ScoresDataFrame()
    _cache: "Cache | None" = None

    def __init__(self, config: Config):
        self.config = config
        self._cache = diskcache.Cache(
            str(config.cache_dir),
            size_limit=0,
            cull_limit=0,
            eviction_policy="none",
        )
