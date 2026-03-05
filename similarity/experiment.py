import logging
import diskcache
from .utils import Config
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame
from .grouping import SpectrumGrouping
from .output import ScoresDataFrame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diskcache import Cache
    import pandas as pd
    import numpy as np
    from .utils import SpectrumIndex

logger = logging.getLogger(__name__)


class Experiment:
    peptides: "pd.DataFrame" = MzIrtDataFrame()
    predicted_spectra: "SpectrumIndex" = PredictedSpectrumCollection()
    score_array: "np.ndarray" = SpectrumGrouping()
    score_df: "pd.DataFrame" = ScoresDataFrame()

    def __init__(self, config: Config):
        self.config = config
