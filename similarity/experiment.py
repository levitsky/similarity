import logging

from .utils import Config
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame
from .grouping import SpectrumGrouping
from .scoring import ProcessedPairs

logger = logging.getLogger(__name__)


class Experiment:
    mz_irt_df = MzIrtDataFrame()
    groups_df = SpectrumGrouping()
    predicted_spectra = PredictedSpectrumCollection()
    processed_pairs = ProcessedPairs()

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        logger.debug("MzIrtDataFrame columns: %s", self.mz_irt_df.columns)
        logger.debug("Groups DataFrame shape: %s", self.groups_df.shape)
        logger.debug("Predicted spectra: %s", self.predicted_spectra)
        return self.processed_pairs
