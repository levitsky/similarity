import logging

from .utils import Config
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame
from .grouping import SpectrumGrouping
from .scoring import ProcessedPairs

logger = logging.getLogger(__name__)


class Experiment:
    predicted_spectra = PredictedSpectrumCollection()
    mz_irt_df = MzIrtDataFrame()
    groups_df = SpectrumGrouping()
    processed_pairs = ProcessedPairs()

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        logger.info("Start predicting spectra...")
        logger.debug("Predicted spectra: %s", self.predicted_spectra)
        logger.info("Predicted spectra size: %d", len(self.predicted_spectra))
        logger.info("Start calculating mz and predicting RT...")
        logger.debug("MzIrtDataFrame shape: %s", self.mz_irt_df.shape)
        logger.info("MzIrtDataFrame shape: %s", self.mz_irt_df.shape)
        logger.info("Start making groups...")
        logger.debug("Groups DataFrame shape: %s", self.groups_df.shape)
        logger.info("Groups DataFrame shape: %s", self.groups_df.shape)
        logger.info("Start processing pairs...")
        logger.info("Processed pairs shape: %s", self.processed_pairs.shape)
        return self.processed_pairs
