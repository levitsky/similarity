import logging

from .utils import Config
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame
from .grouping import SpectrumGrouping
from .scoring import ProcessedPairs

logger = logging.getLogger(__name__)


class Experiment:
    mz_irt_df = MzIrtDataFrame()
    pairs = SpectrumGrouping()
    predicted_spectra = PredictedSpectrumCollection()
    processed_pairs = ProcessedPairs()

    def __init__(self, config: Config):
        self.config = config

    def run(self):

        logger.info("Start calculating mz and predicting RT...")
        logger.debug("MzIrtDataFrame columns: %s", self.mz_irt_df.columns)
        logger.debug("MzIrtDataFrame shape: %s", self.mz_irt_df.shape)
        logger.info("Start making groups...")
        logger.debug("Found %d pairs", len(self.pairs))
        logger.info("Start predicting spectra...")
        logger.debug("Predicted spectra: %s", self.predicted_spectra)
        logger.info("Predicted spectra size: %d", len(self.predicted_spectra))
        logger.info("Start processing pairs...")
        logger.info("Processed pairs shape: %s", self.processed_pairs.shape)
        return self.processed_pairs
