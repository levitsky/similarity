import logging

from .utils import Config
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame
from .grouping import SpectrumGrouping
from .scoring import ProcessedPairs

logger = logging.getLogger(__name__)


class Experiment:
    # peptides = PeptideCollection()
    predicted_spectra = PredictedSpectrumCollection()
    mz_irt_df = MzIrtDataFrame()
    groups_df = SpectrumGrouping()
    processed_pairs = ProcessedPairs()

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        # Placeholder for the main logic of the experiment
        # print(f"Predicted spectra: {self.predicted_spectra}")
        # Perform calculations and comparisons here
        logger.debug("Predicted spectra: %s", self.predicted_spectra)
        logger.debug("MzIrtDataFrame shape: %s", self.mz_irt_df.shape)
        logger.debug("Groups DataFrame shape: %s", self.groups_df.shape)
        return self.processed_pairs
