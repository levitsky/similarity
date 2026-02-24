import logging
import diskcache
from .utils import Config
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame
from .grouping import SpectrumGrouping
from .scoring import ProcessedPairs
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diskcache import Cache

logger = logging.getLogger(__name__)


class Experiment:
    mz_irt_df = MzIrtDataFrame()
    pairs = SpectrumGrouping()
    predicted_spectra = PredictedSpectrumCollection()
    processed_pairs = ProcessedPairs()
    _cache: "Cache | None" = None

    def __init__(self, config: Config):
        self.config = config
        self._cache = diskcache.Cache(
            str(config.cache_dir),
            size_limit=0,
            cull_limit=0,
            eviction_policy="none",
        )

    def run(self):

        logger.info("Start calculating mz and predicting RT...")
        logger.debug("MzIrtDataFrame columns: %s", self.mz_irt_df.columns)
        logger.debug("MzIrtDataFrame shape: %s", self.mz_irt_df.shape)
        logger.info("Start making groups...")
        logger.debug("Found %d pairs", len(self.pairs))
        logger.info("Start predicting spectra...")
        logger.debug("Predicted spectra: %s", self.predicted_spectra)
        logger.info("Spectrum prediction complete.")
        logger.info("Start processing pairs...")
        logger.info("Processed %d pairs.", self.processed_pairs.shape[0])
        return self.processed_pairs
