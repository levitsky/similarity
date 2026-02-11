from typing import Any
from collections.abc import Iterable

from .utils import Fixture, Config
from .prediction import PredictedSpectrumCollection, MzIrtDataFrame
from .grouping import SpectrumGrouping
from .scoring import ProcessedPairs


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
        print(self.predicted_spectra)
        print(self.mz_irt_df.shape)
        print(self.groups_df.shape)
        print(self.processed_pairs)
