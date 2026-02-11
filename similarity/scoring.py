from typing import Any, TYPE_CHECKING
from .utils import Fixture
from .legacy import parallel_process_spectra_pairs


if TYPE_CHECKING:
    from .experiment import Experiment

class ProcessedPairs(Fixture):
    def evaluate(self, experiment: "Experiment") -> Any:
        index_array = experiment.groups_df[["index1", "index2"]].values.astype(int)
        return parallel_process_spectra_pairs(index_array, n_chunks=experiment.config.n_chunks,
                                              spectra=experiment.predicted_spectra, mz_irt_df=experiment.mz_irt_df,
                                              tolerance=experiment.config.peak_tolerance, ppm=experiment.config.peak_ppm)