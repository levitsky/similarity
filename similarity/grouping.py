from typing import Any, TYPE_CHECKING

from .utils import Fixture

if TYPE_CHECKING:
    from .experiment import Experiment

from .legacy import process_peptide_combinations


class SpectrumGrouping(Fixture):
    def evaluate(self, experiment: "Experiment") -> Any:
        # Placeholder for actual grouping logic
        # Use predicted spectra and mz_irt_df from the experiment to perform grouping
        return process_peptide_combinations(experiment.mz_irt_df,
                                            experiment.config.mz_tolerance,
                                            experiment.config.irt_tolerance,
                                            False)