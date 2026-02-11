from typing import Any, TYPE_CHECKING

from .utils import Fixture

if TYPE_CHECKING:
    from .experiment import Experiment

from .legacy import predict_spectra, get_mz_irt_df


class PredictedSpectrumCollection(Fixture):
    def evaluate(self, experiment: "Experiment") -> list:
        return predict_spectra(
            input_file=str(experiment.config.input_file),
            collision_energy=experiment.config.collision_energy,
            charge=experiment.config.charge,
            model_intensity=experiment.config.model_intensity,
            model_irt=experiment.config.model_irt,
        )


class MzIrtDataFrame(Fixture):
    def evaluate(self, experiment: "Experiment") -> Any:
        input_file = experiment.config.input_file
        pred_file = f"{input_file.with_suffix('')}.msp"
        return get_mz_irt_df(pred_file)