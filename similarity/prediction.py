from typing import Any, TYPE_CHECKING
import pandas as pd
from koinapy import Koina
from .utils import Fixture
from pyteomics import cmass

if TYPE_CHECKING:
    from .experiment import Experiment


class PredictedSpectrumCollection(Fixture):
    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        df = experiment.mz_irt_df
        model = Koina(experiment.config.model_intensity, experiment.config.koina_host)
        result = model.predict(df)
        result = result.set_index("peptide_sequences")
        return result


class MzIrtDataFrame(Fixture):
    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        input_file = experiment.config.input_file
        inputs = pd.read_table(input_file, names=["peptide_sequences"], header=None)
        model = Koina(experiment.config.model_irt, experiment.config.koina_host)
        df = model.predict(inputs)
        df["m/z"] = df["peptide_sequences"].apply(
            lambda seq: cmass.fast_mass(seq, charge=experiment.config.charge)
        )
        df["precursor_charges"] = experiment.config.charge
        df["collision_energies"] = experiment.config.collision_energy
        return df
