from typing import Any, TYPE_CHECKING
import pandas as pd
from koinapy import Koina
from .utils import Fixture
from pyteomics import cmass

if TYPE_CHECKING:
    from .experiment import Experiment

from .legacy import predict_spectra, get_mz_irt_df


class PredictedSpectrumCollection(Fixture):
    def evaluate(self, experiment: "Experiment") -> list:
        df = experiment.mz_irt_df
        model = Koina(experiment.config.model_intensity, experiment.config.koina_host)
        result = model.predict(df)
        return result
        # return predict_spectra(
        #     input_file=str(experiment.config.input_file),
        #     collision_energy=experiment.config.collision_energy,
        #     charge=experiment.config.charge,
        #     model_intensity=experiment.config.model_intensity,
        #     model_irt=experiment.config.model_irt,
        # )


class MzIrtDataFrame(Fixture):
    def evaluate(self, experiment: "Experiment") -> Any:
        input_file = experiment.config.input_file
        inputs = pd.read_table(input_file, names=["peptide_sequences"], header=None)
        model = Koina(experiment.config.model_irt, experiment.config.koina_host)
        df = model.predict(inputs)
        df.columns = ["peptide_sequences", "iRT"]  # rename iRT for legacy compatibility
        df["MW"] = df["peptide_sequences"].apply(
            lambda seq: cmass.fast_mass(seq, charge=experiment.config.charge)
        )
        df["precursor_charges"] = experiment.config.charge
        df["collision_energies"] = experiment.config.collision_energy
        df["Name"] = df["peptide_sequences"].str.cat(
            df["precursor_charges"].astype(str), sep="/"
        )  # backwards compatibility with legacy code
        return df
