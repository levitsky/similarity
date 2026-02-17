from typing import Any, TYPE_CHECKING
import pandas as pd
import diskcache
from koinapy import Koina
from .utils import Fixture
from pyteomics import cmass

if TYPE_CHECKING:
    from .experiment import Experiment


class Index(diskcache.Index):
    """Index for predicted spectra. Uses experiment config to add collision energy, charge and model info to the key."""

    def __init__(self, *args, **kwargs):
        self.experiment = kwargs.pop("experiment")
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: str) -> Any:
        # key is peptide sequence, we need to add collision energy, charge and model info to the key
        config = self.experiment.config
        full_key = (key, config.collision_energy, config.charge, config.model_intensity)
        return super().__getitem__(full_key)

    def __setitem__(self, key: str, value: Any) -> None:
        config = self.experiment.config
        full_key = (key, config.collision_energy, config.charge, config.model_intensity)
        super().__setitem__(full_key, value)


class PredictedSpectrumCollection(Fixture):
    def evaluate(self, experiment: "Experiment") -> Index:
        df = experiment.mz_irt_df
        model = Koina(experiment.config.model_intensity, experiment.config.koina_host)
        result = model.predict(df)
        result.set_index("peptide_sequences", inplace=True)
        index = Index(experiment=experiment)
        for key in df.peptide_sequences:
            index[key] = result.loc[key, ["mz", "intensities"]].values.T
        return index


class MzIrtDataFrame(Fixture):
    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        input_file = experiment.config.input_file
        inputs = (
            pd.read_table(input_file, names=["peptide_sequences"], header=None)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        model = Koina(experiment.config.model_irt, experiment.config.koina_host)
        df = model.predict(inputs)
        df["m/z"] = df["peptide_sequences"].apply(
            lambda seq: cmass.fast_mass(seq, charge=experiment.config.charge)
        )
        df["precursor_charges"] = experiment.config.charge
        df["collision_energies"] = experiment.config.collision_energy
        return df
