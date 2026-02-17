from typing import Any, TYPE_CHECKING
import pandas as pd
import diskcache
from koinapy import Koina
from .utils import Fixture
from pyteomics import cmass
import logging

if TYPE_CHECKING:
    from .experiment import Experiment

logger = logging.getLogger(__name__)


class Index(diskcache.Index):
    """Index for predicted spectra. Uses experiment config to add collision energy, charge and model info to the key."""

    def _full_key(self, key: str) -> tuple:
        config = self.experiment.config
        # key is peptide sequence, we need to add collision energy, charge and model info to the key
        return (key, config.collision_energy, config.charge, config.model_intensity)

    def __init__(self, experiment: "Experiment"):
        self.experiment = experiment
        super().__init__(str(experiment.config.cache_dir))

    def __getitem__(self, key: str) -> Any:
        full_key = self._full_key(key)
        return super().__getitem__(full_key)

    def __setitem__(self, key: str, value: Any) -> None:
        full_key = self._full_key(key)
        super().__setitem__(full_key, value)

    def __contains__(self, key: str) -> bool:
        full_key = self._full_key(key)
        return full_key in self.cache  # direct check on Index doesn't work


class PredictedSpectrumCollection(Fixture):
    def evaluate(self, experiment: "Experiment") -> Index:
        df = experiment.mz_irt_df
        index = Index(experiment=experiment)
        logger.info("Found cache with %d entries", len(index))
        df["cached"] = df["peptide_sequences"].apply(lambda seq: seq in index)
        logger.info("%d of %d spectra are cached", df["cached"].sum(), len(df))
        if df["cached"].all():
            logger.info("All spectra are cached, skipping prediction")
            return index
        model = Koina(experiment.config.model_intensity, experiment.config.koina_host)
        result = model.predict(df.loc[~df["cached"]])
        result.set_index("peptide_sequences", inplace=True)

        for _, pep in df.loc[~df["cached"], "peptide_sequences"].items():
            index[pep] = result.loc[pep, ["mz", "intensities"]].values.T
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
