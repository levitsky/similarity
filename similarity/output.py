from typing import TYPE_CHECKING
import pandas as pd
import logging
from .utils.abc import Fixture

if TYPE_CHECKING:
    from .experiment import Experiment

logger = logging.getLogger(__name__)


class ScoresDataFrame(Fixture):
    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        peptides = experiment.peptides
        df = pd.DataFrame.from_records(experiment.score_array)
        df["peptide 1"] = (
            df["i"]
            .apply(lambda i: peptides.loc[i, "peptide_sequences"])
            .astype(bytes)
            .str.decode("ascii")
        )
        df["peptide 2"] = (
            df["j"]
            .apply(lambda j: peptides.loc[j, "peptide_sequences"])
            .astype(bytes)
            .str.decode("ascii")
        )
        df["charge 1"] = df["i"].apply(lambda i: peptides.loc[i, "precursor_charges"])
        df["charge 2"] = df["j"].apply(lambda j: peptides.loc[j, "precursor_charges"])
        df["m/z 1"] = df["i"].apply(lambda i: peptides.loc[i, "m/z"])
        df["m/z 2"] = df["j"].apply(lambda j: peptides.loc[j, "m/z"])
        df["iRT 1"] = df["i"].apply(lambda i: peptides.loc[i, "irt"])
        df["iRT 2"] = df["j"].apply(lambda j: peptides.loc[j, "irt"])
        return df
