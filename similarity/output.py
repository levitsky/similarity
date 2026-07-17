from typing import TYPE_CHECKING, cast
import pandas as pd
import logging
from .utils.abc import Fixture

if TYPE_CHECKING:
    from .experiment import Experiment, SingleInputExperiment, DualInputExperiment

logger = logging.getLogger(__name__)


class ScoresDataFrame(Fixture):
    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        if hasattr(experiment, "peptides_1"):
            e = cast("DualInputExperiment", experiment)
            peptides_1 = e.peptides_1
            peptides_2 = e.peptides_2
        else:
            e = cast("SingleInputExperiment", experiment)
            peptides_1 = peptides_2 = e.peptides
        df = pd.DataFrame.from_records(experiment.score_array)
        df["peptide 1"] = (
            df["i"]
            .apply(lambda i: peptides_1.loc[i, "peptide_sequences"])
            .astype(bytes)
            .str.decode("ascii")
        )
        df["peptide 2"] = (
            df["j"]
            .apply(lambda j: peptides_2.loc[j, "peptide_sequences"])
            .astype(bytes)
            .str.decode("ascii")
        )
        df["charge 1"] = df["i"].apply(lambda i: peptides_1.loc[i, "precursor_charges"])
        df["charge 2"] = df["j"].apply(lambda j: peptides_2.loc[j, "precursor_charges"])
        df["m/z 1"] = df["i"].apply(lambda i: peptides_1.loc[i, "m/z"])
        df["m/z 2"] = df["j"].apply(lambda j: peptides_2.loc[j, "m/z"])
        df["iRT 1"] = df["i"].apply(lambda i: peptides_1.loc[i, "irt"])
        df["iRT 2"] = df["j"].apply(lambda j: peptides_2.loc[j, "irt"])
        if "ccs" in peptides_1.columns and "ccs" in peptides_2.columns:
            df["CCS 1"] = df["i"].apply(lambda i: peptides_1.loc[i, "ccs"])
            df["CCS 2"] = df["j"].apply(lambda j: peptides_2.loc[j, "ccs"])
        return df
