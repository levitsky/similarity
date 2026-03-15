from typing import TYPE_CHECKING
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from .utils.abc import Fixture

if TYPE_CHECKING:
    from .experiment import Experiment

logger = logging.getLogger(__name__)


class ScoreArray(Fixture):
    dtype = np.dtype([("i", np.int32), ("j", np.int32), ("score", np.float32)])

    def evaluate(self, experiment: "Experiment") -> np.ndarray:
        matches, scores = experiment.groups
        n = sum(len(m) for m in matches)
        logger.debug("Matches: %s", matches[:10])
        logger.debug("Scores: %s", scores[:10])
        score_array = np.empty(n, dtype=self.dtype)
        idx = 0
        for i, (m, s) in tqdm(
            enumerate(zip(matches, scores)),
            total=len(matches),
            desc="Filling score array",
            unit="peptides",
            unit_scale=True,
        ):
            size = len(m)
            if size:
                score_array[idx : idx + size] = [
                    (i, j, score) for j, score in zip(m, s)
                ]
                idx += size
        assert idx == n, f"Expected {n} scores, but got {idx}"
        return score_array


class ScoresDataFrame(Fixture):
    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        peptides = experiment.peptides
        df = pd.DataFrame.from_records(experiment.score_array)
        df["peptide 1"] = df["i"].apply(lambda i: peptides.loc[i, "peptide_sequences"])
        df["peptide 2"] = df["j"].apply(lambda j: peptides.loc[j, "peptide_sequences"])
        df["charge 1"] = df["i"].apply(lambda i: peptides.loc[i, "precursor_charges"])
        df["charge 2"] = df["j"].apply(lambda j: peptides.loc[j, "precursor_charges"])
        df["m/z 1"] = df["i"].apply(lambda i: peptides.loc[i, "m/z"])
        df["m/z 2"] = df["j"].apply(lambda j: peptides.loc[j, "m/z"])
        df["iRT 1"] = df["i"].apply(lambda i: peptides.loc[i, "irt"])
        df["iRT 2"] = df["j"].apply(lambda j: peptides.loc[j, "irt"])
        return df
