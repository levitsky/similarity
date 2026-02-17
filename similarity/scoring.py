from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import logging
from .utils import Fixture

if TYPE_CHECKING:
    from .experiment import Experiment

logger = logging.getLogger(__name__)


class ProcessedPairs(Fixture):
    @staticmethod
    def match_peaks(mz1: np.ndarray, mz2: np.ndarray, atol: float, rtol: float):
        mask = np.isclose(
            mz1[:, None],
            mz2[None, :],
            rtol=rtol,
            atol=atol,
        )
        idx1, idx2 = np.where(mask)
        return idx1, idx2

    @staticmethod
    def similarity_score(
        intensities1: np.ndarray,
        intensities2: np.ndarray,
        idx1: np.ndarray,
        idx2: np.ndarray,
    ) -> float:
        logger.debug(
            "Calculating similarity score with intensities1: %s, intensities2: %s, idx1: %s, idx2: %s",
            intensities1,
            intensities2,
            idx1,
            idx2,
        )
        wx = np.sqrt(intensities1[idx1])
        wy = np.sqrt(intensities2[idx2])
        # the numerator only has matching peaks intensities,
        # but the denominator has the sum of all intensities
        num = np.sum(wx * wy) ** 2
        denom1 = np.sum(intensities1)
        denom2 = np.sum(intensities2)

        ndotproduct = num / denom1 / denom2
        score = 1 - 2 * np.arccos(ndotproduct) / np.pi
        logger.debug(
            "Calculated similarity score: %f (num: %f, denom1: %f, denom2: %f, ndotproduct: %f)",
            score,
            num,
            denom1,
            denom2,
            ndotproduct,
        )
        return score

    def score_pair(self, i: int, j: int, experiment: "Experiment") -> float:
        mz_irt_df = experiment.mz_irt_df
        spectra = experiment.predicted_spectra
        pep1 = mz_irt_df.loc[i, "peptide_sequences"]
        pep2 = mz_irt_df.loc[j, "peptide_sequences"]
        mz1, intensities1 = spectra[pep1]
        mz2, intensities2 = spectra[pep2]
        idx1, idx2 = self.match_peaks(
            mz1,
            mz2,
            atol=experiment.config.peak_tolerance,
            rtol=experiment.config.peak_ppm / 1e6,
        )

        logger.debug(
            "For pair (%d, %d), the matching peaks: %s and %s with intensities %s and %s",
            i,
            j,
            mz1[idx1],
            mz2[idx2],
            intensities1[idx1],
            intensities2[idx2],
        )
        logger.debug("Full m/z arrays:\n%s:\n %s and\n%s:\n%s", pep1, mz1, pep2, mz2)
        return self.similarity_score(intensities1, intensities2, idx1, idx2)

    def format_result(
        self, i: int, j: int, score: float, experiment: "Experiment"
    ) -> dict:
        mz_irt_df = experiment.mz_irt_df
        pep1 = mz_irt_df.loc[i, "peptide_sequences"]
        pep2 = mz_irt_df.loc[j, "peptide_sequences"]
        return {
            "index1": i,
            "index2": j,
            "peptide 1": pep1,
            "peptide 2": pep2,
            "m/z 1": mz_irt_df.loc[i, "m/z"],
            "m/z 2": mz_irt_df.loc[j, "m/z"],
            "iRT 1": mz_irt_df.loc[i, "irt"],
            "iRT 2": mz_irt_df.loc[j, "irt"],
            "similarity score": score,
        }

    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        index_array = experiment.pairs
        rows = []
        for i, j in index_array:
            score = self.score_pair(i, j, experiment)
            rows.append(self.format_result(i, j, score, experiment))
        return pd.DataFrame(rows)
