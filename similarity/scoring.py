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
        return np.unique(idx1), np.unique(idx2)

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

    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        index_array = experiment.pairs
        spectra = experiment.predicted_spectra
        mz_irt_df = experiment.mz_irt_df
        rows = []
        for i, j in index_array:
            pep1 = mz_irt_df.loc[i, "peptide_sequences"]
            pep2 = mz_irt_df.loc[j, "peptide_sequences"]
            mz1 = spectra.loc[spectra["peptide_sequences"] == pep1, "mz"].values
            mz2 = spectra.loc[spectra["peptide_sequences"] == pep2, "mz"].values
            idx1, idx2 = self.match_peaks(
                mz1,
                mz2,
                atol=experiment.config.peak_tolerance,
                rtol=experiment.config.peak_ppm / 1e6,
            )

            intensities1 = spectra.loc[
                spectra["peptide_sequences"] == pep1, "intensities"
            ].values
            intensities2 = spectra.loc[
                spectra["peptide_sequences"] == pep2, "intensities"
            ].values
            logger.debug(
                "For pair (%d, %d), the matching peaks: %s and %s with intensities %s and %s",
                i,
                j,
                mz1[idx1],
                mz2[idx2],
                intensities1,
                intensities2,
            )
            logger.debug("Full m/z arrays: %s and %s", mz1, mz2)
            score = self.similarity_score(intensities1, intensities2, idx1, idx2)
            rows.append(
                {
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
            )
        return pd.DataFrame(rows)
