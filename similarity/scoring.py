from typing import Any, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import logging
from .utils import Fixture

if TYPE_CHECKING:
    from .experiment import Experiment

logger = logging.getLogger(__name__)


class ProcessedPairs(Fixture):
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
            mask = np.isclose(
                mz1[:, None],
                mz2[None, :],
                rtol=experiment.config.peak_ppm / 1e6,
                atol=experiment.config.peak_tolerance,
            )
            idx1, idx2 = np.where(mask)

            intensities1 = spectra.loc[
                spectra["peptide_sequences"] == pep1, "intensities"
            ].values[idx1]
            intensities2 = spectra.loc[
                spectra["peptide_sequences"] == pep2, "intensities"
            ].values[idx2]
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
            wx = np.sqrt(intensities1)
            wy = np.sqrt(intensities2)
            ndotproduct = (
                (np.sum(wx * wy) ** 2) / np.sum(intensities1) / np.sum(intensities2)
            )
            score = 1 - 2 * np.arccos(ndotproduct) / np.pi
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
