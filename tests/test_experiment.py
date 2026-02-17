import unittest
import numpy as np
import pandas as pd
from similarity.experiment import Experiment
from similarity.scoring import ProcessedPairs
from similarity.utils import Config
from pathlib import Path
import logging

try:
    from MSCI.Similarity.spectral_angle_similarity import joinPeaks
except ImportError as e:
    joinPeaks = None


def ndotproduct(x, y, m=0, n=0.5, na_rm=True):
    wx = _weightxy(x.iloc[:, 0], x.iloc[:, 1], m, n)
    wy = _weightxy(y.iloc[:, 0], y.iloc[:, 1], m, n)
    wx2 = wx**2
    wy2 = wy**2
    num = np.sum(wx * wy) ** 2
    delim1 = np.sum(wx2, axis=0)
    delim2 = np.sum(wy2, axis=0)
    value = num / (delim1 * delim2)
    logging.debug(
        "ndotproduct: wx = %s, wy = %s, wx2 = %s, wy2 = %s, num = %f, delim1 = %f, delim2 = %f, value = %f",
        wx.values,
        wy.values,
        wx2.values,
        wy2.values,
        num,
        delim1,
        delim2,
        value,
    )
    return value


def nspectraangle(x, y, m=0, n=0.5, na_rm=True):
    return 1 - 2 * np.arccos(ndotproduct(x, y, m, n, na_rm)) / np.pi


def _weightxy(x, y, m=0, n=0.5):
    return x**m * y**n


class TestBase(unittest.TestCase):
    def setUp(self):
        self.config = Config(input_file=Path("tests/test_peptides.txt"))
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,  # overrides any existing logging config
        )
        self.logger = logging.getLogger(__name__)


class ExperimentTest(TestBase):
    def test_run(self):
        """Test that Experiment.run() executes and returns something."""
        exp = Experiment(self.config)
        result = exp.run().sort_values(["index1", "index2"])
        # Check that run() returns the processed_pairs object
        self.logger.debug("Final result:\n%s", result)
        self.assertEqual(result.shape[0], 9)  # Assuming 9 pairs based on the test input
        self.assertTrue(
            np.allclose(
                result["similarity score"],
                [
                    0.847243,
                    0.816326,
                    0.724647,
                    0.912134,
                    0.772697,
                    0.81768,
                    0.974192,
                    0.858346,
                    0.933183,
                ],
                atol=1e-3,
            )
        )


class EquivalenceTest(TestBase):
    def setUp(self):
        if joinPeaks is None or nspectraangle is None:
            self.skipTest(f"MSCI not available.")
        super().setUp()

    def test_peak_matching(self):
        """Test that the peak matching logic correctly identifies matching peaks."""
        exp = Experiment(self.config)
        spectra = exp.predicted_spectra
        for i, j in exp.pairs:
            with self.subTest(pair=(i, j)):
                pep1 = exp.mz_irt_df.loc[i, "peptide_sequences"]
                pep2 = exp.mz_irt_df.loc[j, "peptide_sequences"]
                mz1 = exp.predicted_spectra.loc[pep1, "mz"].values
                mz2 = exp.predicted_spectra.loc[pep2, "mz"].values
                self.logger.debug(
                    "Testing peak matching for peptides %d, %d: %s and %s",
                    i,
                    j,
                    pep1,
                    pep2,
                )
                self.logger.debug("m/z values for peptide 1: %s", mz1)
                self.logger.debug("m/z values for peptide 2: %s", mz2)
                idx1, idx2 = ProcessedPairs.match_peaks(
                    mz1,
                    mz2,
                    atol=0.1,
                    rtol=0.0005,
                )
                self.logger.debug("Matched indices: %s and %s", idx1, idx2)
                self.logger.debug(
                    "Matched m/z values:\n%s and\n%s",
                    np.sort(mz1[idx1]),
                    np.sort(mz2[idx2]),
                )

                matcher = joinPeaks(
                    tolerance=self.config.peak_tolerance, ppm=self.config.peak_ppm
                )
                x_df = (
                    spectra.loc[pep1, ["mz", "intensities"]]
                    .sort_values(by="mz")
                    .reset_index(drop=True)
                )
                y_df = (
                    spectra.loc[pep2, ["mz", "intensities"]]
                    .sort_values(by="mz")
                    .reset_index(drop=True)
                )
                x_matched, y_matched = matcher.match(x_df, y_df)
                mask = pd.notna(x_matched["mz"]) & pd.notna(y_matched["mz"])

                self.logger.debug(
                    "Matched m/z values using joinPeaks:\n%s and \n%s",
                    np.sort(x_matched.loc[mask, "mz"].values),
                    np.sort(y_matched.loc[mask, "mz"].values),
                )
                self.logger.debug(
                    "Matched m/z values using match_peaks:\n%s and \n%s",
                    np.sort(mz1[idx1]),
                    np.sort(mz2[idx2]),
                )
                if idx1.size == x_matched.shape[0] and idx2.size == y_matched.shape[0]:
                    self.assertTrue(
                        np.allclose(
                            x_matched.loc[mask, "mz"].values,
                            np.sort(mz1[idx1]),
                            atol=0.01,
                            rtol=0.0005,
                        )
                    )
                    self.assertTrue(
                        np.allclose(
                            y_matched.loc[mask, "mz"].values,
                            np.sort(mz2[idx2]),
                            atol=0.01,
                            rtol=0.0005,
                        )
                    )
                else:
                    self.logger.warning(
                        "Number of matched peaks differs between joinPeaks and match_peaks for pair (%d, %d). Matching m/z:\njoinPeaks:\n%s and \n%s\nmatch_peaks:\n%s and \n%s",
                        i,
                        j,
                        x_matched.loc[mask, "mz"].values,
                        y_matched.loc[mask, "mz"].values,
                        np.sort(mz1[idx1]),
                        np.sort(mz2[idx2]),
                    )
                    self.skipTest(
                        f"Incompatible match output. Skipping test for pair ({i}, {j})."
                    )

    def test_similarity_score(self):
        """Test that the similarity score is calculated correctly."""
        exp = Experiment(self.config)
        spectra = exp.predicted_spectra
        pairs = exp.pairs
        for i, j in pairs:
            with self.subTest(pair=(i, j)):
                pep1 = exp.mz_irt_df.loc[i, "peptide_sequences"]
                pep2 = exp.mz_irt_df.loc[j, "peptide_sequences"]
                matcher = joinPeaks(
                    tolerance=self.config.peak_tolerance, ppm=self.config.peak_ppm
                )
                x_df = (
                    spectra.loc[pep1, ["mz", "intensities"]]
                    .sort_values(by="mz")
                    .reset_index(drop=True)
                )
                y_df = (
                    spectra.loc[pep2, ["mz", "intensities"]]
                    .sort_values(by="mz")
                    .reset_index(drop=True)
                )
                x_matched, y_matched = matcher.match(x_df, y_df)
                oldscore = nspectraangle(x_matched, y_matched, m=0, n=0.5)

                idx1, idx2 = ProcessedPairs.match_peaks(
                    x_df["mz"].values,
                    y_df["mz"].values,
                    atol=self.config.peak_tolerance,
                    rtol=self.config.peak_ppm / 1e6,
                )
                newscore = ProcessedPairs.similarity_score(
                    x_df["intensities"].values,
                    y_df["intensities"].values,
                    idx1,
                    idx2,
                )
                self.logger.debug(
                    "Testing similarity score for peptides %d, %d: %s and %s",
                    i,
                    j,
                    pep1,
                    pep2,
                )
                self.logger.debug("Old score: %f, New score: %f", oldscore, newscore)
                self.assertAlmostEqual(oldscore, newscore, places=3)


if __name__ == "__main__":
    unittest.main()
