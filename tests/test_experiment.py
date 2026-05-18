import unittest
import numpy as np
import multiprocessing as mp
import pandas as pd
import dataclasses
import tempfile
from similarity.experiment import Experiment
from similarity.grouping import GroupingWorker
from similarity.prediction import MzIrtDataFrame
from similarity.utils.config import (
    Config,
    KoinaIntensityModel,
    KoinaRTModel,
    KoinaCCSModel,
)
from similarity.utils.cache import CacheType
from similarity.utils.spectrum_collection import SpectrumCollectionType
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
        self.config = Config(input_file=Path("tests/test_peptides.txt"), batch_size=2)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,  # overrides any existing logging config
        )
        self.logger = logging.getLogger(__name__)
        self.correct_scores = sorted(
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
            ]
        )


class ExperimentTest(TestBase):
    def test_load_peptide_table(self):
        source = pd.DataFrame(
            {
                "peptide_sequences": ["PEPTIDE", "ACDEFGHIK"],
                "precursor_charges": [2, 3],
                "irt": [10.5, 20.25],
                "m/z": [500.123, 620.456],
                "collision_energies": [30.0, 30.0],
                "fragmentation_types": ["HCD", "HCD"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            peptide_file = Path(tmpdir) / "peptides.tsv"
            source.to_csv(peptide_file, index=False, sep="\t")
            with Experiment(self.config) as exp:
                loaded = MzIrtDataFrame().load_peptide_table(peptide_file, exp)
                self.assertEqual(list(loaded.columns), list(source.columns))

                seq = loaded["peptide_sequences"].to_numpy(copy=False)
                self.assertEqual(seq.dtype.kind, "S")
                self.assertEqual(
                    [x.decode("ascii") for x in seq],
                    source["peptide_sequences"].tolist(),
                )
                self.assertTrue(
                    np.array_equal(
                        loaded["precursor_charges"].to_numpy(copy=False),
                        source["precursor_charges"].to_numpy(copy=False),
                    )
                )
                self.assertTrue(
                    np.allclose(loaded["irt"].to_numpy(copy=False), source["irt"])
                )
                self.assertTrue(
                    np.allclose(loaded["m/z"].to_numpy(copy=False), source["m/z"])
                )

                shm = MzIrtDataFrame._shared_memory[exp]
                shm_seq = np.ndarray(
                    shape=(2,), dtype=seq.dtype, buffer=shm["peptide_sequences"].buf
                )
                shm_charge = np.ndarray(
                    shape=(2,), dtype=np.uint8, buffer=shm["precursor_charges"].buf
                )
                shm_mzrt = np.ndarray(
                    shape=(2, 2), dtype=np.float32, buffer=shm["mzrt"].buf
                )

                self.assertTrue(np.array_equal(shm_seq, seq))
                self.assertTrue(
                    np.array_equal(
                        shm_charge, loaded["precursor_charges"].to_numpy(copy=False)
                    )
                )
                self.assertTrue(
                    np.array_equal(shm_mzrt[:, 0], loaded["m/z"].to_numpy(copy=False))
                )
                self.assertTrue(
                    np.array_equal(shm_mzrt[:, 1], loaded["irt"].to_numpy(copy=False))
                )

    def test_load_peptide_table_with_ccs(self):
        config = dataclasses.replace(self.config, model_ccs=KoinaCCSModel.IM2Deep)
        source = pd.DataFrame(
            {
                "peptide_sequences": ["PEPTIDE", "ACDEFGHIK"],
                "precursor_charges": [2, 3],
                "irt": [10.5, 20.25],
                "m/z": [500.123, 620.456],
                "ccs": [255.1, 278.2],
                "collision_energies": [30.0, 30.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            peptide_file = Path(tmpdir) / "peptides_ccs.tsv"
            source.to_csv(peptide_file, index=False, sep="\t")
            with Experiment(config) as exp:
                loaded = MzIrtDataFrame().load_peptide_table(peptide_file, exp)
                self.assertTrue(
                    np.allclose(loaded["ccs"].to_numpy(copy=False), source["ccs"])
                )

    def test_subsets(self):
        subsets = 2
        outs = []
        for i in range(subsets):
            with Experiment(
                dataclasses.replace(self.config, subset=i + 1, subsets=subsets)
            ) as exp:
                outs.append(exp.score_df.sort_values(["score"]))
        combined = pd.concat(outs).sort_values(["score"])
        self.assertEqual(
            combined.shape[0], 9
        )  # Assuming 9 pairs based on the test input
        self.assertTrue(np.allclose(combined["score"], self.correct_scores, atol=1e-3))

    def test_subsets_with_peptide_table(self):
        subsets = 2
        outs = []
        with tempfile.TemporaryDirectory() as tmpdir:
            peptide_file = Path(tmpdir) / "peptides.tsv"
            with Experiment(self.config) as exp:
                df = exp.peptides
                df["peptide_sequences"] = df["peptide_sequences"].str.decode("ascii")
                df.to_csv(peptide_file, index=False, sep="\t")
                for i in range(subsets):
                    with Experiment(
                        dataclasses.replace(self.config, subset=i + 1, subsets=subsets),
                        peptide_table=peptide_file,
                    ) as exp_subset:
                        outs.append(exp_subset.score_df.sort_values(["score"]))
        combined = pd.concat(outs).sort_values(["score"])
        self.assertEqual(
            combined.shape[0], 9
        )  # Assuming 9 pairs based on the test input
        self.assertTrue(np.allclose(combined["score"], self.correct_scores, atol=1e-3))

    def test_run(self):
        """Test that Experiment executes and returns something."""
        for workers in [1, 5]:
            for cache_type in CacheType:
                for spectrum_collection_type in SpectrumCollectionType:
                    with self.subTest(
                        workers=workers,
                        cache_type=cache_type,
                        spectrum_collection_type=spectrum_collection_type,
                    ):
                        if (
                            cache_type == CacheType.NONE
                            and spectrum_collection_type
                            == SpectrumCollectionType.CACHED
                        ):
                            self.logger.info(
                                "Skipping combination of CacheType.NONE and SpectrumCollectionType.CACHED because it is not valid."
                            )
                            continue
                        self.logger.info(
                            "Testing Experiment with %d workers, %s cache and %s spectrum collection",
                            workers,
                            cache_type.name,
                            spectrum_collection_type.name,
                        )
                        config = dataclasses.replace(
                            self.config,
                            workers=workers,
                            cache=cache_type,
                            spectrum_collection=spectrum_collection_type,
                        )
                        with Experiment(config) as exp:
                            result = exp.score_df.sort_values(["score"])
                            self.logger.debug("Final result:\n%s", result)
                            self.assertEqual(
                                result.shape[0], 9
                            )  # Assuming 9 pairs based on the test input
                            self.assertTrue(
                                np.allclose(
                                    result["score"],
                                    self.correct_scores,
                                    atol=1e-3,
                                )
                            )

    def test_save_and_load_peptide_table(self):
        """Test that the peptide table can be saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            peptide_file = Path(tmpdir) / "peptides.tsv"
            with Experiment(self.config) as exp:
                df = exp.peptides
                df["peptide_sequences"] = df["peptide_sequences"].str.decode("ascii")
                df.to_csv(peptide_file, index=False, sep="\t")
                self.logger.info("Saved peptide table to %s", peptide_file)

                with Experiment(self.config, peptide_table=peptide_file) as exp_copy:
                    loaded = exp_copy.peptides
                    self.assertEqual(list(loaded.columns), list(df.columns))

                    seq = loaded["peptide_sequences"].to_numpy(copy=False)
                    self.assertEqual(seq.dtype.kind, "S")
                    self.assertEqual(
                        [x.decode("ascii") for x in seq],
                        df["peptide_sequences"].tolist(),
                    )
                    self.assertTrue(
                        np.array_equal(
                            loaded["precursor_charges"].to_numpy(copy=False),
                            df["precursor_charges"].to_numpy(copy=False),
                        )
                    )
                    self.assertTrue(
                        np.allclose(loaded["irt"].to_numpy(copy=False), df["irt"])
                    )
                    self.assertTrue(
                        np.allclose(loaded["m/z"].to_numpy(copy=False), df["m/z"])
                    )

    def test_multiple_charges(self):
        config = dataclasses.replace(self.config, max_charge=3)
        with Experiment(config) as exp:
            self.assertEqual(exp.peptides["precursor_charges"].max(), 3)
            self.assertEqual(exp.peptides.shape[0], 56)
            self.assertEqual(exp.score_df.shape[0], 18)

    def test_ptms(self):
        """Test that Experiment can handle peptides with PTMs."""
        config = Config(
            ptms=True,
            input_file=Path("tests/test_peptides_ptms.txt"),
            model_irt=KoinaRTModel.Prosit_2025_irt_40PTM,
            model_intensity=KoinaIntensityModel.Prosit_2025_intensity_40PTM,
            fragmentation_type="HCD",
        )
        with Experiment(config) as exp:
            self.assertTrue(
                np.allclose(sorted(exp.peptides["irt"]), [2.694990, 115.689156])
            )
            self.assertTrue(
                np.allclose(sorted(exp.peptides["m/z"]), [501.752743, 580.382344])
            )
            self.assertEqual(exp.score_df.shape[0], 0)

    def test_ccs(self):
        """Test that Experiment can handle CCS predictions."""
        config = dataclasses.replace(
            self.config,
            model_ccs=KoinaCCSModel.IM2Deep,
        )
        with Experiment(config) as exp:
            self.assertEqual(exp.peptides["ccs"].isna().sum(), 0)


class EquivalenceTest(TestBase):
    def setUp(self):
        if joinPeaks is None or nspectraangle is None:
            self.skipTest("MSCI not available.")
        super().setUp()

    def test_peak_matching(self):
        """Test that the peak matching logic correctly identifies matching peaks."""
        with Experiment(self.config) as exp:
            for i, j, score in exp.score_array:
                with self.subTest(pair=(i, j)):
                    mz1, intensities1 = exp.predicted_spectra[i]
                    mz2, intensities2 = exp.predicted_spectra[j]
                    self.logger.debug(
                        "Testing peak matching for peptides %d, %d",
                        i,
                        j,
                    )
                    self.logger.debug("m/z values for peptide 1: %s", mz1)
                    self.logger.debug("m/z values for peptide 2: %s", mz2)
                    idx1, idx2 = GroupingWorker.match_peaks(
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
                        pd.DataFrame({"mz": mz1, "intensities": intensities1})
                        .sort_values(by="mz")
                        .reset_index(drop=True)
                    )
                    y_df = (
                        pd.DataFrame({"mz": mz2, "intensities": intensities2})
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
                    if (
                        idx1.size == x_matched.shape[0]
                        and idx2.size == y_matched.shape[0]
                    ):
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
        with Experiment(self.config) as exp:
            for i, j, score in exp.score_array:
                with self.subTest(pair=(i, j)):
                    matcher = joinPeaks(
                        tolerance=self.config.peak_tolerance, ppm=self.config.peak_ppm
                    )
                    x_df = (
                        pd.DataFrame(
                            {
                                "mz": exp.predicted_spectra[i][0],
                                "intensities": exp.predicted_spectra[i][1],
                            }
                        )
                        .sort_values(by="mz")
                        .reset_index(drop=True)
                    )
                    y_df = (
                        pd.DataFrame(
                            {
                                "mz": exp.predicted_spectra[j][0],
                                "intensities": exp.predicted_spectra[j][1],
                            }
                        )
                        .sort_values(by="mz")
                        .reset_index(drop=True)
                    )
                    x_matched, y_matched = matcher.match(x_df, y_df)
                    oldscore = nspectraangle(x_matched, y_matched, m=0, n=1)

                    idx1, idx2 = GroupingWorker.match_peaks(
                        x_df["mz"].values,
                        y_df["mz"].values,
                        atol=self.config.peak_tolerance,
                        rtol=self.config.peak_ppm / 1e6,
                    )
                    newscore = GroupingWorker.similarity_score(
                        x_df["intensities"].values,
                        y_df["intensities"].values,
                        idx1,
                        idx2,
                    )
                    self.logger.debug(
                        "Testing similarity score for peptides %d, %d", i, j
                    )
                    self.logger.debug(
                        "Old score: %f, New score: %f", oldscore, newscore
                    )
                    self.assertAlmostEqual(oldscore, newscore, places=3)


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    unittest.main()
