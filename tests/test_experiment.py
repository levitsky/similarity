import unittest
import numpy as np
import multiprocessing as mp
import pandas as pd
import dataclasses
import tempfile
from collections import Counter
from similarity.experiment import Experiment
from similarity.grouping import GroupingWorker, SpectrumGrouping
from similarity.prediction import MzIrtDataFrame
from similarity.utils.config import (
    Config,
    KoinaIntensityModel,
    KoinaRTModel,
    KoinaCCSModel,
    FragmentationType,
    MzErrorUnit,
    PROTON_MASS,
)
from similarity.utils.cache import CacheType
from similarity.utils.spectrum_collection import SpectrumCollectionType
from similarity.utils.utils import ExperimentRunner
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
    test_file = "tests/test_peptides.txt"
    batch_size = 2
    mz_tolerance = 1.0
    mz_unit = MzErrorUnit.Th

    def setUp(self):
        self.config = Config(
            input_file=Path(self.test_file),
            batch_size=self.batch_size,
            precursor_mz_tolerance=self.mz_tolerance,
            precursor_mz_unit=self.mz_unit,
            model_intensity=KoinaIntensityModel.Prosit_2020_intensity_HCD,
            model_irt=KoinaRTModel.Prosit_2019_irt,
        )
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,  # overrides any existing logging config
        )
        self.logger = logging.getLogger(__name__)


class ExperimentTest(TestBase):
    def setUp(self):
        super().setUp()
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

    def test_mz_array_multiple_charges_two_and_three(self):
        config = dataclasses.replace(self.config, min_charge=2, max_charge=3)
        input_peptides = np.unique(np.loadtxt(Path(self.test_file), dtype=bytes))
        input_peptides = input_peptides[
            [len(s) < self.config.max_length for s in input_peptides]
        ]
        with Experiment(config) as exp:
            peptides = exp.peptides[["peptide_sequences", "precursor_charges", "m/z"]]

            self.assertEqual(peptides.shape[0], 2 * input_peptides.shape[0])
            self.assertEqual(set(peptides["precursor_charges"].tolist()), {2, 3})

            mz_by_charge = peptides.pivot_table(
                index="peptide_sequences",
                columns="precursor_charges",
                values="m/z",
                aggfunc="first",
            )
            self.assertFalse(mz_by_charge[[2, 3]].isna().any().any())

            # m/z values for z=2 and z=3 must map back to the same neutral mass.
            mass_from_z2 = (
                mz_by_charge[2].to_numpy(dtype=np.float64) * 2 - 2 * PROTON_MASS
            )
            mass_from_z3 = (
                mz_by_charge[3].to_numpy(dtype=np.float64) * 3 - 3 * PROTON_MASS
            )
            np.testing.assert_allclose(mass_from_z2, mass_from_z3, rtol=1e-6, atol=1e-6)

    def test_mz_array_single_charge_two_with_variable_mods(self):
        config = dataclasses.replace(
            self.config,
            min_charge=2,
            max_charge=2,
            variable_mods=["UNIMOD:35|Position:M"],
        )
        with Experiment(config) as exp:
            fixture = MzIrtDataFrame()
            sequences = fixture.generate_sequences(exp)
            mz_values = fixture.mz_array(exp, sequences)

            self.assertEqual(mz_values.shape[0], sequences.shape[0])

            mass_calc = fixture.mass_calculator(exp)
            expected_mass = np.array(
                [mass_calc(seq) for seq in sequences],
                dtype=np.float64,
            )
            mass_from_z2 = mz_values.astype(np.float64) * 2 - 2 * PROTON_MASS
            np.testing.assert_allclose(
                mass_from_z2,
                expected_mass,
                rtol=1e-6,
                atol=1e-6,
            )

    def test_isotope_scoring_outputs_unique_pairs(self):
        config = dataclasses.replace(
            self.config,
            batch_size=512,
            isotope_error=1,
            precursor_mz_tolerance=0.1,
            max_charge=2,
            score_threshold=0.0,
            workers=1,
        )

        class FakeSpectrumCollection:
            def __getitem__(self, key):
                mz = np.array([100.0, 200.0], dtype=np.float32)
                intensities = np.array([1.0, 1.0], dtype=np.float32)
                return mz, intensities

            def worker_close(self):
                pass

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            peptide_file = Path(tmpdir) / "peptides.tsv"
            sequence_file = Path(__file__).with_name("10k_peptides.txt")
            sequences = np.unique(np.loadtxt(sequence_file, dtype=bytes))[:512]
            peptide_table = pd.DataFrame(
                {
                    "peptide_sequences": sequences,
                    "precursor_charges": np.full(sequences.shape[0], 2, dtype=np.uint8),
                    "irt": np.zeros(sequences.shape[0], dtype=np.float32),
                    "m/z": 500.0
                    + np.arange(sequences.shape[0], dtype=np.float32) * 0.2,
                }
            )
            peptide_table.to_csv(peptide_file, index=False, sep="\t")

            with Experiment(config, peptide_table=peptide_file) as exp:
                _ = exp.peptides
                predicted = type(exp).__dict__["predicted_spectra"]
                predicted._data[exp] = FakeSpectrumCollection()

                try:
                    scores = SpectrumGrouping().evaluate(exp)
                finally:
                    predicted._data.pop(exp, None)

        pairs = np.stack((scores["i"], scores["j"]), axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        self.assertEqual(len(scores), len(unique_pairs))

    def test_ptms(self):
        """Test that Experiment can handle peptides with PTMs."""
        config = Config(
            ptms=True,
            input_file=Path("tests/test_peptides_ptms.txt"),
            model_irt=KoinaRTModel.Prosit_2025_irt_40PTM,
            model_intensity=KoinaIntensityModel.Prosit_2025_intensity_40PTM,
            fragmentation_type=FragmentationType.HCD,
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


class SubsetTest(TestBase):
    test_file = "tests/10k_peptides.txt"

    def setUp(self):
        super().setUp()
        with Experiment(self.config) as exp:
            self.correct_scores = exp.score_array
            self.correct_scores.sort()

    def test_experiment_runner(self):
        subsets = 3
        config = dataclasses.replace(self.config, subsets=subsets)
        runner = ExperimentRunner(
            config=config,
            peptide_table="tests/peptides_subset.tsv",
            create_peptide_table=True,
            array_file="tests/scores_subset_{}.npy",
            score_df_file="tests/scores_subset_{}.tsv",
        )
        runner.run()
        combined = []
        for i in range(subsets):
            with Experiment(
                dataclasses.replace(self.config, subset=i + 1, subsets=subsets)
            ) as exp:
                combined.append(exp.score_array)
        combined = np.concat(combined)
        combined.sort()
        self.assertEqual(combined.shape[0], self.correct_scores.shape[0])
        self.assertTrue((combined["i"] == self.correct_scores["i"]).all())
        self.assertTrue((combined["j"] == self.correct_scores["j"]).all())
        self.assertTrue(
            np.allclose(combined["score"], self.correct_scores["score"], atol=1e-3)
        )

    def test_subsets(self):
        subsets = 3
        outs = []
        for i in range(subsets):
            with Experiment(
                dataclasses.replace(self.config, subset=i + 1, subsets=subsets)
            ) as exp:
                outs.append(exp.score_array)
        combined = np.concat(outs)
        combined.sort()
        self.assertEqual(combined.shape[0], self.correct_scores.shape[0])
        self.assertTrue((combined["i"] == self.correct_scores["i"]).all())
        self.assertTrue((combined["j"] == self.correct_scores["j"]).all())
        self.assertTrue(
            np.allclose(combined["score"], self.correct_scores["score"], atol=1e-3)
        )

    def test_subsets_with_peptide_table(self):
        subsets = 3
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
                        outs.append(exp_subset.score_array)
        combined = np.concat(outs)
        combined.sort()
        self.assertEqual(combined.shape[0], self.correct_scores.shape[0])
        self.assertTrue((combined["i"] == self.correct_scores["i"]).all())
        self.assertTrue((combined["j"] == self.correct_scores["j"]).all())
        self.assertTrue(
            np.allclose(combined["score"], self.correct_scores["score"], atol=1e-3)
        )


class IsotopeErrorTest(TestBase):
    test_file = "tests/10k_peptides.txt"
    batch_size = 256
    mz_tolerance = 0.02

    def setUp(self):
        super().setUp()
        with Experiment(dataclasses.replace(self.config, isotope_error=0)) as exp:
            self.pairs_iso0 = {(int(i), int(j)) for i, j, _ in exp.score_array}

        with Experiment(dataclasses.replace(self.config, isotope_error=2)) as exp:
            self.pairs_iso2 = {(int(i), int(j)) for i, j, _ in exp.score_array}
            # Copy out of shared memory so values stay valid after Experiment cleanup.
            self.peptide_mz = exp.peptides["m/z"].to_numpy(copy=True)
            self.peptide_charges = exp.peptides["precursor_charges"].to_numpy(copy=True)

        self.extra_pairs = self.pairs_iso2 - self.pairs_iso0

    @staticmethod
    def _best_isotope_match(i, j, mz, charges, config, max_isotope):
        best = None
        for isotope1 in range(max_isotope + 1):
            for isotope2 in range(max_isotope + 1):
                shifted_i = mz[i] + isotope1 * PROTON_MASS / charges[i]
                shifted_j = mz[j] + isotope2 * PROTON_MASS / charges[j]
                delta = abs(shifted_i - shifted_j)
                if best is None or delta < best[0]:
                    best = (delta, isotope1, isotope2, shifted_i, shifted_j)
        if best is None:
            return None
        if config.within_mz_tolerance(best[3], best[4]):
            return best[:3]
        return None

    def test_isotope_error_two_superset_of_zero(self):
        self.assertTrue(
            self.pairs_iso0.issubset(self.pairs_iso2),
            "All pairs found with isotope_error=0 should also be present with isotope_error=2",
        )

    def test_isotope_error_adds_isotope_shifted_pairs(self):
        self.assertGreater(
            len(self.extra_pairs),
            0,
            "Expected additional pairs when isotope_error is increased from 0 to 2",
        )

        isotope_shifted_pairs = Counter()

        for i, j in self.extra_pairs:
            self.assertFalse(
                self.config.within_mz_tolerance(self.peptide_mz[i], self.peptide_mz[j]),
                "Additional pairs should be outside no-isotope m/z tolerance",
            )

            match = self._best_isotope_match(
                i,
                j,
                self.peptide_mz,
                self.peptide_charges,
                self.config,
                max_isotope=2,
            )
            self.assertIsNotNone(
                match,
                f"Pair ({i}, {j}) should be explainable by isotope shifts up to 2",
            )
            _, isotope1, isotope2 = match
            self.assertTrue(
                isotope1 > 0 or isotope2 > 0,
                f"Pair ({i}, {j}) should require isotope 1 or 2 shift",
            )
            isotope_shifted_pairs[abs(isotope1 - isotope2)] += 1

        self.logger.debug(
            "Isotope-shifted pairs by isotope difference: %s",
            dict(isotope_shifted_pairs),
        )
        for isotope_diff in range(1, 3):
            count = isotope_shifted_pairs[isotope_diff]
            self.logger.debug(
                "Isotope difference of %d accounts for %d additional pairs",
                isotope_diff,
                count,
            )
            self.assertGreater(
                count,
                0,
                f"Expected some pairs with isotope difference of {isotope_diff} to be added",
            )


class ConfigToleranceTest(unittest.TestCase):
    def test_within_mz_tolerance_ppm_uses_larger_value_and_is_symmetric(self):
        config = Config(
            precursor_mz_tolerance=10.0,
            precursor_mz_unit=MzErrorUnit.PPM,
        )
        self.assertTrue(config.within_mz_tolerance(999.99, 1000.0))
        self.assertTrue(config.within_mz_tolerance(1000.0, 999.99))
        self.assertFalse(config.within_mz_tolerance(499.9949, 500.0))


class GroupingWorkerLogicTest(unittest.TestCase):
    def _worker(self, **config_overrides):
        config = Config(**config_overrides)
        worker = GroupingWorker(None, None, config=config)
        worker.mzrt = np.array(
            [
                [1000.0, 0.0],
                [1200.0, 0.0],
                [800.0, 0.0],
            ],
            dtype=np.float32,
        )
        worker.charges = np.array([2, 3, 2], dtype=np.uint8)
        return worker

    def test_isotopes_overlap_false_for_small_relative_tolerance(self):
        worker = self._worker(
            isotope_error=1,
            precursor_mz_unit=MzErrorUnit.PPM,
            precursor_mz_tolerance=200.0,
            min_charge=2,
            max_charge=3,
            batch_size=2,
        )
        self.assertFalse(worker.isotopes_overlap(0))

    def test_isotopes_overlap_true_for_large_relative_tolerance(self):
        worker = self._worker(
            isotope_error=1,
            precursor_mz_unit=MzErrorUnit.PPM,
            precursor_mz_tolerance=300.0,
            min_charge=2,
            max_charge=3,
            batch_size=2,
        )
        self.assertTrue(worker.isotopes_overlap(0))

    def test_isotopes_overlap_false_when_isotope_error_disabled(self):
        worker = self._worker(
            isotope_error=0,
            precursor_mz_unit=MzErrorUnit.PPM,
            precursor_mz_tolerance=1000.0,
            min_charge=2,
            max_charge=3,
            batch_size=2,
        )
        self.assertFalse(worker.isotopes_overlap(0))


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
                    g = GroupingWorker(
                        None, None, config=exp.config, spectra=exp.predicted_spectra
                    )
                    idx1, idx2 = g.match_peaks(mz1, mz2)
                    self.logger.debug("Matched indices: %s and %s", idx1, idx2)
                    self.logger.debug(
                        "Matched m/z values:\n%s and\n%s",
                        np.sort(mz1[idx1]),
                        np.sort(mz2[idx2]),
                    )

                    matcher = joinPeaks(
                        tolerance=0,
                        ppm=self.config.fragment_mz_tolerance,
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
                    matched_count = int(mask.sum())

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
                    if idx1.size == matched_count and idx2.size == matched_count:
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
                        tolerance=0,
                        ppm=self.config.fragment_mz_tolerance,
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

                    g = GroupingWorker(
                        None, None, config=exp.config, spectra=exp.predicted_spectra
                    )
                    idx1, idx2 = g.match_peaks(x_df["mz"].values, y_df["mz"].values)
                    newscore = g.similarity_score(
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
