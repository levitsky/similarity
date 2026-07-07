import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from similarity.grouping import GroupingWorker
from similarity.prediction import PredictedSpectrumCollection
from similarity.utils.abc import IndexType
from similarity.utils.config import Config, MzErrorUnit
from similarity.utils.spectrum_collection.cached import CachedSpectrumCollection
from similarity.utils.spectrum_collection.sharedarray import (
    SharedArraySpectrumCollection,
)


def dense_match_peaks(mz1: np.ndarray, mz2: np.ndarray, atol: float, rtol: float):
    mask = np.isclose(mz1[:, None], mz2[None, :], atol=atol, rtol=rtol)
    return np.where(mask)


def python_similarity_score(
    intensities1: np.ndarray,
    intensities2: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
) -> float:
    wx = intensities1[idx1]
    wy = intensities2[idx2]
    num = np.sum(wx * wy) ** 2
    denom1 = np.sum(intensities1**2)
    denom2 = np.sum(intensities2**2)
    ndotproduct = np.clip(num / denom1 / denom2, -1.0, 1.0)
    return 1 - 2 * np.arccos(ndotproduct) / np.pi


class FakeSpectrumIndex:
    def __init__(self, spectra):
        self.spectra = spectra

    def __getitem__(self, key):
        return self.spectra[key]

    def __contains__(self, key):
        return key in self.spectra

    def __len__(self):
        return len(self.spectra)

    def close(self):
        pass

    def wait(self):
        pass

    def fill_from_cache(self, inputs, output):
        for i, (_, row) in enumerate(inputs.iterrows()):
            output[i] = self.spectra[
                (row["peptide_sequences"], row["precursor_charges"])
            ]


class MatchPeaksTest(unittest.TestCase):
    def test_match_peaks_matches_dense_reference(self):
        cases = [
            (
                np.array([100.0, 100.01, 100.03, 101.0], dtype=np.float32),
                np.array([99.995, 100.01, 100.02, 101.0], dtype=np.float32),
                MzErrorUnit.Th,
                0.02,
            ),
            (
                np.array([150.0, 150.0, 150.02, 150.04], dtype=np.float64),
                np.array([149.99, 150.0, 150.0, 150.05], dtype=np.float64),
                MzErrorUnit.Th,
                0.02,
            ),
            (
                np.array([], dtype=np.float32),
                np.array([500.0, 500.01], dtype=np.float32),
                MzErrorUnit.PPM,
                10.0,
            ),
        ]

        rng = np.random.default_rng(42)
        for _ in range(25):
            mz1 = np.sort(rng.uniform(100.0, 1500.0, size=32).astype(np.float32))
            mz2 = np.sort(rng.uniform(100.0, 1500.0, size=40).astype(np.float32))
            cases.append((mz1, mz2, MzErrorUnit.Th, 0.015))
            cases.append((mz1, mz2, MzErrorUnit.PPM, 20.0))

        for mz1, mz2, unit, tolerance in cases:
            with self.subTest(
                size=(mz1.size, mz2.size), unit=unit, tolerance=tolerance
            ):
                config = Config(fragment_mz_unit=unit, fragment_mz_tolerance=tolerance)
                g = GroupingWorker(
                    None, None, config=config, spectra_1=None, spectra_2=None
                )
                atol = tolerance if unit == MzErrorUnit.Th else 0.0
                rtol = tolerance / 1e6 if unit == MzErrorUnit.PPM else 0.0
                expected = dense_match_peaks(mz1, mz2, atol, rtol)
                actual = g.match_peaks(mz1, mz2)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])

    def test_similarity_score_matches_python_reference(self):
        rng = np.random.default_rng(123)
        for _ in range(50):
            mz1 = np.sort(rng.uniform(100.0, 1500.0, size=64).astype(np.float32))
            mz2 = np.sort(rng.uniform(100.0, 1500.0, size=80).astype(np.float32))
            intensities1 = np.sqrt(rng.uniform(1e-5, 1.0, size=64)).astype(np.float32)
            intensities2 = np.sqrt(rng.uniform(1e-5, 1.0, size=80)).astype(np.float32)
            g = GroupingWorker(
                None,
                None,
                config=Config(
                    fragment_mz_unit=MzErrorUnit.Th, fragment_mz_tolerance=0.02
                ),
                spectra_1=None,
                spectra_2=None,
            )
            idx1, idx2 = g.match_peaks(mz1, mz2)

            expected = python_similarity_score(intensities1, intensities2, idx1, idx2)
            actual = g.similarity_score(intensities1, intensities2, idx1, idx2)
            self.assertAlmostEqual(actual, expected, places=6)

    def test_similarity_score_matches_python_reference_with_duplicate_matches(self):
        intensities1 = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        intensities2 = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        idx1 = np.array([0, 1, 1, 2, 3], dtype=np.intp)
        idx2 = np.array([1, 0, 1, 1, 2], dtype=np.intp)

        expected = python_similarity_score(intensities1, intensities2, idx1, idx2)
        actual = GroupingWorker.similarity_score(intensities1, intensities2, idx1, idx2)
        self.assertTrue(np.isfinite(expected))
        self.assertTrue(np.isfinite(actual))
        self.assertAlmostEqual(actual, expected, places=6)

    def test_preprocess_predictions_sorts_mz_and_intensities_together(self):
        result = {
            "mz": [np.array([200.0, 100.0, 150.0], dtype=np.float32)],
            "intensities": [np.array([9.0, 1.0, 4.0], dtype=np.float32)],
        }

        PredictedSpectrumCollection.preprocess_predictions(result)

        np.testing.assert_allclose(result["mz"][0], [100.0, 150.0, 200.0])
        np.testing.assert_allclose(result["intensities"][0], [1.0, 2.0, 3.0])

    def test_shared_array_fill_from_cache_preserves_sorted_order_after_truncation(self):
        experiment = SimpleNamespace(
            peptides=pd.DataFrame(
                {
                    "peptide_sequences": [b"PEP"],
                    "precursor_charges": [2],
                }
            ),
            config=SimpleNamespace(max_peaks=3),
            spectrum_file=None,
        )
        collection = SharedArraySpectrumCollection(experiment)
        spectra = {
            (b"PEP", 2): (
                np.array([300.0, 100.0, 250.0, 200.0], dtype=np.float32),
                np.array([1.0, 4.0, 3.0, 2.0], dtype=np.float32),
            )
        }
        index = FakeSpectrumIndex(spectra)

        try:
            collection.fill_from_cache(index)
            mz, intensities = collection[0]
        finally:
            collection.close()

        np.testing.assert_allclose(mz, [100.0, 200.0, 250.0])
        np.testing.assert_allclose(intensities, [4.0, 2.0, 3.0])

    def test_cached_collection_applies_truncation_on_read_and_returns_sorted_mz(self):
        peptides = pd.DataFrame(
            {
                "peptide_sequences": [b"PEP"],
                "precursor_charges": [2],
            }
        )
        spectra = {
            (b"PEP", 2): (
                np.array([300.0, 100.0, 250.0, 200.0], dtype=np.float32),
                np.array([1.0, 4.0, 3.0, 2.0], dtype=np.float32),
            )
        }
        experiment = SimpleNamespace(
            peptides=peptides,
            config=SimpleNamespace(max_peaks=3, batch_size=1),
            cache={IndexType.INTENSITY: FakeSpectrumIndex(spectra)},
        )
        collection = CachedSpectrumCollection(experiment)

        mz, intensities = collection[0]

        np.testing.assert_allclose(mz, [100.0, 200.0, 250.0])
        np.testing.assert_allclose(intensities, [4.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
