from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import logging
import multiprocessing as mp
from .utils import Fixture, Index

if TYPE_CHECKING:
    from .experiment import Experiment

logger = logging.getLogger(__name__)


class ScoringWorker(mp.Process):
    def __init__(
        self,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        experiment: "Experiment",
    ):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.experiment = experiment

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

    def score_pair(self, i: int, j: int) -> float:
        mz_irt_df = self.experiment.mz_irt_df
        spectra = self.experiment.predicted_spectra
        pep1 = mz_irt_df.loc[i, "peptide_sequences"]
        pep2 = mz_irt_df.loc[j, "peptide_sequences"]
        mz1, intensities1 = spectra[pep1]
        mz2, intensities2 = spectra[pep2]
        idx1, idx2 = ScoringWorker.match_peaks(
            mz1,
            mz2,
            atol=self.experiment.config.peak_tolerance,
            rtol=self.experiment.config.peak_ppm / 1e6,
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
        return ScoringWorker.similarity_score(intensities1, intensities2, idx1, idx2)

    def run(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            i, j = task
            score = self.score_pair(i, j)
            self.result_queue.put((i, j, score))


class ProcessedPairs(Fixture):

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
        results = []
        if experiment.config.workers > 1:
            logger.info(
                "Scoring %d pairs with %d workers...",
                len(index_array),
                experiment.config.workers,
            )
            in_queue = mp.Queue()
            out_queue = mp.Queue()
            workers = [
                ScoringWorker(in_queue, out_queue, experiment=experiment)
                for _ in range(experiment.config.workers)
            ]
            for worker in workers:
                worker.start()
            for i, j in index_array:
                in_queue.put((i, j))
            for _ in workers:
                in_queue.put(None)

            for _ in index_array:
                i, j, score = out_queue.get()
                results.append(self.format_result(i, j, score, experiment))
            for worker in workers:
                worker.join()
        else:
            logger.info("Scoring %d pairs with a single worker...", len(index_array))
            for i, j in index_array:
                w = ScoringWorker(None, None, experiment=experiment)
                score = ScoringWorker.score_pair(w, i, j)
                results.append(self.format_result(i, j, score, experiment))
        return pd.DataFrame(results)
