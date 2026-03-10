import math
from typing import Iterable, TYPE_CHECKING, Any
from scipy.spatial import cKDTree
import numpy as np

import logging
import multiprocessing as mp
from .utils import Fixture, ExperimentWorker


if TYPE_CHECKING:
    from .experiment import Experiment, Config
    from .utils import SpectrumIndex
    from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger(__name__)


class GroupingWorker(ExperimentWorker):
    """Worker process that receives batch numbers and performs subtree creation, neighbor search, and tolerance checking for a batch of spectra.
    It then scores the valid pairs and sends the results back to the main process if they pass the configured score threshold.
    """

    config: "Config"
    tree: cKDTree
    radius: float
    shape: tuple[int, ...]
    seq_dtype: np.dtype
    shared_memory: dict[str, "SharedMemory"]
    spectra: "SpectrumIndex"
    peptides: np.ndarray
    charges: np.ndarray

    def within_tolerance_2d(self, i: int, j: int) -> bool:
        arr = self.mzrt
        mz_tol = self.config.mz_tolerance
        irt_tol = self.config.irt_tolerance
        return (
            abs(arr[i, 0] - arr[j, 0]) <= mz_tol
            and abs(arr[i, 1] - arr[j, 1]) <= irt_tol
        )

    def within_tolerance_3d(self, i: int, j: int) -> bool:
        arr = self.mzrt
        mz_tol = self.config.mz_tolerance
        irt_tol = self.config.irt_tolerance
        ccs_rtol = self.config.ccs_rtolerance
        return (
            abs(arr[i, 0] - arr[j, 0]) <= mz_tol
            and abs(arr[i, 1] - arr[j, 1]) <= irt_tol
            and abs(arr[i, 2] - arr[j, 2]) <= ccs_rtol * max(arr[i, 2], arr[j, 2])
        )

    def tolerance_check(self):
        if self.config.model_ccs is not None:
            return self.within_tolerance_3d
        return self.within_tolerance_2d

    def kdtree(self, batch: int | None = None) -> cKDTree:
        arr = self.mzrt
        if batch is not None:
            arr = arr[
                batch * self.config.batch_size : (batch + 1) * self.config.batch_size
            ]
        return cKDTree(arr)

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
        wx = np.sqrt(intensities1[idx1])
        wy = np.sqrt(intensities2[idx2])
        # the numerator only has matching peaks intensities,
        # but the denominator has the sum of all intensities
        num = np.sum(wx * wy) ** 2
        denom1 = np.sum(intensities1)
        denom2 = np.sum(intensities2)

        ndotproduct = num / denom1 / denom2
        score = 1 - 2 * np.arccos(ndotproduct) / np.pi

        return score

    def spectrum_key(self, i: int) -> Any:
        return (self.peptides[i], self.charges[i])

    def get_spectrum(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.spectra[self.spectrum_key(i)]

    def score_pair(self, i: int, j: int) -> float:
        mz1, intensities1 = self.get_spectrum(i)
        mz2, intensities2 = self.get_spectrum(j)
        idx1, idx2 = GroupingWorker.match_peaks(
            mz1,
            mz2,
            atol=self.config.peak_tolerance,
            rtol=self.config.peak_ppm / 1e6,
        )
        return GroupingWorker.similarity_score(intensities1, intensities2, idx1, idx2)

    def process_batch(
        self,
        batch: int,
    ) -> Iterable[tuple[int, list[int], list[float]]]:
        """Make a batch of size config.batch_size of the input array and find potential neighbor pairs, submit them to `in_queue`,
        then get the filtered pairs from `out_queue` (in chunks). Also sets the "in pairs" column of the dataframe to True for spectra that are in any pair.
        """

        logger.info(
            "Processing batch %d...",
            batch + 1,
        )
        subtree = self.kdtree(batch)

        offset = batch * self.config.batch_size
        logger.debug("Batch idx %d, offset %d", batch, offset)

        neighbors = self.tree.query_ball_tree(subtree, r=self.radius)

        for i, indices in enumerate(neighbors):
            matches = []
            scores = []
            for x in indices:
                j = x + offset
                if i < j and self.within_tolerance(i, j):
                    score = self.score_pair(i, j)
                    if score >= self.config.score_threshold:
                        matches.append(j)
                        scores.append(score)
            if matches:
                yield (i, matches, scores)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.within_tolerance = self.tolerance_check()

    def run(self) -> None:
        self.mzrt = np.ndarray(
            shape=self.shape, dtype=np.float32, buffer=self.shared_memory["mzrt"].buf
        )
        self.peptides = np.ndarray(
            shape=(self.shape[0],),
            dtype=self.seq_dtype,
            buffer=self.shared_memory["peptide_sequences"].buf,
        )
        self.charges = np.ndarray(
            shape=(self.shape[0],),
            dtype=np.uint8,
            buffer=self.shared_memory["precursor_charges"].buf,
        )
        while True:
            batch = self.task_queue.get()
            if batch is None:
                break
            for result in self.process_batch(batch):
                self.result_queue.put(result)
        self.result_queue.put(None)
        for shm in self.shared_memory.values():
            shm.close()


class SpectrumGrouping(Fixture):
    def kdtree(self, experiment: "Experiment", batch: int | None = None) -> cKDTree:
        df = experiment.peptides
        names = ["m/z", "irt"]
        if experiment.config.model_ccs is not None:
            names.append("ccs")
        if batch is not None:
            df = df.iloc[
                batch
                * experiment.config.batch_size : (batch + 1)
                * experiment.config.batch_size
            ]
        return cKDTree(df[names].values)

    def radius(self, experiment: "Experiment") -> float:
        mz_tol = experiment.config.mz_tolerance
        irt_tol = experiment.config.irt_tolerance
        if experiment.config.model_ccs is not None:
            ccs_rtol = experiment.config.ccs_rtolerance
            ccs_tol = ccs_rtol * experiment.peptides["ccs"].max()
            return np.sqrt(mz_tol**2 + irt_tol**2 + ccs_tol**2)
        return np.sqrt(mz_tol**2 + irt_tol**2)

    def nbatches(self, experiment: "Experiment") -> int:
        return math.ceil(len(experiment.peptides) / experiment.config.batch_size)

    def evaluate(self, experiment: "Experiment") -> np.ndarray:
        tree = self.kdtree(experiment)
        logger.info("Built cKDTree with %d nodes from %d points", tree.size, tree.n)
        nb = self.nbatches(experiment)
        logger.info("Processing %d spectra in %d batches...", tree.n, nb)
        dtype = np.dtype([("i", int), ("j", int), ("score", float)])
        if experiment.config.workers > 1:
            logger.info(
                "Grouping with %d workers...",
                experiment.config.workers,
            )
            in_queue = mp.Queue()
            out_queue = mp.Queue()
            workers = [
                GroupingWorker(
                    in_queue,
                    out_queue,
                    config=experiment.config,
                    shared_memory=experiment._shared_memory,
                    shape=(
                        len(experiment.peptides),
                        3 if experiment.config.model_ccs is not None else 2,
                    ),
                    seq_dtype=experiment.peptides["peptide_sequences"].dtype,
                    tree=tree,
                    spectra=experiment.predicted_spectra,
                    radius=self.radius(experiment),
                )
                for _ in range(experiment.config.workers)
            ]
            for worker in workers:
                worker.start()

            for batch in range(nb):
                in_queue.put(batch)

            for _ in range(experiment.config.workers):
                in_queue.put(None)

            def produce_results():
                workers_done = 0
                while workers_done < len(workers):
                    item = out_queue.get()
                    if item is None:
                        workers_done += 1
                    else:
                        i, matches, scores = item
                        for m, s in zip(matches, scores):
                            yield (i, m, s)

            scores = np.fromiter(produce_results(), dtype=dtype)
            for worker in workers:
                worker.join()
        else:
            pseudoworker = GroupingWorker(
                None,
                None,
                config=experiment.config,
                tree=tree,
                spectra=experiment.predicted_spectra,
                radius=self.radius(experiment),
            )

            pseudoworker.mzrt = np.ndarray(
                shape=(
                    len(experiment.peptides),
                    3 if experiment.config.model_ccs is not None else 2,
                ),
                dtype=np.float32,
                buffer=experiment._shared_memory["mzrt"].buf,
            )
            pseudoworker.peptides = experiment.peptides["peptide_sequences"].values
            pseudoworker.charges = experiment.peptides["precursor_charges"].values

            def produce_results():
                for batch in range(nb):
                    for item in pseudoworker.process_batch(batch):
                        i, matches, scores = item
                        for m, s in zip(matches, scores):
                            yield (i, m, s)

            scores = np.fromiter(produce_results(), dtype=dtype)
        return scores
