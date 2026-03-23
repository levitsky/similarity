import math
from typing import Iterable, TYPE_CHECKING, Any
from scipy.spatial import cKDTree
import numpy as np
import logging
import multiprocessing as mp
from .utils.abc import Fixture
from .utils.utils import ExperimentWorker
from .prediction import MzIrtDataFrame


if TYPE_CHECKING:
    from .experiment import Experiment, Config
    from .utils.abc import SpectrumCollection
    from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger(__name__)


class GroupingWorker(ExperimentWorker):
    """Worker process that receives batch numbers and performs subtree creation, neighbor search, and tolerance checking for a batch of spectra.
    It then scores the valid pairs and sends the results back to the main process if they pass the configured score threshold.
    """

    config: "Config"
    radius: float
    shape: tuple[int, ...]
    seq_dtype: np.dtype
    shared_memory: dict[str, "SharedMemory"]
    spectra: "SpectrumCollection"
    peptides: np.ndarray
    charges: np.ndarray
    batch_size: int

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

    def kdtree(self, offset: int) -> cKDTree:
        bsize = self.batch_size
        arr = self.mzrt[offset : offset + bsize]
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
        wx = intensities1[idx1]  # sqrt was applied in preprocess_predictions
        wy = intensities2[idx2]
        # the numerator only has matching peaks intensities,
        # but the denominator has the sum of all intensities
        num = np.sum(wx * wy) ** 2
        denom1 = np.sum(intensities1**2)
        denom2 = np.sum(intensities2**2)

        ndotproduct = num / denom1 / denom2
        score = 1 - 2 * np.arccos(ndotproduct) / np.pi

        return score

    def score_pair(self, i: int, j: int) -> float:
        mz1, intensities1 = self.spectra[i]
        mz2, intensities2 = self.spectra[j]
        idx1, idx2 = GroupingWorker.match_peaks(
            mz1,
            mz2,
            atol=self.config.peak_tolerance,
            rtol=self.config.peak_ppm / 1e6,
        )
        return GroupingWorker.similarity_score(intensities1, intensities2, idx1, idx2)

    @staticmethod
    def encode_result(i: int, matches: list[int], scores: list[float]) -> tuple:
        return (
            i,
            np.array(matches, dtype=np.int32).tobytes(),
            np.array(scores, dtype=np.float32).tobytes(),
        )

    @staticmethod
    def decode_result(encoded: tuple) -> tuple[int, np.ndarray, np.ndarray]:
        i, matches_bytes, scores_bytes = encoded
        matches = np.frombuffer(matches_bytes, dtype=np.int32)
        scores = np.frombuffer(scores_bytes, dtype=np.float32)
        return i, matches, scores

    def process_batch(
        self,
        offset: int,
        previous_end: int,  # end of the previous batch, used to skip spectra that were already processed in the previous batch due to overlap
    ) -> Iterable[tuple[int, list[int], list[float]]]:

        logger.info(
            "Processing peptides %d to %d...",
            offset + 1,
            offset + self.batch_size,
        )
        subtree = self.kdtree(offset)

        # format PID as str because it can be None if called from a non-multiprocessing context
        logger.debug(
            "Batch offset %d, end of previous batch %d, PID %s",
            offset,
            previous_end,
            self.pid,
        )

        neighbors = subtree.query_ball_tree(subtree, r=self.radius)

        for i_, indices in enumerate(neighbors):
            matches = []
            scores = []
            i = i_ + offset
            for j_ in indices:
                j = j_ + offset
                if i < j and j >= previous_end and self.within_tolerance(i, j):
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
        logger.debug("Worker started with PID %d", self.pid)
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
            task = self.task_queue.get()
            if task is None:
                logger.debug(
                    "Worker with PID %d received None, wrapping up...", self.pid
                )
                break
            offset, prev = task
            for result in self.process_batch(offset, prev):
                self.result_queue.put(self.encode_result(*result))
            logger.debug(
                "Finished batch at offset %d in worker %d. Current output queue size: %d",
                offset,
                self.pid,
                self.result_queue.qsize(),
            )
        self.result_queue.put(None)
        for shm in self.shared_memory.values():
            shm.close()
        self.spectra.worker_close()
        logger.debug("Worker with PID %d finished", self.pid)


class SpectrumGrouping(Fixture):
    max_queue_size: int = 100000

    def radius(self, experiment: "Experiment") -> float:
        mz_tol = experiment.config.mz_tolerance
        irt_tol = experiment.config.irt_tolerance
        if experiment.config.model_ccs is not None:
            ccs_rtol = experiment.config.ccs_rtolerance
            ccs_tol = ccs_rtol * experiment.peptides["ccs"].max()
            return np.sqrt(mz_tol**2 + irt_tol**2 + ccs_tol**2)
        return np.sqrt(mz_tol**2 + irt_tol**2)

    def batch_offsets(
        self, bsize: int, experiment: "Experiment"
    ) -> tuple[int, list[int]]:
        """
        Returns the offsets for each batch. Given the batch size, partition the "mzrt" array that is sorted by m/z
        so that batches overlap by the configured m/z tolerance. This ensures that all spectra that could potentially
        be within tolerance of each other are processed in the same batch.
        """
        c = experiment.config
        i, dim = MzIrtDataFrame.sorting_dimension(experiment.peptides, c)
        if i == 0:
            tol = c.mz_tolerance
        elif i == 1:
            tol = c.irt_tolerance
        else:
            tol = c.ccs_rtolerance * experiment.peptides["ccs"].max()
        values = experiment.peptides[dim].values
        offsets = [0]
        while offsets[-1] < len(values):
            end_of_batch = next_offset = offsets[-1] + bsize
            if next_offset >= len(values):
                break
            while values[end_of_batch] - values[next_offset - 1] <= tol:
                next_offset -= 1
                if (
                    next_offset <= offsets[-1]
                    or end_of_batch - next_offset >= bsize // 5
                ):
                    logger.warning(
                        "Batch size is too small to accommodate the %s tolerance. "
                        "Increasing the batch size to %d...",
                        dim,
                        bsize * 2,
                    )
                    return self.batch_offsets(bsize * 2, experiment)
            offsets.append(next_offset)
        logger.debug("Calculated batch offsets: %s", offsets[:10])
        return bsize, offsets

    def evaluate(self, experiment: "Experiment") -> np.ndarray:
        bsize, offsets = self.batch_offsets(experiment.config.batch_size, experiment)
        nb = len(offsets)
        logger.info(
            "Processing %d spectra in %d batches...", len(experiment.peptides), nb
        )
        dtype = np.dtype([("i", np.int32), ("j", np.int32), ("score", np.float32)])
        if experiment.config.workers > 1:
            logger.info(
                "Grouping with %d workers...",
                experiment.config.workers,
            )
            in_queue = mp.Queue(maxsize=self.max_queue_size)
            out_queue = mp.Queue(maxsize=self.max_queue_size)
            workers = [
                GroupingWorker(
                    in_queue,
                    out_queue,
                    config=experiment.config,
                    batch_size=bsize,
                    shared_memory=MzIrtDataFrame._shared_memory[experiment],
                    shape=(
                        len(experiment.peptides),
                        3 if experiment.config.model_ccs is not None else 2,
                    ),
                    seq_dtype=experiment.peptides["peptide_sequences"].dtype,
                    spectra=experiment.predicted_spectra,
                    radius=self.radius(experiment),
                )
                for _ in range(experiment.config.workers)
            ]
            for worker in workers:
                worker.start()

            for offset, prev in zip(offsets, [-bsize] + offsets[:-1]):
                in_queue.put((offset, prev + bsize))

            for _ in workers:
                in_queue.put(None)

            def produce_results():
                workers_done = 0
                count = 0
                while workers_done < len(workers):
                    item = out_queue.get()
                    if item is None:
                        workers_done += 1
                        logger.debug(
                            "%d of %d workers done", workers_done, len(workers)
                        )
                    else:
                        i, matches, scores = GroupingWorker.decode_result(item)
                        count += 1
                        for m, s in zip(matches, scores):
                            yield (i, m, s)
                        if count % bsize == 0:
                            logger.debug("Processed %d peptides...", count)

            scores = np.fromiter(produce_results(), dtype=dtype)
            for worker in workers:
                worker.join()
        else:
            pseudoworker = GroupingWorker(
                None,
                None,
                config=experiment.config,
                batch_size=bsize,
                spectra=experiment.predicted_spectra,
                radius=self.radius(experiment),
            )

            pseudoworker.mzrt = np.ndarray(
                shape=(
                    len(experiment.peptides),
                    3 if experiment.config.model_ccs is not None else 2,
                ),
                dtype=np.float32,
                buffer=MzIrtDataFrame._shared_memory[experiment]["mzrt"].buf,
            )
            pseudoworker.peptides = experiment.peptides["peptide_sequences"].values
            pseudoworker.charges = experiment.peptides["precursor_charges"].values

            def produce_results():
                for offset, prev in zip(offsets, [-bsize] + offsets[:-1]):
                    for item in pseudoworker.process_batch(offset, prev + bsize):
                        i, matches, scores = item
                        for m, s in zip(matches, scores):
                            yield (i, m, s)

            scores = np.fromiter(produce_results(), dtype=dtype)
        logger.info(
            "Finished scoring, found %d pairs with score above %f",
            len(scores),
            experiment.config.score_threshold,
        )
        return scores
