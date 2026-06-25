import math
from typing import Iterable, TYPE_CHECKING
from scipy.spatial import cKDTree
import numpy as np
import logging
import multiprocessing as mp
import itertools
from .utils.abc import Fixture
from .utils.utils import ExperimentWorker
from .utils.config import PROTON_MASS, MzErrorUnit
from .prediction import MzIrtDataFrame
from ._match_peaks import match_peaks_sorted, similarity_score as c_similarity_score

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
    trees: list[cKDTree]
    radius: float
    shape: tuple[int, ...]
    nbatches: int
    seq_dtype: np.dtype
    shared_memory: dict[str, "SharedMemory"]
    spectra: "SpectrumCollection"
    peptides: np.ndarray
    charges: np.ndarray
    scaling_factors: np.ndarray
    mzrt: np.ndarray
    previous_end: int

    def within_mz_tolerance_no_isotope(self, i: int, j: int) -> bool:
        arr = self.mzrt
        return self.config.within_mz_tolerance(arr[i, 0], arr[j, 0])

    def within_mz_tolerance_equal_charge(self, i: int, j: int) -> bool:
        arr = self.mzrt
        charge = self.charges[i]
        if i > j:
            i, j = j, i
        # now i < j, so mz_i <= mz_j
        if self.config.within_mz_tolerance(arr[i, 0], arr[j, 0]):
            return True

        # For equal charge, only isotope-difference matters: (iso1 - iso2).
        # This avoids redundant checks like (1, 1), (2, 2), ... while still
        # covering both orderings (iso1 > iso2 and iso1 < iso2).
        shift = PROTON_MASS / charge
        return any(
            self.config.within_mz_tolerance(
                arr[i, 0] + delta_isotope * shift, arr[j, 0]
            )
            for delta_isotope in range(1, self.config.isotope_error + 1)
        )

    def within_mz_tolerance_different_charge(self, i: int, j: int) -> bool:
        arr = self.mzrt
        return any(
            self.config.within_mz_tolerance(
                arr[i, 0] + isotope1 * PROTON_MASS / self.charges[i],
                arr[j, 0] + isotope2 * PROTON_MASS / self.charges[j],
            )
            for isotope1, isotope2 in itertools.product(
                range(self.config.isotope_error + 1), repeat=2
            )
        )  # (0, 0) is included, so this also covers the case where no isotope error is applied

    def within_mz_tolerance_with_isotopes(self, i: int, j: int) -> bool:
        if self.charges[i] == self.charges[j]:
            return self.within_mz_tolerance_equal_charge(i, j)
        else:
            return self.within_mz_tolerance_different_charge(i, j)

    def within_tolerance_2d(self, i: int, j: int) -> bool:
        arr = self.mzrt
        irt_tol = self.config.irt_tolerance
        return self.within_mz_tolerance(i, j) and abs(arr[i, 1] - arr[j, 1]) <= irt_tol

    def within_tolerance_3d(self, i: int, j: int) -> bool:
        arr = self.mzrt
        irt_tol = self.config.irt_tolerance
        ccs_rtol = self.config.ccs_rtolerance
        return (
            self.within_mz_tolerance(i, j)
            and abs(arr[i, 1] - arr[j, 1]) <= irt_tol
            and abs(arr[i, 2] - arr[j, 2]) <= ccs_rtol * max(arr[i, 2], arr[j, 2])
        )

    def tolerance_check(self):
        if self.config.model_ccs is not None:
            return self.within_tolerance_3d
        return self.within_tolerance_2d

    def mz_tolerance_check(self):
        if self.config.isotope_error == 0:
            return self.within_mz_tolerance_no_isotope
        return self.within_mz_tolerance_with_isotopes

    def kdtree(self, batch: int, isotope: int) -> cKDTree:
        bsize = self.config.batch_size
        arr = self.mzrt[batch * bsize : (batch + 1) * bsize]
        if isotope:
            arr = arr.copy()
            arr[:, 0] += (
                isotope
                * PROTON_MASS
                / self.charges[batch * bsize : (batch + 1) * bsize]
            )
        return cKDTree(arr * self.scaling_factors)

    def match_peaks(self, mz1: np.ndarray, mz2: np.ndarray):
        return match_peaks_sorted(mz1, mz2, self.atol, self.rtol)

    @staticmethod
    def similarity_score(
        intensities1: np.ndarray,
        intensities2: np.ndarray,
        idx1: np.ndarray,
        idx2: np.ndarray,
    ) -> float:
        return c_similarity_score(intensities1, intensities2, idx1, idx2)

    def score_pair(self, i: int, j: int) -> float:
        mz1, intensities1 = self.spectra[i]
        mz2, intensities2 = self.spectra[j]
        idx1, idx2 = self.match_peaks(mz1, mz2)
        return self.similarity_score(intensities1, intensities2, idx1, idx2)

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
        batch: int,
    ) -> Iterable[tuple[int, list[int], list[float]]]:

        logger.info("Processing batch %d of %d...", batch + 1, self.nbatches)

        offset = batch * self.config.batch_size
        # format PID as str because it can be None if called from a non-multiprocessing context
        logger.debug("Batch idx %d, offset %d, PID %s", batch, offset, self.pid)

        bsize = self.config.batch_size
        last_idx = min((batch + 1) * bsize, self.mzrt.shape[0]) - 1
        max_batch_mz = self.mzrt[last_idx, 0]
        batch_mz_tol = self.config.absolute_mz_error(max_batch_mz)
        radius = batch_mz_tol * np.sqrt(self.mzrt.shape[1])
        neighbors = []
        subtrees = []
        for isotope2 in range(self.config.isotope_error + 1):
            subtree = self.kdtree(batch, isotope2)
            subtrees.append(subtree)

        for tree in self.trees:
            for subtree in subtrees:
                neighbors.append(subtree.query_ball_tree(tree, r=radius))

        for x, zindices in enumerate(zip(*neighbors)):
            matches = []
            scores = []
            j = x + offset
            if self.config.isotope_error > 0:
                # if isotopes are configured, we need to deduplicate the indices from different trees
                indices = set()
                for ix in zindices:
                    indices.update(ix)
            else:
                indices = itertools.chain.from_iterable(zindices)
            for i in indices:
                if i < j and j >= self.previous_end and self.within_tolerance(i, j):
                    score = self.score_pair(i, j)
                    if score >= self.config.score_threshold:
                        matches.append(i)
                        scores.append(score)
            if matches:
                yield (j, matches, scores)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.within_tolerance = self.tolerance_check()
        self.within_mz_tolerance = self.mz_tolerance_check()
        if self.config.fragment_mz_unit == MzErrorUnit.PPM:
            self.rtol = self.config.fragment_mz_tolerance / 1e6
            self.atol = 0.0
        else:
            self.rtol = 0.0
            self.atol = self.config.fragment_mz_tolerance

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
            batch = self.task_queue.get()
            if batch is None:
                logger.debug(
                    "Worker with PID %d received None, wrapping up...", self.pid
                )
                break
            for result in self.process_batch(batch):
                self.result_queue.put(self.encode_result(*result))
            logger.debug(
                "Finished batch %d of %d in worker %d. Current output queue size: %d",
                batch + 1,
                self.nbatches,
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

    def kdtree(
        self, experiment: "Experiment", factors: np.ndarray, isotope: int = 0
    ) -> cKDTree:
        """Build a cKDTree from the peptide DataFrame, applying the scaling factors to each dimension."""
        df = experiment.peptides
        names = ["m/z", "irt"]
        if experiment.config.model_ccs is not None:
            names.append("ccs")
        if isotope:
            logger.debug("Applying isotope error of %d to m/z values", isotope)
            values = df[names].values.astype(np.float32, copy=True)
            values[:, 0] += (
                isotope
                * PROTON_MASS
                / df["precursor_charges"].values.astype(np.float32)
            )
        else:
            # avoid copying if no isotope error is applied
            values = df[names].values
        return cKDTree(values * factors)

    def scaling_factors(self, experiment: "Experiment") -> np.ndarray:
        """Calculate scaling factors for each dimension based on the configured tolerances.
        With these factors, all tolerances will be scaled to the m/z tolerance."""
        max_mz = float(experiment.peptides["m/z"].max())
        mz_tol = experiment.config.absolute_mz_error(max_mz)
        irt_tol = experiment.config.irt_tolerance
        if experiment.config.model_ccs is not None:
            ccs_rtol = experiment.config.ccs_rtolerance
            ccs_tol = ccs_rtol * experiment.peptides["ccs"].max()
        else:
            ccs_tol = None
        factors = [1.0, mz_tol / irt_tol]
        if ccs_tol is not None:
            factors.append(mz_tol / ccs_tol)
        out = np.array(factors, dtype=np.float32)
        logger.debug(
            "Calculated scaling factors: %s (m/z tol: %f, iRT tol: %f, CCS tol: %s)",
            out,
            mz_tol,
            irt_tol,
            f"{ccs_tol:.2f}" if ccs_tol is not None else "N/A",
        )
        return out

    def nbatches(self, experiment: "Experiment") -> int:
        return math.ceil(len(experiment.peptides) / experiment.config.batch_size)

    def evaluate(self, experiment: "Experiment") -> np.ndarray:
        factors = self.scaling_factors(experiment)
        trees = [
            self.kdtree(experiment, factors, isotope=i)
            for i in range(experiment.config.isotope_error + 1)
        ]
        logger.info(
            "Built %d cKDTree(s) with %s nodes from %d points",
            len(trees),
            ", ".join(str(tree.size) for tree in trees),
            trees[0].n,
        )
        nb = self.nbatches(experiment)
        logger.info("Processing %d spectra in %d batches...", trees[0].n, nb)
        dtype = np.dtype([("i", np.int32), ("j", np.int32), ("score", np.float32)])
        # add global offset to account for the entire peptide dataframe being a subset
        global_offset = experiment.peptides.index[0]
        if experiment.config.subset > 1:
            previous_end = (
                experiment.offsets[experiment.config.subset - 2][1] - global_offset
            )
            logger.debug("End of previous subset: %d", previous_end)
        else:
            previous_end = 0
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
                    shared_memory=MzIrtDataFrame._shared_memory[experiment],
                    nbatches=nb,
                    shape=(
                        len(experiment.peptides),
                        3 if experiment.config.model_ccs is not None else 2,
                    ),
                    seq_dtype=experiment.peptides["peptide_sequences"].dtype,
                    trees=trees,
                    spectra=experiment.predicted_spectra,
                    scaling_factors=factors,
                    previous_end=previous_end,
                )
                for _ in range(experiment.config.workers)
            ]
            for worker in workers:
                worker.start()

            for batch in range(nb):
                in_queue.put(batch)

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
                            yield (i + global_offset, m + global_offset, s)
                        if count % experiment.config.batch_size == 0:
                            logger.debug("Processed %d peptides...", count)

            scores = np.fromiter(produce_results(), dtype=dtype)
            for worker in workers:
                worker.join()
        else:
            pseudoworker = GroupingWorker(
                None,
                None,
                nbatches=nb,
                config=experiment.config,
                trees=trees,
                spectra=experiment.predicted_spectra,
                scaling_factors=factors,
                previous_end=previous_end,
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
                for batch in range(nb):
                    for item in pseudoworker.process_batch(batch):
                        i, matches, scores = item
                        for m, s in zip(matches, scores):
                            yield (i + global_offset, m + global_offset, s)

            scores = np.fromiter(produce_results(), dtype=dtype)
        logger.info(
            "Finished scoring, found %d pairs with score above %f",
            len(scores),
            experiment.config.score_threshold,
        )
        logger.debug(
            "Indices of peptides: %s .. %s",
            experiment.peptides.index[:5],
            experiment.peptides.index[-5:],
        )
        logger.debug("Sample of scored pairs: %s .. %s", scores[:5], scores[-5:])
        return scores
