import math
from typing import Iterable, TYPE_CHECKING, cast
from scipy.spatial import KDTree
import numpy as np
import logging
import multiprocessing as mp
import itertools
from .utils.abc import Fixture
from .utils.utils import ExperimentWorker
from .utils.config import PROTON_MASS, MzErrorUnit
from ._match_peaks import match_peaks_sorted, similarity_score as c_similarity_score

if TYPE_CHECKING:
    import pandas as pd
    from .experiment import (
        Experiment,
        SingleInputExperiment,
        DualInputExperiment,
        Config,
    )
    from .utils.abc import SpectrumCollection
    from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger(__name__)


class GroupingWorker(ExperimentWorker):
    """Worker process that receives batch numbers and performs subtree creation, neighbor search, and tolerance checking for a batch of spectra.
    It then scores the valid pairs and sends the results back to the main process if they pass the configured score threshold.
    """

    config: "Config"
    trees: list[KDTree]
    radius: float
    shape_1: tuple[int, ...]
    shape_2: tuple[int, ...]
    nbatches: int
    seq_dtype_1: np.dtype
    seq_dtype_2: np.dtype
    shared_memory_1: dict[str, "SharedMemory"]
    shared_memory_2: dict[str, "SharedMemory"]
    spectra_1: "SpectrumCollection"
    spectra_2: "SpectrumCollection"
    peptides_1: np.ndarray
    peptides_2: np.ndarray
    charges_1: np.ndarray
    charges_2: np.ndarray
    scaling_factors: np.ndarray
    mzrt_1: np.ndarray
    mzrt_2: np.ndarray
    previous_end: int

    def within_mz_tolerance_no_isotope(self, i: int, j: int) -> bool:
        return self.config.within_mz_tolerance(self.mzrt_1[i, 0], self.mzrt_2[j, 0])

    def within_mz_tolerance_equal_charge(self, i: int, j: int) -> bool:
        # For equal charge, only isotope-difference matters: (iso1 - iso2).
        # This avoids redundant checks like (1, 1), (2, 2), ... while still
        # covering both orderings (iso1 > iso2 and iso1 < iso2).
        charge = self.charges_1[i]
        shift = PROTON_MASS / charge
        return any(
            self.config.within_mz_tolerance(
                self.mzrt_1[i, 0] + delta_isotope * shift, self.mzrt_2[j, 0]
            )
            for delta_isotope in range(
                -self.config.isotope_error, self.config.isotope_error + 1
            )
        )

    def within_mz_tolerance_different_charge(self, i: int, j: int) -> bool:
        return any(
            self.config.within_mz_tolerance(
                self.mzrt_1[i, 0] + isotope1 * PROTON_MASS / self.charges_1[i],
                self.mzrt_2[j, 0] + isotope2 * PROTON_MASS / self.charges_2[j],
            )
            for isotope1, isotope2 in itertools.product(
                range(self.config.isotope_error + 1), repeat=2
            )
        )  # (0, 0) is included, so this also covers the case where no isotope error is applied

    def within_mz_tolerance_with_isotopes(self, i: int, j: int) -> bool:
        if self.charges_1[i] == self.charges_2[j]:
            return self.within_mz_tolerance_equal_charge(i, j)
        else:
            return self.within_mz_tolerance_different_charge(i, j)

    def within_tolerance_2d(self, i: int, j: int) -> bool:
        irt_tol = self.config.irt_tolerance
        return (
            self.within_mz_tolerance(i, j)
            and abs(self.mzrt_1[i, 1] - self.mzrt_2[j, 1]) <= irt_tol
        )

    def within_tolerance_3d(self, i: int, j: int) -> bool:
        irt_tol = self.config.irt_tolerance
        ccs_rtol = self.config.ccs_rtolerance
        return (
            self.within_mz_tolerance(i, j)
            and abs(self.mzrt_1[i, 1] - self.mzrt_2[j, 1]) <= irt_tol
            and abs(self.mzrt_1[i, 2] - self.mzrt_2[j, 2])
            <= ccs_rtol * max(self.mzrt_1[i, 2], self.mzrt_2[j, 2])
        )

    def tolerance_check(self):
        if self.config.model_ccs is not None:
            return self.within_tolerance_3d
        return self.within_tolerance_2d

    def mz_tolerance_check(self):
        if self.config.isotope_error == 0:
            return self.within_mz_tolerance_no_isotope
        return self.within_mz_tolerance_with_isotopes

    def kdtree(self, batch: int, isotope: int) -> KDTree:
        bsize = self.config.batch_size
        arr = self.mzrt_1[batch * bsize : (batch + 1) * bsize]
        if isotope:
            arr = arr.copy()
            arr[:, 0] += (
                isotope
                * PROTON_MASS
                / self.charges_1[batch * bsize : (batch + 1) * bsize]
            )
        return KDTree(arr * self.scaling_factors)

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
        mz1, intensities1 = self.spectra_1[i]
        mz2, intensities2 = self.spectra_2[j]
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

    def process_pair(
        self, i: int, j: int, matches: list[int], scores: list[float]
    ) -> None:
        if (
            self.dual_mode or (i < j and j >= self.previous_end)
        ) and self.within_tolerance(i, j):
            score = self.score_pair(i, j)
            if score >= self.config.score_threshold:
                matches.append(j)
                scores.append(score)

    def process_batch(
        self,
        batch: int,
    ) -> Iterable[tuple[int, list[int], list[float]]]:

        logger.info("Processing batch %d of %d...", batch + 1, self.nbatches)

        offset = batch * self.config.batch_size
        # format PID as str because it can be None if called from a non-multiprocessing context
        logger.debug("Batch idx %d, offset %d, PID %s", batch, offset, self.pid)

        bsize = self.config.batch_size
        last_idx = min((batch + 1) * bsize, self.mzrt_1.shape[0]) - 1
        max_batch_mz = self.mzrt_1[last_idx, 0]
        batch_mz_tol = self.config.absolute_mz_error(max_batch_mz)
        radius = batch_mz_tol * np.sqrt(self.mzrt_1.shape[1])
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
            i = x + offset
            if self.config.isotope_error > 0:
                # Deduplicate candidate indices while iterating over tree pairs.
                seen = set()
                for ix in zindices:
                    for j in ix:
                        if j in seen:
                            continue
                        seen.add(j)
                        self.process_pair(i, j, matches, scores)
            else:
                indices = itertools.chain.from_iterable(zindices)
                for j in indices:
                    self.process_pair(i, j, matches, scores)

            if matches:
                yield (i, matches, scores)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dual_mode = self.spectra_1 is not self.spectra_2
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
        self.mzrt_1 = np.ndarray(
            shape=self.shape_1,
            dtype=np.float32,
            buffer=self.shared_memory_1["mzrt"].buf,
        )
        self.peptides_1 = np.ndarray(
            shape=(self.shape_1[0],),
            dtype=self.seq_dtype_1,
            buffer=self.shared_memory_1["peptide_sequences"].buf,
        )
        self.charges_1 = np.ndarray(
            shape=(self.shape_1[0],),
            dtype=np.uint8,
            buffer=self.shared_memory_1["precursor_charges"].buf,
        )
        if not self.dual_mode:
            self.mzrt_2 = self.mzrt_1
            self.peptides_2 = self.peptides_1
            self.charges_2 = self.charges_1
        else:
            self.mzrt_2 = np.ndarray(
                shape=self.shape_2,
                dtype=np.float32,
                buffer=self.shared_memory_2["mzrt"].buf,
            )
            self.peptides_2 = np.ndarray(
                shape=(self.shape_2[0],),
                dtype=self.seq_dtype_2,
                buffer=self.shared_memory_2["peptide_sequences"].buf,
            )
            self.charges_2 = np.ndarray(
                shape=(self.shape_2[0],),
                dtype=np.uint8,
                buffer=self.shared_memory_2["precursor_charges"].buf,
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
        for shm in self.shared_memory_1.values():
            shm.close()
        if self.shared_memory_2 is not self.shared_memory_1:
            for shm in self.shared_memory_2.values():
                shm.close()
        self.spectra_1.worker_close()
        if self.spectra_2 is not self.spectra_1:
            self.spectra_2.worker_close()
        logger.debug("Worker with PID %d finished", self.pid)


class SpectrumGrouping(Fixture):
    max_queue_size: int = 100000
    dtype = np.dtype([("i", np.int32), ("j", np.int32), ("score", np.float32)])

    def assign_inputs(
        self, experiment: "Experiment"
    ) -> tuple["pd.DataFrame", "pd.DataFrame"]:
        if hasattr(experiment, "peptides_1"):
            logger.info("Performing grouping in dual input mode")
            e = cast("DualInputExperiment", experiment)
            return e.peptides_1, e.peptides_2
        else:
            logger.info("Performing grouping in single input mode")
            e = cast("SingleInputExperiment", experiment)
            return e.peptides, e.peptides

    def kdtree(
        self, experiment: "Experiment", factors: np.ndarray, isotope: int = 0
    ) -> KDTree:
        """Build a KDTree from the (second) peptide DataFrame, applying the scaling factors to each dimension."""
        _, df = self.assign_inputs(experiment)
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
        return KDTree(values * factors)

    def scaling_factors(self, experiment: "Experiment") -> np.ndarray:
        """Calculate scaling factors for each dimension based on the configured tolerances.
        With these factors, all tolerances will be scaled to the m/z tolerance."""
        peptides_1, peptides_2 = self.assign_inputs(experiment)
        min_mz = max(peptides_1["m/z"].iloc[0], peptides_2["m/z"].iloc[0])
        min_mz -= experiment.config.absolute_mz_error(min_mz)
        mz_tol = experiment.config.absolute_mz_error(min_mz)
        irt_tol = experiment.config.irt_tolerance
        if experiment.config.model_ccs is not None:
            ccs_rtol = experiment.config.ccs_rtolerance
            ccs_tol = ccs_rtol * max(peptides_1["ccs"].max(), peptides_2["ccs"].max())
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
        peptides, _ = self.assign_inputs(experiment)
        return math.ceil(len(peptides) / experiment.config.batch_size)

    def evaluate(self, experiment: "Experiment") -> np.ndarray:
        factors = self.scaling_factors(experiment)
        trees = [
            self.kdtree(experiment, factors, isotope=i)
            for i in range(experiment.config.isotope_error + 1)
        ]
        logger.info(
            "Built %d KDTree(s) with %s nodes from %d points",
            len(trees),
            ", ".join(str(tree.size) for tree in trees),
            trees[0].n,
        )
        nb = self.nbatches(experiment)
        logger.info("Processing %d spectra in %d batches...", trees[0].n, nb)

        peptides_1, peptides_2 = self.assign_inputs(experiment)
        # add global offset to account for the entire peptide dataframe being a subset
        global_offset = peptides_1.index[0]
        if experiment.config.subset > 1:
            previous_end = (
                experiment.offsets[experiment.config.subset - 2][1] - global_offset
            )
            logger.debug("End of previous subset: %d", previous_end)
        else:
            previous_end = 0
        if peptides_1 is peptides_2:
            exp = cast("SingleInputExperiment", experiment)
            logger.debug("Grouping in single input mode.")
            shared_memory_1 = shared_memory_2 = exp.__class__.peptides._shared_memory[
                exp
            ]
            shape_1 = shape_2 = (
                len(peptides_1),
                3 if exp.config.model_ccs is not None else 2,
            )
            seq_dtype_1 = seq_dtype_2 = peptides_1["peptide_sequences"].dtype
            spectra_1 = spectra_2 = exp.predicted_spectra
        else:
            exp = cast("DualInputExperiment", experiment)
            logger.debug("Grouping in dual input mode.")
            shared_memory_1 = exp.__class__.peptides_1._shared_memory[exp]
            shared_memory_2 = exp.__class__.peptides_2._shared_memory[exp]
            shape_1 = (
                len(peptides_1),
                3 if exp.config.model_ccs is not None else 2,
            )
            shape_2 = (
                len(peptides_2),
                3 if exp.config.model_ccs is not None else 2,
            )
            seq_dtype_1 = peptides_1["peptide_sequences"].dtype
            seq_dtype_2 = peptides_2["peptide_sequences"].dtype
            spectra_1 = exp.predicted_spectra_1
            spectra_2 = exp.predicted_spectra_2

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
                    shared_memory_1=shared_memory_1,
                    shared_memory_2=shared_memory_2,
                    nbatches=nb,
                    shape_1=shape_1,
                    shape_2=shape_2,
                    seq_dtype_1=seq_dtype_1,
                    seq_dtype_2=seq_dtype_2,
                    trees=trees,
                    spectra_1=spectra_1,
                    spectra_2=spectra_2,
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

            scores = np.fromiter(produce_results(), dtype=self.dtype)
            for worker in workers:
                worker.join()
        else:
            pseudoworker = GroupingWorker(
                None,
                None,
                nbatches=nb,
                config=experiment.config,
                trees=trees,
                spectra_1=spectra_1,
                spectra_2=spectra_2,
                scaling_factors=factors,
                previous_end=previous_end,
            )

            pseudoworker.mzrt_1 = np.ndarray(
                shape=shape_1,
                dtype=np.float32,
                buffer=shared_memory_1["mzrt"].buf,
            )
            pseudoworker.mzrt_2 = np.ndarray(
                shape=shape_2,
                dtype=np.float32,
                buffer=shared_memory_2["mzrt"].buf,
            )
            pseudoworker.peptides_1 = peptides_1["peptide_sequences"].values
            pseudoworker.peptides_2 = peptides_2["peptide_sequences"].values
            pseudoworker.charges_1 = peptides_1["precursor_charges"].values
            pseudoworker.charges_2 = peptides_2["precursor_charges"].values

            def produce_results():
                for batch in range(nb):
                    for item in pseudoworker.process_batch(batch):
                        i, matches, scores = item
                        for m, s in zip(matches, scores):
                            yield (i + global_offset, m + global_offset, s)

            scores = np.fromiter(produce_results(), dtype=self.dtype)
        logger.info(
            "Finished scoring, found %d pairs with score above %f",
            len(scores),
            experiment.config.score_threshold,
        )
        logger.debug(
            "Indices of peptides: %s .. %s and %s .. %s",
            peptides_1.index[:5],
            peptides_1.index[-5:],
            peptides_2.index[:5],
            peptides_2.index[-5:],
        )
        logger.debug("Sample of scored pairs: %s .. %s", scores[:5], scores[-5:])
        return scores
