import math
from typing import Iterable, TYPE_CHECKING
from scipy.spatial import cKDTree
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import logging
import multiprocessing as mp
from .utils import Fixture, ExperimentWorker


if TYPE_CHECKING:
    from .experiment import Experiment, Config
    from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger(__name__)


class GroupingWorker(ExperimentWorker):
    config: "Config"
    shape: tuple[int, ...]
    dtype: np.dtype
    shared_memory: "SharedMemory"

    def within_tolerance_2d(self, i: int, j: int, arr: np.ndarray) -> bool:
        mz_tol = self.config.mz_tolerance
        irt_tol = self.config.irt_tolerance
        return (
            abs(arr[i, 0] - arr[j, 0]) <= mz_tol
            and abs(arr[i, 1] - arr[j, 1]) <= irt_tol
        )

    def within_tolerance_3d(self, i: int, j: int, arr: np.ndarray) -> bool:
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

    def process_task(
        self, task: tuple[int, Iterable[int]], mzrt: np.ndarray
    ) -> tuple[int, list[int]]:
        i, indices = task
        result_indices = []
        for j in indices:
            if self.within_tolerance(i, j, mzrt):
                result_indices.append(j)
        return (i, result_indices)

    def run(self) -> None:
        self.within_tolerance = self.tolerance_check()
        mzrt = np.ndarray(
            shape=self.shape, dtype=self.dtype, buffer=self.shared_memory.buf
        )
        while True:
            chunk = self.task_queue.get()
            if chunk is None:
                break
            result = []
            for i, indices in chunk:
                result.append(self.process_task((i, indices), mzrt))
            self.result_queue.put(result)


class SpectrumGrouping(Fixture):
    chunk_size: int = 1000

    def get_array(self, experiment: "Experiment") -> np.ndarray:
        if experiment.config.model_ccs is not None:
            mzrt = experiment.mz_irt_df[["m/z", "irt", "ccs"]].values
        else:
            mzrt = experiment.mz_irt_df[["m/z", "irt"]].values
        return mzrt

    def kdtree(self, experiment: "Experiment", batch: int | None = None) -> cKDTree:
        df = experiment.mz_irt_df
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
            ccs_tol = ccs_rtol * experiment.mz_irt_df["ccs"].max()
            return np.sqrt(mz_tol**2 + irt_tol**2 + ccs_tol**2)
        return np.sqrt(mz_tol**2 + irt_tol**2)

    def nbatches(self, experiment: "Experiment") -> int:
        return math.ceil(len(experiment.mz_irt_df) / experiment.config.batch_size)

    def process_batch(
        self,
        batch: int,
        tree: cKDTree,
        experiment: "Experiment",
        in_queue: "mp.Queue | None",
        out_queue: "mp.Queue | None",
    ) -> list[tuple[int, list[int]]]:
        """Make a batch of size config.batch_size of the input array and find potential neighbor pairs, submit them to `in_queue`,
        then get the filtered pairs from `out_queue` (in chunks). Also sets the "in pairs" column of the dataframe to True for spectra that are in any pair.
        """

        pairs: list[tuple[int, list[int]]] = []
        nb = self.nbatches(experiment)
        logger.info(
            "Processing batch %d of %d...",
            batch + 1,
            nb,
        )
        subtree = self.kdtree(experiment, batch)

        inpairs = np.zeros(len(experiment.mz_irt_df), dtype=bool)
        offset = batch * experiment.config.batch_size
        logger.debug("Batch idx %d, offset %d", batch, offset)

        radius = self.radius(experiment)
        neighbors = tree.query_ball_tree(subtree, r=radius)
        if in_queue is None or out_queue is None:
            pseudoworker = GroupingWorker(None, None, config=experiment.config)
            pseudoworker.within_tolerance = pseudoworker.tolerance_check()
            mzrt = self.get_array(experiment)
            # Single-threaded mode
            for i, indices in enumerate(neighbors):
                candidates = (
                    j + offset for j in indices if i < j + offset
                )  # Avoid self-pairing and duplicate pairs

                i, valid_indices = pseudoworker.process_task((i, candidates), mzrt)
                if valid_indices:
                    pairs.append((i, valid_indices))
                    inpairs[i] = True
                    for j in valid_indices:
                        inpairs[j] = True
        else:
            count = 0
            chunk = []
            for i, indices in enumerate(neighbors):
                candidates = [
                    j + offset for j in indices if i < j + offset
                ]  # Avoid self-pairing and duplicate pairs
                if candidates:
                    chunk.append((i, candidates))
                    if len(chunk) >= self.chunk_size:
                        in_queue.put(chunk)
                        chunk = []
                        count += 1
            if chunk:
                in_queue.put(chunk)
                count += 1
            for _ in range(count):
                batch_pairs = out_queue.get()
                pairs.extend(batch_pairs)
                for i, indices in batch_pairs:
                    inpairs[i] = True
                    for j in indices:
                        inpairs[j] = True

        experiment.mz_irt_df["in pairs"] |= inpairs
        return pairs

    def evaluate(self, experiment: "Experiment") -> list[tuple[int, list[int]]]:
        df = experiment.mz_irt_df
        df["in pairs"] = False  # Add a column to track if a spectrum is in any pair
        tree = self.kdtree(experiment)
        logger.info("Built cKDTree with %d nodes from %d points", tree.size, tree.n)
        mzrt = self.get_array(experiment)
        nb = self.nbatches(experiment)
        pairs = []
        if experiment.config.workers > 1:
            logger.info(
                "Processing pairs with %d workers...",
                experiment.config.workers,
            )
            in_queue = mp.Queue()
            out_queue = mp.Queue()
            with SharedMemoryManager() as smm:
                shm = smm.SharedMemory(mzrt.nbytes)
                arr = np.ndarray(mzrt.shape, dtype=mzrt.dtype, buffer=shm.buf)
                np.copyto(arr, mzrt)
                workers = [
                    GroupingWorker(
                        in_queue,
                        out_queue,
                        config=experiment.config,
                        shared_memory=shm,
                        shape=mzrt.shape,
                        dtype=mzrt.dtype,
                    )
                    for _ in range(experiment.config.workers)
                ]
                for worker in workers:
                    worker.start()

                batch_index = 0
                while batch_index < nb:
                    batch_pairs = self.process_batch(
                        batch_index, tree, experiment, in_queue, out_queue
                    )
                    pairs.extend(batch_pairs)
                    batch_index += 1

                for _ in range(experiment.config.workers):
                    in_queue.put(None)

                for worker in workers:
                    worker.join()
        else:
            batch_index = 0
            while batch_index < nb:
                batch_pairs = self.process_batch(
                    batch_index, tree, experiment, None, None
                )
                pairs.extend(batch_pairs)
                batch_index += 1

        return pairs
