from typing import TYPE_CHECKING, Generic
import logging
import numpy as np
from pandas import DataFrame
from multiprocessing.shared_memory import SharedMemory
from ..abc import SpectrumCollection

if TYPE_CHECKING:
    from ..abc import Index
    from ...experiment import Experiment
    from numpy.typing import NDArray, DTypeLike
    import numpy as np

logger = logging.getLogger(__name__)


class SharedArraySpectrumCollection(SpectrumCollection):
    """A SpectrumCollection that keeps predicted spectra in a 3D array in shared memory."""

    shared_memory: "SharedMemory"
    sm_name: str
    dtype: "DTypeLike" = np.float32
    array: "NDArray[np.float32]"
    shape: tuple[int, int, int]

    def __init__(self, experiment: "Experiment"):
        super().__init__(experiment)
        self.shape = (experiment.peptides.shape[0], 2, experiment.config.max_peaks)
        size = (
            self.shape[0]
            * self.shape[1]
            * self.shape[2]
            * np.dtype(self.dtype).itemsize
        )
        logger.info(
            "Allocating %.2f GB of shared memory for predicted spectra", size / 2**30
        )
        self.shared_memory = SharedMemory(
            create=True,
            name=f"SharedArraySpectrumCollection-Experiment-{id(experiment)}",
            size=size,
        )
        self.sm_name = self.shared_memory.name
        self.array = np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self.shared_memory.buf,
        )
        self.array.fill(np.nan)

    def __getitem__(
        self, key: int
    ) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
        mz, intensities = self.array[key, 0], self.array[key, 1]
        idx = intensities > 0
        return mz[idx], intensities[idx]

    def __getstate__(self):
        return {"shape": self.shape, "sm_name": self.sm_name}

    def __setstate__(self, state):
        self.shape = state["shape"]
        self.sm_name = state["sm_name"]
        self.shared_memory = SharedMemory(name=self.sm_name)
        self.array = np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self.shared_memory.buf,
        )

    def fill_from_cache(self, experiment: "Experiment", index: "Index") -> None:
        maxpeaks = experiment.config.max_peaks
        for i, (pep, charge) in experiment.peptides[
            ["peptide_sequences", "precursor_charges"]
        ].itertuples():
            key = (pep, charge)
            mz, intensities = index.get(key, (None, None))
            if mz is not None and intensities is not None:
                len_spectrum = min(len(mz), maxpeaks)
                if len_spectrum < len(mz):
                    idx = np.argsort(intensities)[-maxpeaks:]
                    mz = mz[idx]
                    intensities = intensities[idx]
                self.array[i, 0, :len_spectrum] = mz
                self.array[i, 1, :len_spectrum] = intensities

    def fill_from_predictions(
        self, inputs: DataFrame, predictions: dict[str, list["np.ndarray"]]
    ) -> None:
        maxpeaks = self.experiment.config.max_peaks
        for i in range(len(inputs)):
            mz = predictions["mz"][i]
            intensities = predictions["intensities"][i]
            len_spectrum = min(len(mz), maxpeaks)
            if len_spectrum < len(mz):
                idx = np.argsort(intensities)[-maxpeaks:]
                mz = mz[idx]
                intensities = intensities[idx]
            self.array[i, 0, :len_spectrum] = mz
            self.array[i, 1, :len_spectrum] = intensities

    @property
    def spectra_available(self) -> "NDArray[np.bool_]":
        available = np.zeros(len(self.experiment.peptides), dtype=np.bool_)
        for i in range(len(available)):
            available[i] = not np.isnan(self.array[i, 0, 0])
        logger.info(
            "%d of %d spectra are available in cache", available.sum(), len(available)
        )
        return available

    def close(self):
        if hasattr(self, "shared_memory"):
            self.shared_memory.close()
            self.shared_memory.unlink()
            del self.shared_memory

    def worker_close(self):
        self.shared_memory.close()

    def is_ready(self) -> bool:
        return True
