from typing import TYPE_CHECKING
import logging
import numpy as np
from pandas import DataFrame
from multiprocessing.shared_memory import SharedMemory
from ..abc import SpectrumCollection

if TYPE_CHECKING:
    from pathlib import Path
    from ..cache.common import SpectrumCache
    from ...experiment import Experiment
    from numpy.typing import NDArray, DTypeLike
    import numpy as np

logger = logging.getLogger(__name__)


class SharedArraySpectrumCollection(SpectrumCollection):
    """
    A SpectrumCollection that keeps predicted spectra in a 3D array in shared memory.
    """

    shared_memory: "SharedMemory"
    sm_name: str
    dtype: "DTypeLike" = np.float32
    array: "NDArray[np.float32]"
    shape: tuple[int, int, int]

    def __init__(self, experiment: "Experiment"):
        super().__init__(experiment)
        self.shape = (experiment.peptides.shape[0], 2, experiment.config.max_peaks)
        self.offset = experiment.peptides.index[0]
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
        if experiment.spectrum_file:
            self.load_from_file(experiment.spectrum_file)

    def load_from_file(self, file: "str | Path") -> None:
        arr = np.load(file)
        nspectra, narrays, maxpeaks = arr.shape
        if narrays != 2:
            raise ValueError(
                f"Unsupported spectrum array file format. Expected shape {self.shape}, got {arr.shape}"
            )
        if maxpeaks > self.shape[2]:
            raise NotImplementedError(
                f"Saved array has {maxpeaks} peaks, but expected at most {self.shape[2]}. Reduction currently not implemented."
            )
        if maxpeaks < self.shape[2]:
            raise ValueError(
                f"Saved array has {maxpeaks} peaks, but expected at least {self.shape[2]}. Cannot load into larger array."
            )
        if nspectra < self.shape[0]:
            raise ValueError(
                f"Saved array has {nspectra} spectra, but expected at least {self.shape[0]}. Cannot load into larger array."
            )
        if nspectra > self.shape[0]:
            if self.experiment.config.subsets == 1:
                raise ValueError(
                    f"Saved array has {nspectra} spectra, but expected at most {self.shape[0]}. Cannot load into smaller array."
                )
            logger.info("Loading a subset of the spectrum array from file %s", file)
            self.array[:, :, :] = arr[self.offset : self.offset + self.shape[0], :, :]

    def __getitem__(
        self, key: int
    ) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
        mz, intensities = (
            self.array[key, 0],
            self.array[key, 1],
        )
        idx = intensities > 0
        return mz[idx], intensities[idx]

    def __getstate__(self):
        return {"shape": self.shape, "sm_name": self.sm_name, "offset": self.offset}

    def __setstate__(self, state):
        self.shape = state["shape"]
        self.sm_name = state["sm_name"]
        self.offset = state["offset"]
        self.shared_memory = SharedMemory(name=self.sm_name)
        self.array = np.ndarray(
            shape=self.shape,
            dtype=self.dtype,
            buffer=self.shared_memory.buf,
        )

    def fill_from_cache(self, experiment: "Experiment", index: "SpectrumCache") -> None:
        spectra = [[] for _ in range(len(experiment.peptides))]
        index.fill_from_cache(experiment.peptides, spectra)
        for i, cached in enumerate(spectra):
            if cached is not np.nan:
                mz, intensities = cached
                mz, intensities = self._truncate_and_sort_spectrum(mz, intensities)
                len_spectrum = len(mz)
                self.array[i, 0, :len_spectrum] = mz
                self.array[i, 1, :len_spectrum] = intensities

    def fill_from_predictions(
        self, inputs: DataFrame, predictions: dict[str, list["np.ndarray"]]
    ) -> None:
        for iloc, loc in enumerate(inputs.index):
            mz = predictions["mz"][iloc]
            intensities = predictions["intensities"][iloc]
            mz, intensities = self._truncate_and_sort_spectrum(mz, intensities)
            len_spectrum = len(mz)
            self.array[loc - self.offset, 0, :len_spectrum] = mz
            self.array[loc - self.offset, 1, :len_spectrum] = intensities

    def save(self, file: "str | Path") -> None:
        """Save the collection to a file."""
        logger.info("Saving predicted spectra to %s...", file)
        np.save(file, self.array)

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
