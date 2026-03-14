from typing import TYPE_CHECKING
import logging
import numpy as np

from ..abc import SpectrumCollection, IndexType

if TYPE_CHECKING:
    from ..abc import Index
    from ...experiment import Experiment
    from numpy.typing import NDArray
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


class CachedSpectrumCollection(SpectrumCollection):
    """A SpectrumCollection that caches predictions to disk using a configurable cache backend."""

    index: "Index"
    batch_factor: int = 10

    def __init__(self, experiment: "Experiment"):
        super().__init__(experiment)
        index = experiment.config.cache.value.get_index(IndexType.INTENSITY, experiment)
        if index is None:
            raise ValueError(
                "Cache is not configured, cannot use CachedSpectrumCollection"
            )
        self.index = index

    def _index_key(self, key: int) -> tuple[bytes, int]:
        df = self.experiment.peptides
        peptide = df.loc[key, "peptide_sequences"]
        charge = df.loc[key, "precursor_charges"]
        return peptide, charge  # type: ignore

    def __getitem__(
        self, key: int
    ) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
        return self.index[self._index_key(key)]

    def fill_from_cache(self, experiment: "Experiment", index: "Index") -> None:
        """Nothing to do because all spectra are loaded on demand from cache."""
        pass

    def fill_from_predictions(
        self, inputs: "pd.DataFrame", predictions: dict[str, list["np.ndarray"]]
    ) -> None:
        """Nothing to do because all spectra are saved to cache anyway before the collection can be used."""
        pass

    @property
    def spectra_available(self) -> "NDArray[np.bool_]":
        available = np.zeros(len(self.experiment.peptides), dtype=np.bool_)
        if len(self.index) < available.size / 2:
            logger.info(
                "Cache size too small, skipping cache loading for spectra",
            )
            return available
        bsize = self.experiment.config.batch_size * self.batch_factor
        for i in range(len(available)):
            available[i] = self._index_key(i) in self.index
            if i % bsize == 0:
                logger.info(
                    "Checked cache availability for %d of %d spectra",
                    i,
                    available.size,
                )
        logger.info(
            "%d of %d spectra are available in cache", available.sum(), available.size
        )
        return available

    def close(self):
        self.index.close()

    def worker_close(self):
        self.index.close()

    def is_ready(self) -> bool:
        self.index.wait()
        return True
