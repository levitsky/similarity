from typing import TYPE_CHECKING
import logging
import numpy as np
import pandas as pd

from ..abc import SpectrumCollection, IndexType

if TYPE_CHECKING:
    from ..cache.common import SpectrumCache
    from ...experiment import Experiment
    from numpy.typing import NDArray
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


class CachedSpectrumCollection(SpectrumCollection):
    """A SpectrumCollection that caches predictions to disk using a configurable cache backend."""

    index: "SpectrumCache"
    batch_factor: int = 10

    def __init__(self, experiment: "Experiment", suffix: str = ""):
        super().__init__(experiment, suffix)
        index = experiment.cache[IndexType.INTENSITY]
        if index is None:
            raise ValueError(
                "Cache is not configured, cannot use CachedSpectrumCollection"
            )
        self.index = index
        self._peptides: "pd.DataFrame | None" = None

    @property
    def peptides(self) -> "pd.DataFrame":
        if self._peptides is not None:
            return self._peptides
        return super().peptides

    def set_peptides_from_shared(
        self,
        sequences: "NDArray[np.bytes_]",
        charges: "NDArray[np.uint8]",
        mzrt: "NDArray[np.float32]",
        index_start: int,
    ) -> None:
        """Build the peptide table from shared-memory arrays.

        For use in worker processes, where the peptides fixture cannot be evaluated.
        """
        columns: dict[str, "np.ndarray"] = {
            "peptide_sequences": sequences,
            "precursor_charges": charges,
            "m/z": mzrt[:, 0],
            "irt": mzrt[:, 1],
        }
        if mzrt.shape[1] == 3:
            columns["ccs"] = mzrt[:, 2]
        self._peptides = pd.DataFrame(
            columns,
            copy=False,
            index=range(index_start, index_start + sequences.shape[0]),
        )

    def _index_key(self, key: int) -> tuple[bytes, int]:
        df = self.peptides
        peptide = df.loc[key, "peptide_sequences"]
        charge = df.loc[key, "precursor_charges"]
        return peptide, charge  # type: ignore

    def __getitem__(
        self, key: int
    ) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
        mz, intensities = self.index[self._index_key(key)]
        return self._truncate_and_sort_spectrum(mz, intensities)

    def fill_from_cache(self, index: "SpectrumCache") -> None:
        """Nothing to do because all spectra are loaded on demand from cache."""
        pass

    def fill_from_predictions(
        self, inputs: "pd.DataFrame", predictions: dict[str, list["np.ndarray"]]
    ) -> None:
        """Nothing to do because all spectra are saved to cache anyway before the collection can be used."""
        pass

    @property
    def spectra_available(self) -> "NDArray[np.bool_]":
        available = np.zeros(len(self.peptides), dtype=np.bool_)
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
