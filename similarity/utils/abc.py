from abc import ABC, abstractmethod
import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..experiment import Experiment
    from .cache.common import SpectrumCache
    import pandas as pd
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Fixture(ABC):
    """A descriptor for a calculation result. The result is calculated lazily on first access and then cached for subsequent accesses."""

    _data: dict["Experiment", Any] = {}

    @abstractmethod
    def evaluate(self, experiment: "Experiment") -> Any:
        pass

    def __init_subclass__(cls) -> None:
        cls._data = {}
        super().__init_subclass__()

    def __get__(self, obj, objtype=None):
        if obj not in self._data:
            logger.info(
                "Started evaluating %s for %s %d",
                self.__class__.__name__,
                obj.__class__.__name__,
                id(obj),
            )
            start_time = datetime.now()
            self._data[obj] = self.evaluate(obj)
            end_time = datetime.now()

            logger.info(
                "Finished evaluating %s for %s %d in %s",
                self.__class__.__name__,
                obj.__class__.__name__,
                id(obj),
                end_time - start_time,
            )
        return self._data[obj]

    def __set__(self, obj, value):
        raise AttributeError(f"Cannot set value of fixture {self.__class__.__name__}")

    @classmethod
    def exists(cls, obj) -> bool:
        return obj in cls._data


class IndexType(Enum):
    INTENSITY = "intensity"
    IRT = "irt"
    CCS = "ccs"


class Cache(ABC):
    name: str
    index_type: dict[IndexType, type["Cache"]]

    def __init__(self, experiment: "Experiment"):
        self.experiment = experiment

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        pass

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> None:
        pass

    @abstractmethod
    def __contains__(self, key: Any) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def save_predictions(
        self,
        inputs: "pd.DataFrame",
        predictions: dict[str, list["np.ndarray"]],
    ) -> None:
        """Asyncronously save predictions to cache."""
        pass

    @abstractmethod
    def wait(self):
        """Wait for all pending saves to finish."""
        pass

    def finalize(self):
        """Signals that no further saves will be called."""
        pass

    @abstractmethod
    def fill_from_cache(self, inputs: "pd.DataFrame", output: "np.ndarray") -> None:
        """Fill missing values in inputs from cache."""
        pass

    @abstractmethod
    def get(self, key: Any, default: Any = None) -> Any:
        pass

    @classmethod
    def get_index(
        cls, index_type: IndexType, experiment: "Experiment"
    ) -> "Cache | None":
        if (
            index_type in {IndexType.IRT, IndexType.CCS}
            and not experiment.config.cache_conf.cache_properties
        ):
            return None
        index_cls = cls.index_type.get(index_type)
        if index_cls is None:
            raise ValueError(f"No cache configured for index type {index_type}")
        return index_cls(experiment)

    @abstractmethod
    def close(self):
        pass


class SpectrumCollection(ABC):
    """A container of spectra for use in grouping and scoring. Needs to support returning spectra by integer peptide index and support multiprocessing."""

    def __init__(self, experiment: "Experiment"):
        self.experiment = experiment

    @abstractmethod
    def __getitem__(
        self, i: int
    ) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
        pass

    @abstractmethod
    def fill_from_cache(self, experiment: "Experiment", index: "SpectrumCache") -> None:
        """Fill missing spectra from cache."""
        pass

    @abstractmethod
    def fill_from_predictions(
        self, inputs: "pd.DataFrame", predictions: dict[str, list["np.ndarray"]]
    ) -> None:
        """Fill missing spectra from predictions."""
        pass

    @property
    @abstractmethod
    def spectra_available(self) -> "NDArray[np.bool_]":
        """Boolean array indicating which spectra are available in the collection."""
        pass

    def is_ready(self) -> bool:
        return True

    def close(self):
        """Close any resources used by the collection, such as shared memory."""
        pass

    @abstractmethod
    def worker_close(self):
        """Close any resources used by worker processes, such as database connections."""
        pass


class CacheBasedSpectrumCollection(SpectrumCollection):
    """A SpectrumCollection that just retrieves spectra from cache on demand."""

    def __init__(self, experiment: "Experiment", index: "SpectrumCache"):
        super().__init__(experiment)
        self.index = index
