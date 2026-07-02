from abc import ABC, abstractmethod
import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING, overload, Self
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
    from ..experiment import Experiment
    from .cache.common import SpectrumCache
    import pandas as pd
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Fixture(ABC):
    """A descriptor for a calculation result. The result is calculated lazily on first access and then cached for subsequent accesses."""

    _data: dict["Experiment", Any]

    @abstractmethod
    def evaluate(self, experiment: "Experiment") -> Any:
        pass

    def __init__(self):
        self._data = {}

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: "Experiment", objtype: type | None) -> Any: ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if obj not in self._data:
            logger.info(
                "Started evaluating %s%s for %s %d",
                self.__class__.__name__,
                self.suffix,
                obj.__class__.__name__,
                id(obj),
            )
            start_time = datetime.now()
            self._data[obj] = self.evaluate(obj)
            end_time = datetime.now()

            logger.info(
                "Finished evaluating %s%s for %s %d in %s",
                self.__class__.__name__,
                self.suffix,
                obj.__class__.__name__,
                id(obj),
                end_time - start_time,
            )
        return self._data[obj]

    def get(self, obj: Any, name: str) -> Any:
        """
        Use another fixture on the same experiment. This automatically adds the suffix to the name of the fixture, if applicable.
        """
        attr = name + self.suffix
        return getattr(obj, attr)

    def __set__(self, obj, value):
        raise AttributeError(f"Cannot set value of fixture {self.__class__.__name__}")

    @staticmethod
    def get_suffix(name: str) -> str:
        if "_" not in name:
            return ""
        right = name.rsplit("_", 1)[1]
        if right.isdigit():
            return "_" + right
        return ""

    @property
    def is_first(self) -> bool:
        return self.suffix in {"", "_1"}

    def __set_name__(self, owner, name):
        self.name = name
        self.suffix = self.get_suffix(name)
        logger.debug(
            "Setting name of fixture %s of %s %d to %s with suffix %s",
            self.__class__.__name__,
            owner.__class__.__name__,
            id(owner),
            self.name,
            self.suffix,
        )

    def exists(self, obj: "Experiment") -> bool:
        return obj in self._data

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


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

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

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

    def __init__(self, experiment: "Experiment", suffix: str = ""):
        self.experiment = experiment
        self.suffix = suffix

    @property
    def peptides(self) -> "pd.DataFrame":
        return getattr(self.experiment, "peptides" + self.suffix)

    @abstractmethod
    def __getitem__(
        self, i: int
    ) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
        pass

    @abstractmethod
    def fill_from_cache(self, index: "SpectrumCache") -> None:
        """Fill missing spectra from cache."""
        pass

    @abstractmethod
    def fill_from_predictions(
        self, inputs: "pd.DataFrame", predictions: dict[str, list["np.ndarray"]]
    ) -> None:
        """Fill missing spectra from predictions."""
        pass

    def save(self, file: "str | Path") -> None:
        """Save the collection to a file."""
        logger.error("%s does not support saving to file", self.__class__.__name__)
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support saving to file"
        )

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

    def _truncate_and_sort_spectrum(
        self, mz: "NDArray[np.float32]", intensities: "NDArray[np.float32]"
    ) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
        maxpeaks = self.experiment.config.max_peaks
        if mz.size > maxpeaks:
            idx = np.argpartition(intensities, -maxpeaks)[-maxpeaks:]
            mz = mz[idx]
            intensities = intensities[idx]
        order = np.argsort(mz, kind="mergesort")
        return mz[order], intensities[order]

    def __str__(self):
        return f"{self.__class__.__name__}{self.suffix} for {self.experiment.__class__.__name__}({id(self.experiment)})"
