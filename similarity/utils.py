from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
import argparse
import multiprocessing as mp
import numpy as np
import threading
import queue
import diskcache
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, Iterable
from types import UnionType
import logging

from numpy import ndarray
from pandas import DataFrame, Series

if TYPE_CHECKING:
    from .experiment import Experiment
    import pandas as pd

logger = logging.getLogger(__name__)


class Fixture(ABC):
    """A descriptor for a calculation result. The result is calculated lazily on first access and then cached for subsequent accesses."""

    _data: dict["Experiment", Any] = {}

    @abstractmethod
    def evaluate(self, experiment: "Experiment") -> Any:
        pass

    def __init__(self):
        super().__init__()
        self._data = {}

    def __get__(self, obj, objtype=None):
        logger.debug(
            "Accessing %s fixture %s on obj %s. Keys cached are currently:\n%s",
            self.__class__.__name__,
            self,
            obj,
            self._data.keys(),
        )
        if obj not in self._data:
            logger.debug("%s not in cache on %s, evaluating...", obj, self)
            self._data[obj] = self.evaluate(obj)
            logger.info(
                "Finished evaluating %s for %s %d",
                self.__class__.__name__,
                obj.__class__.__name__,
                id(obj),
            )
        else:
            logger.debug("%s already in cache: (type %s)", self, type(self._data[obj]))
        return self._data[obj]

    def __set__(self, obj, value):
        raise AttributeError(f"Cannot set value of fixture {self.__class__.__name__}")


@dataclass(frozen=True, slots=True)
class BaseConfig:
    @staticmethod
    def get_type(ftype):
        if isinstance(ftype, UnionType):
            return ftype.__args__[0]
        return ftype

    @staticmethod
    def get_required(field):
        if isinstance(field.type, UnionType):
            return field.default is None and type(None) not in field.type.__args__
        return field.default is None

    @classmethod
    def argparser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Experiment configuration")
        for field in fields(cls):
            kw = dict(default=field.default, required=cls.get_required(field))
            if field.type is bool:
                # for bools, use action='store_true' and default to False
                kw["action"] = "store_true"
            else:
                kw["type"] = cls.get_type(field.type)
            parser.add_argument(f"--{field.name.replace('_', '-')}", **kw)
        return parser


@dataclass(frozen=True, slots=True)
class Config(BaseConfig):
    input_file: Path
    collision_energy: int = 30
    fragmentation_type: str | None = None
    min_charge: int = 2
    max_charge: int = 2
    model_intensity: str = "Prosit_2020_intensity_HCD"
    model_irt: str = "Prosit_2019_irt"
    model_ccs: str | None = None
    mz_tolerance: float = 1.0
    irt_tolerance: float = 5.0
    peak_tolerance: float = 0.0
    peak_ppm: float = 10.0
    ccs_rtolerance: float = 0.02
    nonstandard_aminoacids: bool = False
    ptms: bool = False
    koina_host: str = "koina.wilhelmlab.org:443"
    cache_dir: Path = Path(".")
    cache_properties: bool = False
    workers: int = mp.cpu_count()
    batch_size: int = 100000
    score_threshold: float = 0.0


class BaseIndex(ABC):
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
    def save_predictions(
        self,
        inputs: "pd.DataFrame",
        predictions: dict[str, np.ndarray],
    ) -> None:
        """Asyncronously save predictions to cache."""
        pass

    @abstractmethod
    def wait(self):
        """Wait for all pending saves to finish."""
        pass

    @abstractmethod
    def fill_from_cache(self, inputs: "pd.DataFrame") -> None:
        """Fill missing values in inputs from cache."""
        pass


class Index(diskcache.Index, ABC):
    """Index for predicted spectra. Uses experiment config to add collision energy, charge and model info to the key."""

    _saving_thread: threading.Thread
    _save_queue: queue.Queue
    _done = threading.Event()
    name: str

    @abstractmethod
    def _full_key(self, key: Any) -> bytes:
        pass

    def __new__(cls, experiment: "Experiment", *args, **kwargs):
        assert (
            experiment._cache is not None
        ), "Experiment cache must be initialized before creating Index"
        instance = super().__new__(cls)
        instance._cache = experiment._cache
        return instance

    def __init__(self, experiment: "Experiment"):
        self.experiment = experiment
        self._save_queue = queue.Queue()
        self._saving_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._saving_thread.start()
        # not calling super().__init__() because it would reassign self._cache

    def __getitem__(self, key: Any) -> Any:
        full_key = self._full_key(key)
        return super().__getitem__(full_key)

    def __setitem__(self, key: Any, value: Any) -> None:
        full_key = self._full_key(key)
        super().__setitem__(full_key, value)

    def __contains__(self, key: Any) -> bool:
        full_key = self._full_key(key)
        return full_key in self.cache  # direct check on Index doesn't work

    @abstractmethod
    def _key_from_row(self, row: "pd.Series") -> Any:
        pass

    def _preprocess_predictions(self, predictions: dict[str, np.ndarray]) -> Iterable:
        """
        Preprocess raw predictions from the model into the format expected by _write_to_cache.
        Should return an iterable of values to be cached, the same size as `inputs`.
        """
        return predictions[self.name].reshape(
            -1
        )  # default implementation for 1D predictions

    def _write_to_cache(self, inputs: "pd.DataFrame", predictions: Iterable) -> None:
        for (_, row), value in zip(inputs.iterrows(), predictions):
            key = self._key_from_row(row)
            self[key] = value

    def _save_worker(self):
        while not self._done.is_set():
            inputs, predictions = self._save_queue.get()
            data = self._preprocess_predictions(predictions)
            with self.transact():
                self._write_to_cache(inputs, data)
            logger.info(
                "Saved %d %s predictions to cache",
                len(predictions[self.name]),
                self.name,
            )
            self._save_queue.task_done()

    def save_predictions(
        self,
        inputs: "pd.DataFrame",
        predictions: dict[str, np.ndarray],
    ) -> None:
        logger.info(
            "Queueing %d %s predictions for saving to cache",
            len(predictions[self.name]),
            self.name,
        )
        self._save_queue.put((inputs, predictions))

    def wait(self):
        self._save_queue.join()
        self._saving_thread.join()
        logger.info("All pending %s predictions have been saved to cache", self.name)

    def fill_from_cache(self, inputs: "pd.DataFrame") -> None:
        inputs[self.name] = inputs.apply(
            lambda row: self.get(self._key_from_row(row), np.nan), axis=1
        )


class SpectrumIndex(Index):
    name = "intensity"  # not used as a key, but for logging and debugging

    def _full_key(self, key: tuple[str, int]) -> bytes:
        config = self.experiment.config
        return bytes(
            f"{key[0]}_{config.collision_energy}_{key[1]}_{config.model_intensity}",
            "ascii",
        )

    def _key_from_row(self, row: "pd.Series") -> Any:
        return row["peptide_sequences"], row["precursor_charges"]

    def _preprocess_predictions(
        self, predictions: dict[str, np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        processed = []
        for mz, intensities in zip(predictions["mz"], predictions["intensities"]):
            idx = mz > 0
            processed.append(
                (
                    mz[idx],
                    intensities[idx],
                )
            )
        return processed


class RTIndex(Index):
    name = "irt"

    def _key_from_row(self, row: "pd.Series") -> str:
        return row["peptide_sequences"]

    def _full_key(self, key: str) -> bytes:
        config = self.experiment.config
        return bytes(f"{key}_{config.model_irt}", "ascii")


class IMIndex(Index):
    name = "ccs"

    def _key_from_row(self, row: "pd.Series") -> tuple[str, int]:
        return row["peptide_sequences"], row["charge"]

    def _full_key(self, key: tuple[str, int]) -> bytes:
        config = self.experiment.config
        return bytes(f"{key[0]}_{key[1]}_{config.model_ccs}", "ascii")


class ExperimentWorker(ABC, mp.Process):
    def __init__(self, task_queue: mp.Queue, result_queue: mp.Queue, **kwargs):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def run(self) -> None:
        return super().run()
