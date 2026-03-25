import threading
import queue
import diskcache
from abc import ABC
from typing import Any, TYPE_CHECKING, Iterable
import logging
import numpy as np
from tqdm import tqdm
from ..abc import IndexType
from .common import (
    ByteStringKeyCache,
    ByteStringSpectrumCache,
    ByteStringRTCache,
    ByteStringCCSCache,
)

if TYPE_CHECKING:
    from ...experiment import Experiment
    import pandas as pd


logger = logging.getLogger(__name__)


class Index(diskcache.Index, ByteStringKeyCache, ABC):
    """Index for predicted spectra. Uses experiment config to add collision energy, charge and model info to the key."""

    _saving_thread: threading.Thread
    writable: bool = False
    _save_queue: queue.Queue
    _done = threading.Event()
    name: str
    _cache_registry: dict["Experiment", diskcache.Cache] = {}

    @staticmethod
    def _get_cache_object(experiment: "Experiment") -> "diskcache.Cache":
        if experiment not in Index._cache_registry:
            Index._cache_registry[experiment] = diskcache.Cache(
                str(experiment.config.cache_conf.cache_dir),
                size_limit=0,
                cull_limit=0,
                eviction_policy="none",
            )
        return Index._cache_registry[experiment]

    def __new__(cls, experiment: "Experiment"):
        cache = cls._get_cache_object(experiment)
        instance = super().__new__(cls)
        instance._cache = cache
        return instance

    def __reduce__(self):
        return self.__class__, (self.experiment,)

    def __init__(self, experiment: "Experiment"):
        ByteStringKeyCache.__init__(self, experiment)
        self._save_queue = queue.Queue()
        self._saving_thread = threading.Thread(
            target=self._save_worker,
            name=f"{self.__class__.__name__}-{self.name}-SavingThread",
        )
        self._done = threading.Event()
        # not starting the thread here because `Index` can also be instantiated by workers
        # not calling super().__init__() because it would reassign self._cache

    def __getitem__(self, key: Any) -> Any:
        full_key = self._full_key(key)
        return super().__getitem__(full_key)

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: Any, value: Any) -> None:
        full_key = self._full_key(key)
        super().__setitem__(full_key, value)

    def __contains__(self, key: Any) -> bool:
        full_key = self._full_key(key)
        return full_key in self.cache  # direct check on Index doesn't work

    def _write_to_cache(self, inputs: "pd.DataFrame", predictions: Iterable) -> None:
        for (_, row), value in zip(inputs.iterrows(), predictions):
            key = self._key_from_row(row)
            self[key] = value

    def _save_worker(self):
        while True:
            try:
                logger.debug("%s saving worker checking the save queue...", self.name)
                inputs, predictions = self._save_queue.get(timeout=1)
                data = self._preprocess_predictions(predictions)
                with self.transact():
                    self._write_to_cache(inputs, data)
                logger.debug(
                    "Saved %d %s predictions to cache",
                    len(next(iter(predictions.values()))),
                    self.name,
                )
                self._save_queue.task_done()
            except queue.Empty:
                if self._done.is_set():
                    break
                else:
                    logger.debug(
                        "Save worker for %s is waiting for new tasks...", self.name
                    )
        logger.info("Saving %s complete", self.name)

    def save_predictions(
        self,
        inputs: "pd.DataFrame",
        predictions: dict[str, list["np.ndarray"]],
    ) -> None:
        logger.debug(
            "Queueing %d %s predictions for saving to cache",
            len(next(iter(predictions.values()))),
            self.name,
        )
        if not self.writable:
            self.writable = True
            self._saving_thread.start()
        self._save_queue.put((inputs, predictions))

    def wait(self):
        if self.writable:
            logger.debug("Flushing %s cache...", self.name)
            self.finalize()
            self._save_queue.join()
            self._saving_thread.join()

    def finalize(self):
        self._done.set()

    def fill_from_cache(self, inputs: "pd.DataFrame", output: "np.ndarray") -> None:
        if len(self._cache) < len(inputs) / 2:
            logger.info(
                "Cache size too small, skipping cache loading for %s", self.name
            )
            return
        for i, (_, row) in tqdm(
            enumerate(inputs.iterrows()),
            total=len(inputs),
            desc=f"Loading {self.name} from cache",
            unit="peptides",
            unit_scale=True,
        ):
            key = self._key_from_row(row)
            value = self.get(key, np.nan)
            output[i] = value

    def close(self):
        self.wait()
        self._cache.close()


class SpectrumIndex(ByteStringSpectrumCache, Index):
    pass


class RTIndex(ByteStringRTCache, Index):
    pass


class IMIndex(ByteStringCCSCache, Index):
    pass


Index.index_type = {
    IndexType.INTENSITY: SpectrumIndex,
    IndexType.IRT: RTIndex,
    IndexType.CCS: IMIndex,
}
