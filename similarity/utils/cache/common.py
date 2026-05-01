import threading
import queue
import logging
from tqdm import tqdm
from abc import abstractmethod
from typing import Any, Iterable, TYPE_CHECKING
from ..abc import Cache as CacheABC
import numpy as np

if TYPE_CHECKING:
    from ...experiment import Experiment
    import pandas as pd


logger = logging.getLogger(__name__)


class Cache(CacheABC):
    """Common base class for all cache indices."""

    _saving_thread: threading.Thread
    writable: bool = False
    _save_queue: queue.Queue
    _done = threading.Event()
    name: str
    _cache_registry: dict["Experiment", "Cache"] = {}

    def __init__(self, experiment: "Experiment"):
        super().__init__(experiment)
        self._save_queue = queue.Queue()
        self._saving_thread = threading.Thread(
            target=self._save_worker, name=f"{self.name}_saving_thread"
        )

    def _save_worker(self):
        while True:
            try:
                logger.debug("%s saving worker checking the save queue...", self.name)
                inputs, predictions = self._save_queue.get(timeout=1)
                data = self._preprocess_predictions(predictions)
                self.transact_to_cache(inputs, data)
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

    def _preprocess_predictions(self, predictions: dict[str, "np.ndarray"]) -> Iterable:
        """
        Preprocess raw predictions from the model into the format expected by _write_to_cache.
        Should return an iterable of values to be cached, the same size as `inputs`.
        """
        return predictions[self.name].reshape(
            -1
        )  # default implementation for 1D predictions

    @abstractmethod
    def _key_from_row(self, row: "pd.Series") -> Any:
        pass

    def write_to_cache(self, inputs: "pd.DataFrame", predictions: Iterable) -> None:
        for (_, row), value in zip(inputs.iterrows(), predictions):
            key = self._key_from_row(row)
            self[key] = value

    @abstractmethod
    def transact_to_cache(self, inputs: "pd.DataFrame", predictions: Iterable) -> None:
        """Write the given predictions to cache in a transaction. Used by the saving thread to ensure atomicity of multiple writes."""
        pass

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

    def fill_from_cache(
        self, inputs: "pd.DataFrame", output: "np.ndarray | list"
    ) -> None:
        if len(self) < len(inputs) / 2:
            logger.info(
                "Cache size too small, skipping cache loading for %s", self.name
            )
            for i in range(len(inputs)):
                output[i] = np.nan
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

    def wait(self):
        if self.writable:
            logger.debug("Flushing %s cache...", self.name)
            self.finalize()
            self._save_queue.join()
            self._saving_thread.join()

    def finalize(self):
        self._done.set()

    def __reduce__(self):
        return self.__class__, (self.experiment,)


class SpectrumCache(Cache):
    name = "intensity"  # not used as a key, but for logging and debugging

    def _key_from_row(self, row: "pd.Series") -> Any:
        return row["peptide_sequences"], row["precursor_charges"]

    def _preprocess_predictions(
        self, predictions: dict[str, "np.ndarray"]
    ) -> list[tuple["np.ndarray", "np.ndarray"]]:
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

    def __getitem__(self, key: tuple[bytes, int]) -> tuple["np.ndarray", "np.ndarray"]:
        return super().__getitem__(key)

    def __setitem__(
        self, key: tuple[bytes, int], value: tuple["np.ndarray", "np.ndarray"]
    ) -> None:
        return super().__setitem__(key, value)


class RTCache(Cache):
    name = "irt"

    def _key_from_row(self, row: "pd.Series") -> bytes:
        return row["peptide_sequences"]

    def __getitem__(self, key: bytes) -> float:
        return super().__getitem__(key)

    def __setitem__(self, key: bytes, value: float) -> None:
        return super().__setitem__(key, value)


class CCSCache(Cache):
    name = "ccs"

    def _key_from_row(self, row: "pd.Series") -> tuple[bytes, int]:
        return row["peptide_sequences"], row["precursor_charges"]

    def __getitem__(self, key: tuple[bytes, int]) -> float:
        return super().__getitem__(key)

    def __setitem__(self, key: tuple[bytes, int], value: float) -> None:
        return super().__setitem__(key, value)


class ByteStringKeyCache(Cache):
    """Base class for indices that use byte strings as keys."""

    @abstractmethod
    def _full_key(self, key: Any) -> bytes:
        """Convert the given key to a byte string key."""
        pass

    def __getitem__(self, key: Any) -> Any:
        full_key = self._full_key(key)
        return super().__getitem__(full_key)

    def __setitem__(self, key: Any, value: Any) -> None:
        full_key = self._full_key(key)
        super().__setitem__(full_key, value)


class ByteStringSpectrumCache(ByteStringKeyCache, SpectrumCache):
    def _full_key(self, key: tuple[bytes, int]) -> bytes:
        config = self.experiment.config
        return key[0] + bytes(
            f"_{key[1]}_{config.collision_energy}_{config.fragmentation_type}_{config.model_intensity.value}",
            "ascii",
        )


class ByteStringRTCache(ByteStringKeyCache, RTCache):
    def _full_key(self, key: bytes) -> bytes:
        config = self.experiment.config
        return key + bytes(config.model_irt.value, "ascii")


class ByteStringCCSCache(ByteStringKeyCache, CCSCache):
    def _full_key(self, key: tuple[bytes, int]) -> bytes:
        config = self.experiment.config
        return key[0] + bytes(f"_{key[1]}_{config.model_ccs.value}", "ascii")
