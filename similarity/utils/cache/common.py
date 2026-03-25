from abc import ABC, abstractmethod
from typing import Any, Iterable, TYPE_CHECKING
from ..abc import Cache as CacheABC

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np


class Cache(CacheABC):
    """Common base class for all cache indices."""

    def _preprocess_predictions(self, predictions: dict[str, "np.ndarray"]) -> Iterable:
        """
        Preprocess raw predictions from the model into the format expected by _write_to_cache.
        Should return an iterable of values to be cached, the same size as `inputs`.
        """
        return predictions[self.name].reshape(
            -1
        )  # default implementation for 1D predictions"


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

    def __getitem__(self, key: tuple[bytes, int]) -> "np.ndarray":
        return super().__getitem__(key)

    def __setitem__(self, key: tuple[bytes, int], value: "np.ndarray") -> None:
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
