from contextlib import AbstractContextManager
import queue
import diskcache
from abc import ABC
from typing import Any, Iterable, TYPE_CHECKING
import logging
import numpy as np
from pandas import DataFrame
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


class Index(ByteStringKeyCache, diskcache.Index, ABC):
    """Index for predicted spectra. Uses experiment config to add collision energy, charge and model info to the key."""

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

    def __init__(self, experiment: "Experiment"):
        ByteStringKeyCache.__init__(self, experiment)
        self._save_queue = queue.Queue()

        # not starting the thread here because `Index` can also be instantiated by workers
        # not calling super().__init__() because it would reassign self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        full_key = self._full_key(key)
        return full_key in self.cache  # direct check on Index doesn't work

    def close(self):
        self.wait()
        self._cache.close()

    def transact_to_cache(self, inputs: DataFrame, predictions: Iterable) -> None:
        with self.transact():
            self.write_to_cache(inputs, predictions)

    def transact(self) -> AbstractContextManager[Any, bool | None]:
        return super().transact()


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
