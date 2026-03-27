from pandas import DataFrame, Series
import redis
from dataclasses import asdict
import logging
from typing import TYPE_CHECKING, Any, Iterable
from abc import abstractmethod, ABC
import numpy as np
from ..abc import IndexType
from tqdm import trange

from .common import (
    ByteStringKeyCache,
    ByteStringSpectrumCache,
    ByteStringRTCache,
    ByteStringCCSCache,
)

if TYPE_CHECKING:
    from ...experiment import Experiment
    import redis.client
    import pandas as pd

logger = logging.getLogger(__name__)


class RedisAgent(ByteStringKeyCache, ABC):
    _client: "redis.Redis | redis.client.Pipeline"

    @staticmethod
    @abstractmethod
    def encode_value(value: Any) -> bytes:
        pass

    @staticmethod
    @abstractmethod
    def decode_value(value: bytes) -> Any:
        pass

    def __getitem__(self, key: Any) -> Any:
        value = self._client.get(self._full_key(key))
        if value is None:
            raise KeyError(f"Key {key} not found in Redis cache")
        return self.decode_value(value)

    def __setitem__(self, key: Any, value: Any) -> None:
        full_key = self._full_key(key)
        self._client.set(full_key, self.encode_value(value))

    def __contains__(self, key: Any) -> bool:
        full_key = self._full_key(key)
        return self._client.exists(full_key) > 0

    def __len__(self) -> int:
        return self._client.dbsize()


class RedisPipeline(RedisAgent):
    _client: "redis.client.Pipeline"

    def __init__(self, parent: "RedisCache"):
        self.name = parent.name
        super().__init__(parent.experiment)
        self.parent = parent
        self._client = parent._client.pipeline()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()

    def transact(self) -> "RedisPipeline":
        return self

    def _full_key(self, key: Any) -> bytes:
        return self.parent._full_key(key)

    def _key_from_row(self, row: Series) -> Any:
        return self.parent._key_from_row(row)

    def close(self):
        self.execute()

    def decode_value(self, value: bytes) -> Any:
        return self.parent.decode_value(value)

    def encode_value(self, value: Any) -> bytes:
        return self.parent.encode_value(value)

    def transact_to_cache(self, inputs: DataFrame, predictions: Iterable) -> None:
        return self.write_to_cache(inputs, predictions)

    def execute(self):
        return self._client.execute()


class RedisCache(RedisAgent):
    _client: "redis.Redis"
    batch_size: int = 50000

    def __init__(self, experiment: "Experiment"):
        super().__init__(experiment)
        cconf = experiment.config.cache_conf
        assert cconf is not None, "Cache configuration must be provided for RedisCache"
        kwargs = asdict(cconf)
        kwargs.pop("cache_properties", None)
        self._client = redis.Redis(**kwargs, decode_responses=False)

    def transact(self) -> "RedisPipeline":
        return RedisPipeline(self)

    def transact_to_cache(self, inputs: "pd.DataFrame", predictions: Iterable) -> None:
        with self.transact() as pipe:
            pipe.write_to_cache(inputs, predictions)

    def fill_from_cache(self, inputs: DataFrame, output: np.ndarray) -> None:
        # override default behavior to use a pipeline for batch retrieval
        bsize = self.batch_size
        nb = (len(inputs) + bsize - 1) // bsize
        logger.debug("Filling from Redis cache in %d batches of size %d", nb, bsize)
        for batch in trange(nb, desc=f"Loading {self.name} from cache", unit="batch"):
            with self.transact() as pipe:
                for _, row in inputs.iloc[
                    batch * bsize : (batch + 1) * bsize
                ].iterrows():
                    key = self._key_from_row(row)
                    pipe._client.get(pipe._full_key(key))
                results = pipe.execute()
                for i, value in enumerate(results, start=batch * bsize):
                    if value is not None:
                        output[i] = self.decode_value(value)
                    else:
                        output[i] = np.nan

    def close(self):
        self.wait()
        self._client.close()


class RedisSpectrumCache(ByteStringSpectrumCache, RedisCache):
    def encode_value(self, value: tuple[np.ndarray, np.ndarray]) -> bytes:
        mzs, intensities = value
        return mzs.tobytes() + intensities.tobytes()

    def decode_value(self, value: bytes) -> tuple[np.ndarray, np.ndarray]:
        arr = np.frombuffer(value, dtype=np.float32)
        half = len(arr) // 2
        mzs = arr[:half]
        intensities = arr[half:]
        return mzs, intensities


class FloatRedisCache(RedisCache):
    def encode_value(self, value: float) -> bytes:
        return str(value).encode("ascii")

    def decode_value(self, value: bytes) -> float:
        return float(value)


class RedisRTCache(ByteStringRTCache, FloatRedisCache):
    pass


class RedisCCSCache(ByteStringCCSCache, FloatRedisCache):
    pass


RedisCache.index_type = {
    IndexType.INTENSITY: RedisSpectrumCache,
    IndexType.IRT: RedisRTCache,
    IndexType.CCS: RedisCCSCache,
}
