import redis
from dataclasses import asdict
import logging
from typing import TYPE_CHECKING, Any, Iterable
from abc import abstractmethod, ABC
import numpy as np
from ..abc import IndexType

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

    @abstractmethod
    @staticmethod
    def encode_value(value: Any) -> bytes | int | float:
        pass

    @abstractmethod
    @staticmethod
    def decode_value(value: bytes | int) -> Any:
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
        super().__init__(parent.experiment)
        self._client = parent._client.pipeline()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self._client.execute()

    def transact(self) -> "RedisPipeline":
        return self


class RedisCache(RedisAgent):
    _client: "redis.Redis"

    def __init__(self, experiment: "Experiment"):
        super().__init__(experiment)
        cconf = experiment.config.cache_conf
        assert cconf is not None, "Cache configuration must be provided for RedisCache"
        self._client = redis.Redis(**asdict(cconf), decode_responses=False)

    def close(self):
        self._client.close()

    def transact(self) -> "RedisPipeline":
        return RedisPipeline(self)

    def transact_to_cache(self, inputs: "pd.DataFrame", predictions: Iterable) -> None:
        with self.transact() as pipe:
            pipe.write_to_cache(inputs, predictions)

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


class FloatRedisCache(ByteStringKeyCache, RedisCache):
    def encode_value(self, value: float) -> bytes:
        return str(value).encode("ascii")

    def decode_value(self, value: bytes) -> float:
        return float(value)


class RedisRTCache(FloatRedisCache, ByteStringRTCache):
    pass


class RedisCCSCache(FloatRedisCache, ByteStringCCSCache):
    pass


RedisCache.index_type = {
    IndexType.INTENSITY: RedisSpectrumCache,
    IndexType.IRT: RedisRTCache,
    IndexType.CCS: RedisCCSCache,
}
