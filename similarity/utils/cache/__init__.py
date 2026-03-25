from enum import Enum
from typing import TYPE_CHECKING
from . import diskcache
from ..abc import Cache, IndexType

if TYPE_CHECKING:
    from ...experiment import Experiment


class NoCache(Cache):
    """This type means that caching is not configured."""

    @classmethod
    def get_index(
        cls, index_type: IndexType, experiment: "Experiment"
    ) -> "Cache | None":
        return None


NoCache.index_type = {
    IndexType.INTENSITY: NoCache,
    IndexType.IRT: NoCache,
    IndexType.CCS: NoCache,
}


class CacheType(Enum):
    DISKCACHE = diskcache.Index
    NONE = NoCache
