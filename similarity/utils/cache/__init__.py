from enum import Enum
from typing import TYPE_CHECKING
from . import diskcache
from ..abc import Index, IndexType

if TYPE_CHECKING:
    from ...experiment import Experiment


class NoCache(Index):
    """This type means that caching is not configured."""

    @classmethod
    def get_index(
        cls, index_type: IndexType, experiment: "Experiment"
    ) -> "Index | None":
        return None


NoCache.index_type = {
    IndexType.INTENSITY: NoCache,
    IndexType.IRT: NoCache,
    IndexType.CCS: NoCache,
}


class CacheType(Enum):
    DISKCACHE = diskcache.Index
    NONE = NoCache
